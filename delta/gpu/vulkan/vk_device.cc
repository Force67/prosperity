/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_device.h"

#include "gpu/ps4/gcn/gcn_translate.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_backend.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace gpu::vk {

bool g_has_device_fault = false;

// Ask the driver what the GPU actually faulted on (VK_EXT_device_fault).
// Prints once per process — every later DEVICE_LOST is collateral of the first.
void ReportDeviceFault(VkDevice device) {
  static bool reported = false;
  if (reported || !g_has_device_fault)
    return;
  reported = true;
  auto p_get_fault = (PFN_vkGetDeviceFaultInfoEXT)vkGetDeviceProcAddr(
      device, "vkGetDeviceFaultInfoEXT");
  if (!p_get_fault)
    return;
  VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
  if (p_get_fault(device, &counts, nullptr) < 0)
    return;
  std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
  std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
  VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
  info.pAddressInfos = addrs.data();
  info.pVendorInfos = vendors.data();
  counts.vendorBinarySize = 0;
  p_get_fault(device, &counts, &info);
  std::fprintf(stderr, "[gpuvk] device fault: '%s' addrs=%u vendor=%u\n",
               info.description, counts.addressInfoCount,
               counts.vendorInfoCount);
  for (const auto& a : addrs)
    std::fprintf(stderr, "[gpuvk]   fault addr type=%d va=%#llx prec=%#llx\n",
                 (int)a.addressType, (unsigned long long)a.reportedAddress,
                 (unsigned long long)a.addressPrecision);
  for (const auto& v : vendors)
    std::fprintf(stderr, "[gpuvk]   vendor '%s' code=%#llx data=%#llx\n",
                 v.description, (unsigned long long)v.vendorFaultCode,
                 (unsigned long long)v.vendorFaultData);
}

uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return 0;
}

// Pick a memory type matching `pref` if any exists, else fall back to `req`.
// Used for the readback buffer: the CPU READS it every frame (the scanout
// flip), so it must be HOST_CACHED -- reading from the default HOST_COHERENT
// (write-combined, uncached) staging memory byte-by-byte is ~30x slower and was
// dominating frame time. CACHED+COHERENT (present on desktop GPUs) needs no
// manual invalidate.
uint32_t FindMemoryTypePref(uint32_t type_bits,
                            VkMemoryPropertyFlags pref,
                            VkMemoryPropertyFlags req) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & pref) == pref)
      return i;
  return FindMemoryType(type_bits, req);
}

void ImageBarrier(VkCommandBuffer c,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_a,
                  VkAccessFlags dst_a,
                  uint32_t layers,
                  uint32_t mip_levels) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, layers};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b);
}

// Transition a depth image (aspect = DEPTH) between layouts.
void DepthBarrier(VkCommandBuffer c,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_a,
                  VkAccessFlags dst_a) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b);
}

bool CreateDevice() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  app.pApplicationName = "prosperity-gpu";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  VKOK(vkCreateInstance(&ic, nullptr, &g_dev.instance));

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(g_dev.instance, &n, nullptr);
  if (!n) {
    std::fprintf(stderr, "[gpuvk] no device\n");
    return false;
  }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(g_dev.instance, &n, devs.data());

  // Prefer a real GPU over the llvmpipe software rasteriser (reported as type
  // CPU): discrete > integrated > virtual > CPU. The loader can enumerate both
  // a discrete GPU and llvmpipe on the same box, so picking devs[0] blindly may
  // land on software. DELTA_VK_GPU=<name-substring> forces a specific device.
  const char* want = std::getenv("DELTA_VK_GPU");
  int best = -1;
  g_dev.phys = VK_NULL_HANDLE;
  for (VkPhysicalDevice d : devs) {
    uint32_t dqn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &dqn, nullptr);
    std::vector<VkQueueFamilyProperties> dq(dqn);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &dqn, dq.data());
    bool gfx = false;
    for (auto& q : dq)
      if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        gfx = true;
        break;
      }
    if (!gfx)
      continue;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    int score;
    switch (p.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score = 4;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score = 3;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score = 2;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        score = 0;
        break;  // llvmpipe
      default:
        score = 1;
        break;
    }
    if (want && std::strstr(p.deviceName, want))
      score = 100;
    if (score > best) {
      best = score;
      g_dev.phys = d;
    }
  }
  if (g_dev.phys == VK_NULL_HANDLE) {
    std::fprintf(stderr, "[gpuvk] no gfx device\n");
    return false;
  }

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &qn, qprops.data());
  bool found = false;
  for (uint32_t i = 0; i < qn; i++)
    if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      g_dev.qfam = i;
      found = true;
      break;
    }
  if (!found) {
    std::fprintf(stderr, "[gpuvk] no gfx queue\n");
    return false;
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qc.queueFamilyIndex = g_dev.qfam;
  qc.queueCount = 1;
  qc.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan12Features avail12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 avail2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  avail2.pNext = &avail12;
  vkGetPhysicalDeviceFeatures2(g_dev.phys, &avail2);
  VkPhysicalDeviceVulkan12Features f12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.samplerMirrorClampToEdge = avail12.samplerMirrorClampToEdge;
  VkPhysicalDeviceVulkan13Features f13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.pNext = &f12;
  f13.dynamicRendering = VK_TRUE;
  VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dc.pNext = &f13;
  dc.queueCreateInfoCount = 1;
  dc.pQueueCreateInfos = &qc;
  // VK_EXT_device_fault: on DEVICE_LOST, vkGetDeviceFaultInfoEXT reports what
  // the GPU actually faulted on (page fault address etc.) — keep it enabled,
  // it costs nothing until a fault is queried.
  const char* dev_exts[1] = {};
  {
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> eprops(en);
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en,
                                         eprops.data());
    for (const auto& ep : eprops)
      if (!std::strcmp(ep.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME))
        g_has_device_fault = true;
  }
  static VkPhysicalDeviceFaultFeaturesEXT fault_feat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
  if (g_has_device_fault) {
    fault_feat.deviceFault = VK_TRUE;
    fault_feat.pNext = f13.pNext;
    f13.pNext = &fault_feat;
    dev_exts[0] = VK_EXT_DEVICE_FAULT_EXTENSION_NAME;
    dc.enabledExtensionCount = 1;
    dc.ppEnabledExtensionNames = dev_exts;
  }
  // robustBufferAccess makes out-of-bounds storage-buffer loads/stores safe
  // (return 0 / drop the write) so the compute path can't corrupt memory on a
  // miscomputed index.
  VkPhysicalDeviceFeatures want_feat{};
  if (avail2.features.robustBufferAccess)
    want_feat.robustBufferAccess = VK_TRUE;
  if (avail2.features.samplerAnisotropy)
    want_feat.samplerAnisotropy = VK_TRUE;
  if (avail2.features.geometryShader)
    want_feat.geometryShader = VK_TRUE;
  if (avail2.features.shaderStorageImageWriteWithoutFormat)
    want_feat.shaderStorageImageWriteWithoutFormat = VK_TRUE;
  g_dev.sampler_anisotropy = want_feat.samplerAnisotropy;
  g_dev.sampler_mirror_clamp = f12.samplerMirrorClampToEdge;
  g_dev.geometry_shader = want_feat.geometryShader;
  g_dev.storage_image_write_without_format =
      want_feat.shaderStorageImageWriteWithoutFormat;
  dc.pEnabledFeatures = &want_feat;
  VKOK(vkCreateDevice(g_dev.phys, &dc, nullptr, &g_dev.device));
  vkGetDeviceQueue(g_dev.device, g_dev.qfam, 0, &g_dev.queue);

  g_cmd_begin_rendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(
      g_dev.device, "vkCmdBeginRendering");
  g_cmd_end_rendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(
      g_dev.device, "vkCmdEndRendering");
  if (!g_cmd_begin_rendering) {
    g_cmd_begin_rendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(
        g_dev.device, "vkCmdBeginRenderingKHR");
    g_cmd_end_rendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(
        g_dev.device, "vkCmdEndRenderingKHR");
  }
  if (!g_cmd_begin_rendering) {
    std::fprintf(stderr, "[gpuvk] no dynamic rendering\n");
    return false;
  }

  VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pc.queueFamilyIndex = g_dev.qfam;
  VKOK(vkCreateCommandPool(g_dev.device, &pc, nullptr, &g_dev.pool));
  VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VKOK(vkCreateFence(g_dev.device, &fc, nullptr,
                     &g_dev.fence));  // aux submits only
  if (!CreateFrameSlots())
    return false;

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_dev.phys, &props);
  g_dev.max_cs_resources = std::min(
      {gcn::kMaxCsResources, props.limits.maxPerStageDescriptorStorageBuffers,
       props.limits.maxDescriptorSetStorageBuffers});
  g_dev.max_storage_buffer_range = props.limits.maxStorageBufferRange;
  std::fprintf(stderr, "[gpuvk] device: %s\n", props.deviceName);
  if (!CreateUploadRings(props))
    return false;
  return true;
}

VkShaderModule MakeModule(const uint32_t* spv, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = spv;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(g_dev.device, &ci, nullptr, &m);
  return m;
}

VkShaderModule MakeModuleVec(const std::vector<uint32_t>& spv) {
  return MakeModule(spv.data(), spv.size() * 4);
}

}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

bool Init(Renderer& renderer) {
  if (renderer.available())
    return true;
  // Create the device from a clean host thread: Init() is reached on a FEX
  // guest thread (guest stack / TLS), where the NVIDIA ICD's
  // vk_icdGetInstanceProcAddr silently fails and enumeration falls back to
  // llvmpipe -- a ~30ms/frame software rasteriser on a box with a real GPU.
  // llvmpipe never cared, so this is behaviour-neutral for pure-software runs.
  bool ok = false;
  std::thread init_thread([&ok] { ok = CreateDevice(); });
  init_thread.join();
  if (!ok) {
    std::fprintf(stderr, "[gpuvk] headless Vulkan unavailable; gpu disabled\n");
    return false;
  }
  g_dev.ready = true;
  renderer.state = &g_backend;
  return true;
}

}  // namespace gpu::rhi
