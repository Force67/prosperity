/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_device.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <base/logging.h>

#include <sys/stat.h>
#include <unistd.h>

#include "gpu/gcn/gcn_translate.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_backend.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/gpu_perf.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_trace.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utl/options.h>

namespace {
DELTA_OPTION(const char*, kVkGpu, "DELTA_VK_GPU", nullptr);
}  // namespace

namespace gpu::vk {

namespace {
DELTA_OPTION(bool, kShaderCacheOn, "DELTA_GPU_SHADER_CACHE", true);
DELTA_OPTION(const char*,
             kShaderCacheDirOpt,
             "DELTA_GPU_SHADER_CACHE_DIR",
             nullptr);

// Where the driver's pipeline cache blob lives. Same convention as the SPIR-V
// cache (DELTA_GPU_SHADER_CACHE_DIR, else $XDG_CACHE_HOME/ps4delta, else
// ~/.cache/ps4delta), and disabled by the same DELTA_GPU_SHADER_CACHE=0.
std::string PipelineCachePath() {
  static const std::string path = [] {
    if (!kShaderCacheOn)
      return std::string();
    std::string d;
    if (kShaderCacheDirOpt && *kShaderCacheDirOpt)
      d = kShaderCacheDirOpt;
    else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
      d = std::string(xdg) + "/ps4delta";
    else if (const char* home = std::getenv("HOME"); home && *home)
      d = std::string(home) + "/.cache/ps4delta";
    else
      return std::string();
    for (size_t i = 1; i <= d.size(); i++)
      if (i == d.size() || d[i] == '/')
        ::mkdir(d.substr(0, i).c_str(), 0755);
    return d + "/pipeline.bin";
  }();
  return path;
}

std::vector<u8> ReadPipelineCacheBlob() {
  std::vector<u8> out;
  const std::string p = PipelineCachePath();
  if (p.empty())
    return out;
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f)
    return out;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    if (std::fread(out.data(), 1, out.size(), f) != out.size())
      out.clear();
  }
  std::fclose(f);
  return out;
}
}  // namespace

void SavePipelineCache(bool force) {
  static size_t last_size = 0;
  static u64 last_write_ns = 0;
  const std::string p = PipelineCachePath();
  if (p.empty() || g_dev.pipeline_cache == VK_NULL_HANDLE)
    return;
  // A level's worth of pipelines is built in one burst, and every one of them
  // would otherwise re-serialize and rewrite the WHOLE blob on the submit
  // thread -- turning the save into a second source of the hitch it exists to
  // remove. One write a second keeps the burst cheap; what it drops is rebuilt
  // and saved on the next run.
  const u64 now = NowNs();
  if (!force && last_write_ns && now - last_write_ns < 1000000000ull)
    return;
  size_t size = 0;
  if (vkGetPipelineCacheData(g_dev.device, g_dev.pipeline_cache, &size,
                             nullptr) != VK_SUCCESS ||
      !size)
    return;
  // Only write when the driver has actually added something.
  if (!force && size == last_size)
    return;
  std::vector<u8> blob(size);
  if (vkGetPipelineCacheData(g_dev.device, g_dev.pipeline_cache, &size,
                             blob.data()) != VK_SUCCESS)
    return;
  last_size = size;
  last_write_ns = now;
  char tmp[512];
  std::snprintf(tmp, sizeof(tmp), "%s.%d.tmp", p.c_str(), (int)::getpid());
  FILE* f = std::fopen(tmp, "wb");
  if (!f)
    return;
  const bool ok = std::fwrite(blob.data(), 1, size, f) == size;
  std::fclose(f);
  if (ok)
    ::rename(tmp, p.c_str());
  else
    ::unlink(tmp);
}

namespace {

VkPipelineStageFlags StageForAccess(VkAccessFlags access, bool source) {
  VkPipelineStageFlags stages = 0;
  if (access & (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT))
    stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (access & (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT))
    stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (access & (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT))
    stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  // Every stage that can express a shader access on this queue, not just
  // fragment: image descriptors are fragment-only today, but a barrier helper
  // that silently under-covers vertex fetch or compute the day one appears is
  // exactly the class of bug that is unfindable later. The extra stage bits
  // are free when no such access exists in the frame.
  if (access & (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT))
    stages |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  return stages   ? stages
         : source ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                  : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
}

}  // namespace

// Ask the driver what the GPU actually faulted on (VK_EXT_device_fault).
// Prints once per device -- every later DEVICE_LOST is collateral of the first.
void ReportDeviceFault(DeviceState& device) {
  if (device.device_fault_reported || !device.device_fault_available)
    return;
  device.device_fault_reported = true;
  auto p_get_fault = (PFN_vkGetDeviceFaultInfoEXT)vkGetDeviceProcAddr(
      device.device, "vkGetDeviceFaultInfoEXT");
  if (!p_get_fault)
    return;
  VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
  if (p_get_fault(device.device, &counts, nullptr) < 0)
    return;
  std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
  std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
  VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
  info.pAddressInfos = addrs.data();
  info.pVendorInfos = vendors.data();
  counts.vendorBinarySize = 0;
  p_get_fault(device.device, &counts, &info);
  BASE_LOGI("gpuvk", "device fault: '{}' addrs={} vendor={}",
            info.description, counts.addressInfoCount,
            counts.vendorInfoCount);
  for (const auto& a : addrs)
    BASE_LOGI("gpuvk", "  fault addr type={} va={:#x} prec={:#x}",
              (int)a.addressType, (unsigned long long)a.reportedAddress,
              (unsigned long long)a.addressPrecision);
  for (const auto& v : vendors)
    BASE_LOGI("gpuvk", "  vendor '{}' code={:#x} data={:#x}",
              v.description, (unsigned long long)v.vendorFaultCode,
              (unsigned long long)v.vendorFaultData);
}

u32 FindMemoryType(u32 type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &mp);
  for (u32 i = 0; i < mp.memoryTypeCount; i++)
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
u32 FindMemoryTypePref(u32 type_bits,
                            VkMemoryPropertyFlags pref,
                            VkMemoryPropertyFlags req) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &mp);
  for (u32 i = 0; i < mp.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & pref) == pref)
      return i;
  return FindMemoryType(type_bits, req);
}

VkAccessFlags ColorImageAccess(VkImageLayout layout) {
  switch (layout) {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT;
    case VK_IMAGE_LAYOUT_GENERAL:
      return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
    default:
      return 0;
  }
}

void ImageBarrier(VkCommandBuffer c,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_a,
                  VkAccessFlags dst_a,
                  u32 layers,
                  u32 mip_levels) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, layers};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  if (trace::Recording())
    trace::RecordBarrier("color", img, from, to, src_a, dst_a);
  vkCmdPipelineBarrier(c, StageForAccess(src_a, true),
                       StageForAccess(dst_a, false), 0, 0, nullptr, 0, nullptr,
                       1, &b);
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
  if (trace::Recording())
    trace::RecordBarrier("depth", img, from, to, src_a, dst_a);
  vkCmdPipelineBarrier(c, StageForAccess(src_a, true),
                       StageForAccess(dst_a, false), 0, 0, nullptr, 0, nullptr,
                       1, &b);
}

void StencilBarrier(VkCommandBuffer c,
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
  b.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  if (trace::Recording())
    trace::RecordBarrier("stencil", img, from, to, src_a, dst_a);
  vkCmdPipelineBarrier(c, StageForAccess(src_a, true),
                       StageForAccess(dst_a, false), 0, 0, nullptr, 0, nullptr,
                       1, &b);
}

bool CreateDevice() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  app.pApplicationName = "prosperity-gpu";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  // Object names + command labels for capture tools (vk_debug). Gated on a
  // consumer actually listening -- the loader always advertises the extension,
  // but formatting labels for nobody costs real frame time.
  bool debug_utils = false;
  if (WantDebugUtils() || trace::WantValidation()) {
    u32 ext_n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_n, nullptr);
    std::vector<VkExtensionProperties> exts(ext_n);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_n, exts.data());
    for (const auto& e : exts)
      if (!std::strcmp(e.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        debug_utils = true;
  }
  const char* instance_exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  if (debug_utils) {
    ic.enabledExtensionCount = 1;
    ic.ppEnabledExtensionNames = instance_exts;
  }
  // DELTA_GPU_VALIDATE=1: the Khronos validation layers, with their messages
  // routed into the frame capture next to the draw that provoked them. Off by
  // default -- the layers cost real frame time and the loader only finds them
  // when the layer path is on the environment.
  const char* validation_layer = trace::ValidationLayerName();
  if (trace::WantValidation()) {
    u32 layer_n = 0;
    vkEnumerateInstanceLayerProperties(&layer_n, nullptr);
    std::vector<VkLayerProperties> layers(layer_n);
    vkEnumerateInstanceLayerProperties(&layer_n, layers.data());
    bool found = false;
    for (const auto& l : layers)
      found |= !std::strcmp(l.layerName, validation_layer);
    if (found) {
      ic.enabledLayerCount = 1;
      ic.ppEnabledLayerNames = &validation_layer;
      if (trace::WantSyncValidation()) {
        static const VkValidationFeatureEnableEXT sync_feat[1] = {
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
        static VkValidationFeaturesEXT vf{
            VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
        vf.enabledValidationFeatureCount = 1;
        vf.pEnabledValidationFeatures = sync_feat;
        vf.pNext = ic.pNext;
        ic.pNext = &vf;
      }
    } else {
      BASE_LOGI("vkval", "{} not available on this loader",
                validation_layer);
    }
  }
  VKOK(vkCreateInstance(&ic, nullptr, &g_dev.instance));
  InitDebugUtils(g_dev.instance, debug_utils);
  trace::InstallValidationMessenger(g_dev.instance);

  u32 n = 0;
  vkEnumeratePhysicalDevices(g_dev.instance, &n, nullptr);
  if (!n) {
    BASE_LOGI("gpuvk", "no device");
    return false;
  }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(g_dev.instance, &n, devs.data());

  // Prefer a real GPU over the llvmpipe software rasteriser (reported as type
  // CPU): discrete > integrated > virtual > CPU. The loader can enumerate both
  // a discrete GPU and llvmpipe on the same box, so picking devs[0] blindly may
  // land on software. DELTA_VK_GPU=<name-substring> forces a specific device.
  const char* want = kVkGpu;
  int best = -1;
  g_dev.phys = VK_NULL_HANDLE;
  for (VkPhysicalDevice d : devs) {
    u32 dqn = 0;
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
    BASE_LOGI("gpuvk", "no gfx device");
    return false;
  }

  u32 qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &qn, qprops.data());
  bool found = false;
  for (u32 i = 0; i < qn; i++)
    if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      g_dev.qfam = i;
      found = true;
      break;
    }
  if (!found) {
    BASE_LOGI("gpuvk", "no gfx queue");
    return false;
  }
  g_dev.timestamp_valid_bits = qprops[g_dev.qfam].timestampValidBits;

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
  f12.separateDepthStencilLayouts = avail12.separateDepthStencilLayouts;
  f12.bufferDeviceAddress = avail12.bufferDeviceAddress && avail2.features.shaderInt64;
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
  // VK_EXT_external_memory_host lets a buffer be backed by guest pages DIRECTLY,
  // so a compute dispatch reading guest memory needs no staging copy in and no
  // writeback out (see vk_compute.cc, DELTA_GPU_CSIMPORT).
  // VK_KHR_fragment_shader_barycentric supplies the per-vertex attribute values
  // a pixel shader needs for v_interp_mov_f32's P10/P20 parameters (the deltas
  // P1-P0 and P2-P0), which an interpolated input cannot express.
  const char* dev_exts[3] = {};
  u32 dev_ext_count = 0;
  {
    u32 en = 0;
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> eprops(en);
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en,
                                         eprops.data());
    for (const auto& ep : eprops) {
      if (!std::strcmp(ep.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME))
        g_dev.device_fault_available = true;
      if (!std::strcmp(ep.extensionName,
                       VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
        g_dev.host_import_available = true;
      if (!std::strcmp(ep.extensionName,
                       VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME))
        g_dev.barycentric_available = true;
    }
  }
  if (g_dev.host_import_available) {
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &hp;
    vkGetPhysicalDeviceProperties2(g_dev.phys, &p2);
    g_dev.host_import_align =
        (size_t)hp.minImportedHostPointerAlignment;
    g_dev.storage_buffer_offset_align =
        (size_t)p2.properties.limits.minStorageBufferOffsetAlignment;
    if (!g_dev.storage_buffer_offset_align)
      g_dev.storage_buffer_offset_align = 1;
    dev_exts[dev_ext_count++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
  }
  VkPhysicalDeviceFaultFeaturesEXT fault_feat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
  if (g_dev.device_fault_available) {
    fault_feat.deviceFault = VK_TRUE;
    fault_feat.pNext = f13.pNext;
    f13.pNext = &fault_feat;
    dev_exts[dev_ext_count++] = VK_EXT_DEVICE_FAULT_EXTENSION_NAME;
  }
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_feat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  if (g_dev.barycentric_available) {
    bary_feat.fragmentShaderBarycentric = VK_TRUE;
    bary_feat.pNext = f13.pNext;
    f13.pNext = &bary_feat;
    dev_exts[dev_ext_count++] = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME;
  }
  if (dev_ext_count) {
    dc.enabledExtensionCount = dev_ext_count;
    dc.ppEnabledExtensionNames = dev_exts;
  }
  // robustBufferAccess makes out-of-bounds storage-buffer loads/stores safe
  // (return 0 / drop the write) so the compute path can't corrupt memory on a
  // miscomputed index.
  VkPhysicalDeviceFeatures want_feat{};
  want_feat.shaderInt64 = f12.bufferDeviceAddress;
  g_dev.buffer_device_address = f12.bufferDeviceAddress;
  want_feat.textureCompressionBC = avail2.features.textureCompressionBC;
  if (avail2.features.robustBufferAccess)
    want_feat.robustBufferAccess = VK_TRUE;
  if (avail2.features.depthClamp)
    want_feat.depthClamp = VK_TRUE;
  if (avail2.features.samplerAnisotropy)
    want_feat.samplerAnisotropy = VK_TRUE;
  if (avail2.features.geometryShader)
    want_feat.geometryShader = VK_TRUE;
  // Without independentBlend, "all elements of pAttachments must be identical"
  // -- so a G-buffer pass that blends its targets differently (SotC disables
  // blending on its integer planes and accumulates additively on the others)
  // gets undefined behaviour across EVERY attachment, not just the odd one out.
  if (avail2.features.independentBlend)
    want_feat.independentBlend = VK_TRUE;
  // Guest blend state names dual-source factors (Astro's intro card uses
  // SRC1_COLOR); a pipeline that carries one without this feature is invalid.
  // The shader model's Index-1 output is still not translated, so the second
  // source's values are whatever the implementation hands out -- valid, not
  // exact.
  want_feat.dualSrcBlend = avail2.features.dualSrcBlend;
  if (avail2.features.shaderStorageImageWriteWithoutFormat)
    want_feat.shaderStorageImageWriteWithoutFormat = VK_TRUE;
  // A recompiled VERTEX shader can index a guest buffer by hand, which becomes
  // a storage buffer SPIR-V considers writable. Declaring one is only legal
  // with this feature on; without it the module was used anyway and the access
  // is undefined (VUID-RuntimeSpirv-NonWritable-06341).
  if (avail2.features.vertexPipelineStoresAndAtomics)
    want_feat.vertexPipelineStoresAndAtomics = VK_TRUE;
  if (avail2.features.fragmentStoresAndAtomics)
    want_feat.fragmentStoresAndAtomics = VK_TRUE;
  g_dev.sampler_anisotropy = want_feat.samplerAnisotropy;
  g_dev.independent_blend = want_feat.independentBlend;
  g_dev.sampler_mirror_clamp = f12.samplerMirrorClampToEdge;
  g_dev.geometry_shader = want_feat.geometryShader;
  g_dev.storage_image_write_without_format =
      want_feat.shaderStorageImageWriteWithoutFormat;
  dc.pEnabledFeatures = &want_feat;
  VKOK(vkCreateDevice(g_dev.phys, &dc, nullptr, &g_dev.device));
  vkGetDeviceQueue(g_dev.device, g_dev.qfam, 0, &g_dev.queue);
  // Seed the driver's pipeline cache from disk. Without this every run
  // recompiles every pipeline from scratch, which on SotC is several hundred.
  // A blob from another driver/device is rejected by the driver itself (it
  // checks its own header), so a stale file costs nothing but the read.
  std::vector<u8> cache_blob = ReadPipelineCacheBlob();
  VkPipelineCacheCreateInfo pipeline_cache_info{
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  pipeline_cache_info.initialDataSize = cache_blob.size();
  pipeline_cache_info.pInitialData =
      cache_blob.empty() ? nullptr : cache_blob.data();
  if (vkCreatePipelineCache(g_dev.device, &pipeline_cache_info, nullptr,
                            &g_dev.pipeline_cache) != VK_SUCCESS)
    g_dev.pipeline_cache = VK_NULL_HANDLE;

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
    BASE_LOGI("gpuvk", "no dynamic rendering");
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
  g_dev.timestamp_period = props.limits.timestampPeriod;
  g_dev.max_cs_resources = std::min(
      {gcn::kMaxCsResources, props.limits.maxPerStageDescriptorStorageBuffers,
       props.limits.maxDescriptorSetStorageBuffers});
  g_dev.max_storage_buffer_range = props.limits.maxStorageBufferRange;
  BASE_LOGI("gpuvk", "device: {}", props.deviceName);
  if (!CreateUploadRings(props))
    return false;
  return true;
}

VkShaderModule MakeModule(const u32* spv, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = spv;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(g_dev.device, &ci, nullptr, &m);
  return m;
}

VkShaderModule MakeModuleVec(const std::vector<u32>& spv) {
  return MakeModule(spv.data(), spv.size() * 4);
}

}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

bool Init(Renderer& renderer) {
  if (renderer.available())
    return true;
  // A partial creation or a disabled live device cannot safely be initialized
  // over: its pools and caches still contain handles from that attempt. A clean
  // failure before instance creation remains retryable on the first submit.
  if (g_dev.ready || g_dev.instance)
    return false;
  // Create the device from a clean host thread: Init() is reached on a FEX
  // guest thread (guest stack / TLS), where the NVIDIA ICD's
  // vk_icdGetInstanceProcAddr silently fails and enumeration falls back to
  // llvmpipe -- a ~30ms/frame software rasteriser on a box with a real GPU.
  // llvmpipe never cared, so this is behaviour-neutral for pure-software runs.
  bool ok = false;
  std::thread init_thread([&ok] { ok = CreateDevice(); });
  init_thread.join();
  if (!ok) {
    BASE_LOGI("gpuvk", "headless Vulkan unavailable; gpu disabled");
    return false;
  }
  g_dev.ready = true;
  // Registered after all static initialization, so the worker is joined before
  // BackendState and gfx translation-unit globals begin destruction.
  std::atexit([] { g_backend.presenter.Stop(); });
  renderer.state = &g_backend;
  return true;
}

}  // namespace gpu::rhi
