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
#include "gpu/render/backend.h"
#include "gpu/render/renderer.h"
#include "gpu/vulkan/vk_backend.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/gpu_perf.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_rhi.h"
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
DELTA_OPTION(bool, kCheckpoints, "DELTA_GPU_CHECKPOINTS", false);
bool g_checkpoints_available = false;
PFN_vkCmdSetCheckpointNV g_set_checkpoint = nullptr;
DELTA_OPTION(bool, kShaderCacheOn, "DELTA_GPU_SHADER_CACHE", true);
DELTA_OPTION(const char*,
             kShaderCacheDirOpt,
             "DELTA_GPU_SHADER_CACHE_DIR",
             nullptr);

// Where the driver's pipeline cache blob lives. Same convention as the SPIR-V
// cache (DELTA_GPU_SHADER_CACHE_DIR, else $XDG_CACHE_HOME/ps4delta, else
// ~/.cache/ps4delta), and disabled by the same DELTA_GPU_SHADER_CACHE=0.
std::string PipelineCacheDir() {
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
    return d;
  }();
  return path;
}

std::string PipelineCachePath() {
  const std::string& d = PipelineCacheDir();
  return d.empty() ? d : d + "/pipeline.bin";
}

}  // namespace

namespace {
u64 g_last_pipeline_build_ns = 0;
}  // namespace

void NotePipelineBuilt() {
  g_last_pipeline_build_ns = NowNs();
}

void SavePipelineCache(bool force) {
  static size_t last_size = 0;
  static u64 last_write_ns = 0;
  const std::string p = PipelineCachePath();
  if (p.empty() || g_dev.pipeline_cache == VK_NULL_HANDLE)
    return;
  // Each save re-serializes and rewrites the WHOLE blob on the submit thread,
  // and a long-lived cache is hundreds of MB: saving once a second through a
  // level's compile burst added seconds to the very stall that trips UE4's
  // render-thread watchdog. Wait for the burst to end (or a minute of it).
  const u64 now = NowNs();
  if (!force && g_last_pipeline_build_ns <= last_write_ns)
    return;
  if (!force && now - g_last_pipeline_build_ns < 3000000000ull &&
      now - last_write_ns < 60000000000ull)
    return;
  size_t size = 0;
  if (vkGetPipelineCacheData(g_dev.device, g_dev.pipeline_cache, &size,
                             nullptr) != VK_SUCCESS ||
      !size)
    return;
  last_write_ns = now;
  // Only write when the driver has actually added something.
  if (!force && size == last_size)
    return;
  std::vector<u8> blob(size);
  if (vkGetPipelineCacheData(g_dev.device, g_dev.pipeline_cache, &size,
                             blob.data()) != VK_SUCCESS)
    return;
  last_size = size;
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
// Prints once per device; every later DEVICE_LOST is collateral of the first.
void ReportDeviceFault(DeviceState& device) {
  if (device.device_fault_reported)
    return;
  device.device_fault_reported = true;
  if (g_checkpoints_available && kCheckpoints) {
    auto get = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(
        vkGetDeviceProcAddr(device.device, "vkGetQueueCheckpointDataNV"));
    if (get) {
      u32 count = 0;
      get(device.queue, &count, nullptr);
      std::vector<VkCheckpointDataNV> checkpoints(
          count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
      get(device.queue, &count, checkpoints.data());
      for (const auto& checkpoint : checkpoints) {
        const u64 marker = reinterpret_cast<uintptr_t>(checkpoint.pCheckpointMarker);
        if (marker >> 63)
          BASE_LOGI("gpuvk", "  checkpoint stage={:#x} cs={:#x} {}",
                    u32(checkpoint.stage), (marker << 1) >> 2,
                    marker & 1 ? "after" : "before");
        else
          BASE_LOGI("gpuvk", "  checkpoint stage={:#x} frame={} draw={} {}",
                    u32(checkpoint.stage), (marker >> 32) - 1, u32(marker) >> 1,
                    marker & 1 ? "after" : "before");
      }
    }
  }
  if (!device.device_fault_available)
    return;
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

void DrawCheckpoint(rhi::CommandList* list, u32 frame, u32 draw, bool after) {
  if (g_checkpoints_available && kCheckpoints)
    list->Checkpoint(((u64(frame) + 1) << 32) | (u64(draw) << 1) | u32(after));
}

// Bit 63 tells a dispatch marker from a draw marker.
void DispatchCheckpoint(VkCommandBuffer cmd, u64 cs_addr, bool after) {
  if (!g_checkpoints_available || !kCheckpoints)
    return;
  if (!g_set_checkpoint)
    g_set_checkpoint = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(
        vkGetDeviceProcAddr(g_dev.device, "vkCmdSetCheckpointNV"));
  const uintptr_t marker = (1ull << 63) | ((cs_addr << 1) & ~(1ull << 63)) | u32(after);
  if (g_set_checkpoint)
    g_set_checkpoint(cmd, reinterpret_cast<const void*>(marker));
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
// flip), so it must be HOST_CACHED: reading from the default HOST_COHERENT
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
  b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0,
                        VK_REMAINING_ARRAY_LAYERS};
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
  b.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0,
                        VK_REMAINING_ARRAY_LAYERS};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  if (trace::Recording())
    trace::RecordBarrier("stencil", img, from, to, src_a, dst_a);
  vkCmdPipelineBarrier(c, StageForAccess(src_a, true),
                       StageForAccess(dst_a, false), 0, 0, nullptr, 0, nullptr,
                       1, &b);
}

bool CreateDevice() {
  render::BackendOptions options;
  options.gpu_filter = kVkGpu;
  options.debug_labels = WantDebugUtils() || trace::WantValidation();
  options.validation_layer =
      trace::WantValidation() ? trace::ValidationLayerName() : nullptr;
  options.sync_validation = trace::WantSyncValidation();
  options.checkpoints = kCheckpoints;
  const std::string cache_dir = PipelineCacheDir();
  options.cache_dir = cache_dir.empty() ? nullptr : cache_dir.c_str();
  // Lives for the process: tearing a device down during static
  // destruction races the driver's own exit handlers.
  rhi::Device* device =
      render::CreateBackendDevice(rhi::Backend::kVulkan, options).release();
  if (!device)
    return false;
  g_backend.device = device;
  const NativeDevice& native = Native(device);
  const rhi::Caps& caps = device->caps();
  g_dev.instance = native.instance;
  g_dev.phys = native.phys;
  g_dev.device = native.device;
  g_dev.queue = native.queue;
  g_dev.qfam = native.queue_family;
  g_dev.pipeline_cache = native.pipeline_cache;
  g_dev.timestamp_valid_bits = native.timestamp_valid_bits;
  g_dev.timestamp_period = static_cast<float>(caps.timestamp_period_ns);
  g_dev.push_descriptor = native.push_descriptor;
  g_dev.device_fault_available = native.device_fault;
  g_checkpoints_available = native.checkpoints;
  g_dev.mesh_shader = caps.mesh_shader;
  g_dev.mesh_limits = native.mesh_limits;
  g_dev.sampler_anisotropy = caps.sampler_anisotropy;
  g_dev.independent_blend = caps.independent_blend;
  g_dev.sampler_mirror_clamp = caps.sampler_mirror_clamp;
  g_dev.geometry_shader = caps.geometry_shader;
  g_dev.storage_image_write_without_format =
      caps.storage_image_write_without_format;
  g_dev.host_import_available = caps.host_import;
  g_dev.host_import_align = caps.host_import_alignment;
  g_dev.buffer_device_address = caps.buffer_address;
  g_dev.barycentric_available = caps.fragment_barycentric;
  g_dev.storage_buffer_offset_align = caps.storage_offset_alignment;
  g_dev.max_storage_buffer_range = caps.max_storage_buffer_range;
  g_dev.max_cs_resources =
      std::min(gcn::kMaxCsResources, caps.max_compute_resources);
  InitDebugUtils(g_dev.instance, options.debug_labels);
  trace::InstallValidationMessenger(g_dev.instance);
  if (g_dev.mesh_shader)
    g_dev.draw_mesh_tasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(g_dev.device, "vkCmdDrawMeshTasksEXT"));
  if (g_dev.push_descriptor)
    g_dev.push_descriptor_set = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(g_dev.device, "vkCmdPushDescriptorSetKHR"));
  g_cmd_begin_rendering = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
      vkGetDeviceProcAddr(g_dev.device, "vkCmdBeginRendering"));
  g_cmd_end_rendering = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
      vkGetDeviceProcAddr(g_dev.device, "vkCmdEndRendering"));

  VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pc.queueFamilyIndex = g_dev.qfam;
  VKOK(vkCreateCommandPool(g_dev.device, &pc, nullptr, &g_dev.pool));
  VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VKOK(vkCreateFence(g_dev.device, &fc, nullptr,
                     &g_dev.fence));  // aux submits only
  if (!CreateFrameSlots())
    return false;

  if (!CreateUploadRings())
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

namespace gpu::render {
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
  // llvmpipe, a ~30ms/frame software rasteriser on a box with a real GPU.
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

}  // namespace gpu::render
