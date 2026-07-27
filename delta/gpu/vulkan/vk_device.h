/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The Vulkan device the renderer owns: instance, adapter, queue, command pool,
// and the driver limits and features every other unit reads. There is no
// surface -- guest frames render offscreen and are read back (see vk_frame).

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace gpu::vk {

// Bail out of the enclosing bool-returning creation function on a Vulkan error.
#define VKOK(x)                                                     \
  do {                                                              \
    VkResult _r = (x);                                              \
    if (_r != VK_SUCCESS) {                                         \
      std::fprintf(stderr, "[gpuvk] %s failed: %d\n", #x, (int)_r); \
      return false;                                                 \
    }                                                               \
  } while (0)

struct DeviceState {
  bool ready = false;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t qfam = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  // Dedicated to the synchronous aux submits: texture uploads, rtstat.
  VkFence fence = VK_NULL_HANDLE;
  uint32_t max_cs_resources = 0;
  VkDeviceSize max_storage_buffer_range = 0;
  bool sampler_anisotropy = false;
  bool sampler_mirror_clamp = false;
  bool geometry_shader = false;
  bool storage_image_write_without_format = false;
};

extern DeviceState g_dev;

// Dynamic rendering (core in 1.3, KHR on older drivers), resolved at device
// creation: the renderer records no render passes.
extern PFN_vkCmdBeginRenderingKHR g_cmd_begin_rendering;
extern PFN_vkCmdEndRenderingKHR g_cmd_end_rendering;

bool CreateDevice();

uint32_t FindMemoryType(uint32_t type_bits, VkMemoryPropertyFlags props);
uint32_t FindMemoryTypePref(uint32_t type_bits,
                            VkMemoryPropertyFlags pref,
                            VkMemoryPropertyFlags req);

void ImageBarrier(VkCommandBuffer c,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_a,
                  VkAccessFlags dst_a,
                  uint32_t layers = 1,
                  uint32_t mip_levels = 1);
void DepthBarrier(VkCommandBuffer c,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_a,
                  VkAccessFlags dst_a);

VkShaderModule MakeModule(const uint32_t* spv, size_t bytes);
VkShaderModule MakeModuleVec(const std::vector<uint32_t>& spv);

// Ask the driver what the GPU actually faulted on (VK_EXT_device_fault).
void ReportDeviceFault(VkDevice device);

}  // namespace gpu::vk
