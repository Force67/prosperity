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
#define VKOK(x)                                                                \
  do {                                                                         \
    VkResult _r = (x);                                                         \
    if (_r != VK_SUCCESS) {                                                    \
      std::fprintf(stderr, "[gpuvk] %s failed: %d\n", #x, (int)_r);            \
      return false;                                                            \
    }                                                                          \
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
  uint32_t maxCsResources = 0;
  VkDeviceSize maxStorageBufferRange = 0;
  bool samplerAnisotropy = false;
  bool samplerMirrorClamp = false;
  bool geometryShader = false;
  bool storageImageWriteWithoutFormat = false;
};

extern DeviceState g_dev;

// Dynamic rendering (core in 1.3, KHR on older drivers), resolved at device
// creation: the renderer records no render passes.
extern PFN_vkCmdBeginRenderingKHR p_vkCmdBeginRendering;
extern PFN_vkCmdEndRenderingKHR p_vkCmdEndRendering;

bool createDevice();

uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);
uint32_t findMemoryTypePref(uint32_t typeBits, VkMemoryPropertyFlags pref,
                            VkMemoryPropertyFlags req);

void imageBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                  VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA,
                  uint32_t layers = 1, uint32_t mip_levels = 1);
void depthBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                  VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA);

VkShaderModule makeModule(const uint32_t *spv, size_t bytes);
VkShaderModule makeModuleVec(const std::vector<uint32_t> &spv);

// Ask the driver what the GPU actually faulted on (VK_EXT_device_fault).
void reportDeviceFault(VkDevice device);

}  // namespace gpu::vk
