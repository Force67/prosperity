/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The frame ring. Two slots let frame N record (and the guest emulate) while
// frame N-1 still rasterizes; each slot owns a command buffer, a fence, a
// readback buffer and half of each upload ring. Slot N-1's fence is waited --
// and its pixels presented, one frame late -- at frame N's endFrame.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace gpu::vk {

struct FrameSlot {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readbackMem = VK_NULL_HANDLE;
  void *readbackMap = nullptr;
  VkDeviceSize readbackSize = 0;
  bool submitted = false;    // fence submitted and not yet waited
  bool presentable = false;  // the frame copied pixels into `readback`
  // Metadata of the recorded frame, consumed when it is presented.
  uint32_t w = 0, h = 0;
  VkFormat fmt = VK_FORMAT_UNDEFINED;
  int frameNum = 0;
  uint32_t frameDraws = 0, frameMaxIdx = 0;
  bool frameHadRoom = false;
  uint64_t presentBase = 0, scanoutBase = 0;
};

struct FrameState {
  // The active slot's command buffer and readback buffer, aliased here so the
  // recording path does not thread the slot through every call.
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readbackMem = VK_NULL_HANDLE;
  void *readbackMap = nullptr;
  VkDeviceSize readbackSize = 0;

  int num = 0;  // monotonic frame counter; the caches age against it
  uint32_t draws = 0;
  uint32_t heuristic = 0;  // draws that fell back to the heuristic quad path
  uint32_t maxIdx = 0;     // largest indexCount this frame (3D detector)
  bool recording = false;
  bool hadRoom = false;   // this frame sampled a room-sized (~832w) RT
  bool roomBake = false;  // this frame RENDERED into a room-sized RT

  FrameSlot slots[2];
  uint32_t slotIdx = 0;
};

extern FrameState g_frame;

bool createFrameSlots();
// Pipelined by default; DELTA_GPU_SYNC=1 restores the submit-and-wait frame.
bool framePipelined();
// Grow the active slot's readback buffer to hold one w*h image of `fmt`.
void ensureReadback(uint32_t w, uint32_t h, VkFormat fmt);

}  // namespace gpu::vk
