/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The frame ring. Two slots let frame N record (and the guest emulate) while
// frame N-1 still rasterizes; each slot owns a command buffer, a fence, a
// readback buffer and half of each upload ring. Slot N-1's fence is waited --
// and its pixels presented, one frame late -- at frame N's EndFrame.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace gpu::vk {

struct FrameSlot {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool timestamps = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readback_mem = VK_NULL_HANDLE;
  void* readback_map = nullptr;
  VkDeviceSize readback_size = 0;
  bool submitted = false;    // fence submitted and not yet waited
  bool presentable = false;  // the frame copied pixels into `readback`
  bool present_to_window = false;
  // Metadata of the recorded frame, consumed when it is presented.
  uint32_t w = 0, h = 0;
  VkFormat fmt = VK_FORMAT_UNDEFINED;
  int frame_num = 0;
  uint32_t frame_draws = 0, frame_max_idx = 0;
  bool frame_had_room = false;
  uint64_t present_base = 0, scanout_base = 0;
};

struct FrameState {
  // The active slot's command buffer and readback buffer, aliased here so the
  // recording path does not thread the slot through every call.
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readback_mem = VK_NULL_HANDLE;
  void* readback_map = nullptr;
  VkDeviceSize readback_size = 0;

  int num = 0;  // monotonic frame counter; the caches age against it
  uint32_t draws = 0;
  uint32_t heuristic = 0;  // draws that fell back to the heuristic quad path
  uint32_t max_idx = 0;    // largest index_count this frame (3D detector)
  bool recording = false;
  bool had_room = false;   // this frame sampled a room-sized (~832w) RT
  bool room_bake = false;  // this frame RENDERED into a room-sized RT

  FrameSlot slots[2];
  uint32_t slot_idx = 0;
};

extern FrameState& g_frame;

bool CreateFrameSlots();
// Pipelined by default; DELTA_GPU_SYNC=1 restores the submit-and-wait frame.
bool FramePipelined();
// Grow the active slot's readback buffer to hold one w*h image of `fmt`.
void EnsureReadback(uint32_t w, uint32_t h, VkFormat fmt);

}  // namespace gpu::vk
