/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The frame debugger: one armed guest frame recorded, in order, into one
// machine-readable file.
//
// The ~150 DELTA_GPU_* printf switches answer one question each, cost a
// recompile when the question changes, and cap themselves at an arbitrary line
// count. This records the whole frame instead -- every region, draw, dispatch,
// barrier and validation message with its complete state -- so a new question
// is a query over an existing capture (tools/gpu_capture.py) rather than a new
// build.
//
// Everything is keyed by GUEST addresses, the vocabulary the rest of the
// module speaks. Every entry point below is a no-op unless a capture is
// recording, so the disarmed cost is one predictable branch on a global bool.

#include <vulkan/vulkan.h>

#include <cstdint>

#include "gpu/rhi/command.h"

namespace gpu::vk::trace {

// True only while an armed frame is recording. Call sites test this before
// building any argument, so a disarmed run pays nothing.
extern bool g_recording;
inline bool Recording() {
  return g_recording;
}

// Frame lifecycle. FrameBegin decides whether this frame is the armed one and
// opens its file; FrameEnd closes it, after reading back and writing every
// requested resource dump.
void FrameBegin(int frame_num);
void FrameEnd(uint64_t scanout_base);

// The colour/depth attachments of one dynamic-rendering region, as
// vk_render_target binds them.
struct RegionInfo {
  const uint64_t* mrt_base = nullptr;
  const uint32_t* mrt_info = nullptr;
  uint32_t mrt_count = 0;
  uint32_t width = 0, height = 0;
  uint64_t depth_base = 0;
  uint64_t stencil_base = 0;
  uint32_t color_clear_mask = 0;  // bit i: attachment i opens with loadOp CLEAR
  bool depth_clear = false;
  float depth_clear_value = 1.0f;
};
void RegionBegin(const RegionInfo& region);
void RegionEnd();

// How each of a draw's sampler bindings actually resolved. This is the
// question a capture exists to answer -- a binding that silently falls through
// to guest memory reads what no draw ever wrote -- and only the draw path
// knows it, so it is passed in rather than recomputed.
struct DrawBindings {
  const uint64_t* tex_color = nullptr;     // resolved to a live colour RT
  const uint64_t* tex_feedback = nullptr;  // resolved to a feedback copy
  const uint64_t* tex_depth = nullptr;     // resolved to a depth target
  const uint64_t* tex_storage = nullptr;   // bound as a storage image
  const void* const* tex_guest = nullptr;  // VkImageView of a guest upload
  uint32_t tex_count = 0;
  uint32_t cbuf_mask = 0;    // bit i: cbuffer binding i staged real memory
  uint32_t rawbuf_mask = 0;  // bit i: set-2 binding i staged real memory
};

// A draw the recompiled path issued (`path` = "recomp"), or the heuristic quad
// fallback ("quad"). `bindings` may be null for paths that resolve no samplers.
void RecordDraw(const rhi::DrawInfo& draw,
                const char* path,
                const DrawBindings* bindings);
// The recompiled path declining a draw, tallied by reason at its one choke
// point. A frame whose draws silently fall back is otherwise only visible as a
// number in a periodic report.
void RecordDecline(const char* reason);

void RecordDispatch(const rhi::ComputeInfo& dispatch);

// Every layout transition the backend records, named after the guest resource
// the image holds.
void RecordBarrier(const char* aspect,
                   VkImage image,
                   VkImageLayout from,
                   VkImageLayout to,
                   VkAccessFlags src_access,
                   VkAccessFlags dst_access);

// A CP DMA fill over guest memory (this hardware's clear).
void RecordMemoryFill(uint64_t base, uint64_t bytes, uint32_t value);

// --- Vulkan-level visibility -----------------------------------------------

// DELTA_GPU_VALIDATE=1: enable the Khronos validation layers and route their
// messages into the capture (and stderr) alongside the draw that provoked
// them. Read once, before instance creation.
bool WantValidation();
const char* ValidationLayerName();
void InstallValidationMessenger(VkInstance instance);
void DestroyValidationMessenger(VkInstance instance);

// Object-name registry. vk_debug names every object it creates after the guest
// resource it represents; the registry keeps those names on the host side too,
// so a barrier or a validation message can say "rt 0x8142f00000 1920x1080"
// with no capture tool attached. Populated only when a capture is armed or
// validation is on.
bool NamesWanted();
void RegisterObjectName(VkObjectType type, uint64_t handle, const char* name);
// "" when the handle was never named.
const char* ObjectName(uint64_t handle);

}  // namespace gpu::vk::trace
