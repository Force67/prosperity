/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Debug-utils object names and command labels for capture tools. Everything
// here is a no-op unless VK_EXT_debug_utils is present, which in practice
// means RenderDoc or the validation layer is attached: names turn anonymous
// handles into "rt 0x8142f00000 1920x1080 tex" in the resource inspector, and
// labels group the event browser into frame / region / draw / dispatch scopes
// instead of a flat run of vkCmdDraw calls.
//
// Everything is keyed by GUEST addresses (the rt_base, shader address, texture
// base), because that is the vocabulary every other diagnostic in this module
// speaks -- a capture can be lined up against DELTA_GPU_* logs by eye.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace gpu::vk {

// Whether names/labels should be emitted at all. The Vulkan loader implements
// VK_EXT_debug_utils itself, so the extension is always advertised -- but with
// no consumer every label is a formatted string handed to nobody (~0.3 ms per
// Isaac frame). True only when RenderDoc is injected into the process or
// DELTA_GPU_MARKERS=1 forces it (e.g. for a validation-layer run).
bool WantDebugUtils();

// Resolve the entry points off the instance; called once by CreateDevice
// after instance creation. Safe to call when the extension is missing.
void InitDebugUtils(VkInstance instance, bool extension_enabled);

// True when names/labels actually reach a tool.
bool DebugUtilsActive();

// Attach a printf-formatted name to any Vulkan object.
void NameObject(VkObjectType type, uint64_t handle, const char* fmt, ...)
    __attribute__((format(printf, 3, 4)));

// Open/close a nested label region in a command buffer.
void CmdBeginLabel(VkCommandBuffer cmd, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
void CmdEndLabel(VkCommandBuffer cmd);
// One-shot marker between commands.
void CmdInsertLabel(VkCommandBuffer cmd, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

// RAII label region for a scope that records into one command buffer.
class ScopedCmdLabel {
 public:
  ScopedCmdLabel(VkCommandBuffer cmd, const char* fmt, ...)
      __attribute__((format(printf, 3, 4)));
  ~ScopedCmdLabel();
  ScopedCmdLabel(const ScopedCmdLabel&) = delete;
  ScopedCmdLabel& operator=(const ScopedCmdLabel&) = delete;

 private:
  VkCommandBuffer cmd_;
};

}  // namespace gpu::vk
