/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Debug-utils object names + command labels for capture tools; no-ops unless
// VK_EXT_debug_utils has a consumer (RenderDoc or the validation layer). Names turn
// anonymous handles into "rt 0x8142f00000 1920x1080 tex"; labels group the event
// browser into frame/region/draw/dispatch. Keyed by GUEST addresses so a capture
// lines up against DELTA_GPU_* logs by eye.

#include <vulkan/vulkan.h>
#include "base/arch.h"


namespace gpu::vk {

// Whether to emit names/labels at all: the loader always advertises the extension,
// but with no consumer every label is a formatted string for nobody (~0.3 ms per
// Isaac frame). True only with RenderDoc injected or DELTA_GPU_MARKERS=1.
bool WantDebugUtils();

// Resolve the entry points off the instance; called once by CreateDevice
// after instance creation. Safe to call when the extension is missing.
void InitDebugUtils(VkInstance instance, bool extension_enabled);

// True when names/labels actually reach a tool.
bool DebugUtilsActive();

// Attach a printf-formatted name to any Vulkan object.
void NameObject(VkObjectType type, u64 handle, const char* fmt, ...)
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
