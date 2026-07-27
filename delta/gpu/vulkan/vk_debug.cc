/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_debug.h"

#include "gpu/vulkan/vk_device.h"

#include <dlfcn.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace gpu::vk {

namespace {

PFN_vkSetDebugUtilsObjectNameEXT g_set_name = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT g_begin_label = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT g_end_label = nullptr;
PFN_vkCmdInsertDebugUtilsLabelEXT g_insert_label = nullptr;

// One shared formatting buffer per call keeps this allocation-free; the
// renderer records from a single thread (the guest GPU thread), which is the
// only caller of everything here.
const char* Format(char (&buf)[192], const char* fmt, va_list args) {
  std::vsnprintf(buf, sizeof buf, fmt, args);
  return buf;
}

}  // namespace

bool WantDebugUtils() {
  static const bool want = [] {
    const char* e = std::getenv("DELTA_GPU_MARKERS");
    if (e)
      return e[0] && e[0] != '0';
    return dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD) != nullptr;
  }();
  return want;
}

void InitDebugUtils(VkInstance instance, bool extension_enabled) {
  if (!instance || !extension_enabled)
    return;
  g_set_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
      vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
  g_begin_label = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
      vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
  g_end_label = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
      vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
  g_insert_label = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
      vkGetInstanceProcAddr(instance, "vkCmdInsertDebugUtilsLabelEXT"));
  if (g_set_name)
    std::fprintf(stderr, "[gpuvk] debug utils active (names + labels)\n");
}

bool DebugUtilsActive() {
  return g_set_name != nullptr;
}

void NameObject(VkObjectType type, uint64_t handle, const char* fmt, ...) {
  if (!g_set_name || !handle)
    return;
  char buf[192];
  va_list args;
  va_start(args, fmt);
  VkDebugUtilsObjectNameInfoEXT info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  info.objectType = type;
  info.objectHandle = handle;
  info.pObjectName = Format(buf, fmt, args);
  va_end(args);
  g_set_name(g_dev.device, &info);
}

void CmdBeginLabel(VkCommandBuffer cmd, const char* fmt, ...) {
  if (!g_begin_label || !cmd)
    return;
  char buf[192];
  va_list args;
  va_start(args, fmt);
  VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  label.pLabelName = Format(buf, fmt, args);
  va_end(args);
  g_begin_label(cmd, &label);
}

void CmdEndLabel(VkCommandBuffer cmd) {
  if (g_end_label && cmd)
    g_end_label(cmd);
}

void CmdInsertLabel(VkCommandBuffer cmd, const char* fmt, ...) {
  if (!g_insert_label || !cmd)
    return;
  char buf[192];
  va_list args;
  va_start(args, fmt);
  VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  label.pLabelName = Format(buf, fmt, args);
  va_end(args);
  g_insert_label(cmd, &label);
}

ScopedCmdLabel::ScopedCmdLabel(VkCommandBuffer cmd, const char* fmt, ...)
    : cmd_(cmd) {
  if (!g_begin_label || !cmd)
    return;
  char buf[192];
  va_list args;
  va_start(args, fmt);
  VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  label.pLabelName = Format(buf, fmt, args);
  va_end(args);
  g_begin_label(cmd, &label);
}

ScopedCmdLabel::~ScopedCmdLabel() {
  if (g_begin_label)  // paired: if the begin ran, the end must too
    CmdEndLabel(cmd_);
}

}  // namespace gpu::vk
