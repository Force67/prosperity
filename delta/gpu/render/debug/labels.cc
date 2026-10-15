/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/debug/labels.h"

#include <dlfcn.h>
#include <cstdarg>
#include <cstdio>

#include "options/options.h"

#include "gpu/render/frame.h"
#include "gpu/render/debug/trace.h"

namespace {
DELTA_OPTION(bool, kMarkers, "DELTA_GPU_MARKERS", false);
}  // namespace

namespace gpu::render {

namespace {

const char* Format(char (&buf)[192], const char* fmt, va_list args) {
  std::vsnprintf(buf, sizeof buf, fmt, args);
  return buf;
}

}  // namespace

bool WantDebugUtils() {
  static const bool kWant = [] {
    if (kMarkers.overridden())
      return kMarkers.get();
    if (trace::WantValidation())
      return true;
    return dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD) != nullptr;
  }();
  return kWant;
}

void CmdBeginLabel(rhi::CommandList* list, const char* fmt, ...) {
  if (!Device().caps().debug_labels)
    return;
  char buf[192];
  va_list args;
  va_start(args, fmt);
  list->PushLabel(Format(buf, fmt, args));
  va_end(args);
}

void CmdEndLabel(rhi::CommandList* list) {
  if (Device().caps().debug_labels)
    list->PopLabel();
}

void CmdInsertLabel(rhi::CommandList* list, const char* fmt, ...) {
  if (!Device().caps().debug_labels)
    return;
  char buf[192];
  va_list args;
  va_start(args, fmt);
  list->InsertLabel(Format(buf, fmt, args));
  va_end(args);
}

}  // namespace gpu::render
