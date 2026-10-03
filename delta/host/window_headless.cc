/*
 * PS4Delta : PS4 emulation and research project
 *
 * Headless window backend for the Android adb runner.
 *
 * SDL3 on Android needs a Java Activity / APK, which the adb-shell native
 * runner doesn't have. The GPU renderer (delta/gpu) creates its own surfaceless
 * Vulkan device and dumps frames (DELTA_GPU_DUMP) regardless of a window, so on
 * Android we drop SDL entirely and stub the window/present/input out: init
 * fails (the VideoOut path already handles "no window this run"), present is a
 * no-op, and the keyboard pad reports no input.
 *
 * The on-screen app build (DELTA_ANDROID_APP) uses window_android.cc instead.
 */

#include "base/arch.h"
#include "host/window.h"

namespace host {

bool Init(const char*, u32, u32) {
  return false;
}
bool Ensure(const char*, u32, u32) {
  return false;
}
bool Available() {
  return false;
}
bool CanPresent() {
  return false;
}
void RequestPresentStop() {}
void RefreshFrame(bool) {}

void Present(const void*, u32, u32, u32, PixelFormat) {}
bool PumpEvents() {
  return true;
}
bool PollKeyboardPad(PadKeys&) {
  return false;
}
void SetRumble(u8, u8) {}
void ResetGuest() {}

void Shutdown() {}
void QueryVram(u64& used, u64& total) {
  used = total = 0;
}

}  // namespace host
