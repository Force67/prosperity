/*
 * PS4Delta : PS4 emulation and research project
 *
 * Headless gfx backend for Android.
 *
 * SDL3 on Android needs a Java Activity / APK, which the adb-shell native
 * runner doesn't have. The GPU renderer (delta/gpu) creates its own surfaceless
 * Vulkan device and dumps frames (DELTA_GPU_DUMP) regardless of a window, so on
 * Android we drop SDL entirely and stub the window/present/input out: init
 * fails (the VideoOut path already handles "no window this run"), present is a
 * no-op, and the keyboard pad reports no input.
 *
 * The on-screen app build (DELTA_ANDROID_APP) uses gfx_android.cpp instead.
 */
#if defined(__ANDROID__) && !defined(DELTA_ANDROID_APP)

#include "gfx.h"
#include "overlay.h"

namespace gfx {

bool init(const char *, uint32_t, uint32_t) { return false; }
bool ensure(const char *, uint32_t, uint32_t) { return false; }
bool available() { return false; }
bool canPresent() { return false; }
void requestPresentStop() {}
void present(const void *, uint32_t, uint32_t, uint32_t, PixelFormat) {}
bool pumpEvents() { return true; }
bool pollKeyboardPad(PadKeys &) { return false; }
void setRumble(uint8_t, uint8_t) {}
void shutdown() {}
void setTitle(const char *) {}
void overlaySetPerf(float, float, float) {}

} // namespace gfx

#endif // __ANDROID__
