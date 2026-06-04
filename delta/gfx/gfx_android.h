#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Android app (DELTA_ANDROID_APP) glue between the NativeActivity event loop and
 * the on-screen Vulkan backend in gfx_android.cpp. android_main owns the window
 * and input; the GPU renderer drives gfx::present() as usual.
 */
#if defined(__ANDROID__) && defined(DELTA_ANDROID_APP)

#include "gfx.h"

struct ANativeWindow;

namespace gfx {

// Hand the app's native window (or nullptr on teardown) to the gfx backend.
// Set before the guest renderer first calls gfx::ensure().
void setAndroidWindow(ANativeWindow *window);

// Publish the current touch-derived pad state; pollKeyboardPad() returns it.
void setAndroidPad(const PadKeys &keys);

}  // namespace gfx

#endif
