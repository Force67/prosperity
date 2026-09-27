#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Android app (DELTA_ANDROID_APP) glue between the NativeActivity event loop and
 * the on-screen Vulkan backend in window_android.cc. android_main owns the window
 * and input; the GPU renderer drives host::present() as usual.
 */

#include "window.h"

struct ANativeWindow;

namespace host {

// Hand the app's native window (or nullptr on teardown) to the window backend.
// Set before the guest renderer first calls host::ensure().
void setAndroidWindow(ANativeWindow *window);

// Publish the currently-down touch points (surface/window pixel coords).
// The window owns the on-screen control layout, so it maps these to the DS4 pad
// (pollKeyboardPad) and draws the matching helper overlay on present.
struct Touch {
  float x, y;
};
void setAndroidTouches(const Touch *pts, int count);

}  // namespace host
