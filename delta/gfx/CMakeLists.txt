# Graphics backend: SDL3 window + Vulkan present.
#
# Android has no SDL3 window (no Activity in the adb-shell runner), so it builds
# the headless stub (gfx_headless.cpp) with no SDL/Vulkan dependency; gfx_vk.cpp
# is #ifndef __ANDROID__ and compiles to nothing there.
if(ANDROID)
  if(DELTA_ANDROID_APP)
    # On-screen build: gfx_android.cpp needs Vulkan (android surface) + the NDK
    # android/log libs; gfx_headless.cpp compiles to nothing here.
    add_delta_module(gfx DEPS vulkan android log)
  else()
    add_delta_module(gfx)
  endif()
else()
  find_package(Vulkan REQUIRED)
  find_package(SDL3 REQUIRED CONFIG)
  add_delta_module(gfx DEPS Vulkan::Vulkan SDL3::SDL3)
endif()
