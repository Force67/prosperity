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

  # Dear ImGui core (reuse the copy vendored with tracy) for the controls-legend
  # overlay. We only need it to build a raw draw list; overlay.cpp rasterises the
  # ImDrawData into the present framebuffer itself, so no ImGui backend is used.
  set(_imgui ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/imgui)
  target_sources(delta_gfx PRIVATE
    ${_imgui}/imgui.cpp
    ${_imgui}/imgui_draw.cpp
    ${_imgui}/imgui_tables.cpp
    ${_imgui}/imgui_widgets.cpp)
  target_include_directories(delta_gfx PRIVATE ${_imgui})

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_logo ${CMAKE_SOURCE_DIR}/delta/main/_res/logo.png)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_logo})
    file(READ ${_logo} _logo_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _logo_bytes
      "${_logo_hex}")
    file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/prosperity_logo.h
      CONTENT
        "static constexpr unsigned char kProsperityLogoPng[] = {${_logo_bytes}};\n"
      @ONLY)
    target_include_directories(delta_gfx PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/profiler/src)
  endif()
endif()
