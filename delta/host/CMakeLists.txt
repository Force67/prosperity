# delta/host: the host machine's window, input, audio and overlay.
#
# One window_* variant is built per platform: SDL3 + Vulkan on the desktop, the
# NativeActivity surface in the Android app, and a headless stub for the adb
# runner (which has no Activity). Audio and the overlay need SDL/ImGui, so the
# Android builds leave them out.
set(_desktop_only audio_output_sdl.cc overlay.cc overlay_log.cc overlay_theme.cc overlay_vk.cc
                  window_sdl.cc)
if(ANDROID)
  if(DELTA_ANDROID_APP)
    add_delta_module(host DEPS vulkan android log)
    set_source_files_properties(window_headless.cc PROPERTIES HEADER_FILE_ONLY ON)
  else()
    add_delta_module(host)
    set_source_files_properties(window_android.cc PROPERTIES HEADER_FILE_ONLY ON)
  endif()
  set_source_files_properties(${_desktop_only} PROPERTIES HEADER_FILE_ONLY ON)
else()
  find_package(Vulkan REQUIRED)
  find_package(SDL3 REQUIRED CONFIG)
  add_delta_module(host DEPS Vulkan::Vulkan SDL3::SDL3)
  set_source_files_properties(window_android.cc window_headless.cc
                              audio_output_stub.cc PROPERTIES HEADER_FILE_ONLY ON)

  # Dear ImGui core (reuse the copy vendored with tracy) for the controls-legend
  # overlay. We only need it to build a raw draw list; overlay.cc rasterises the
  # ImDrawData into the present framebuffer itself, so no ImGui backend is used.
  set(_imgui ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/imgui)
  target_sources(delta_host PRIVATE
    ${_imgui}/imgui.cpp
    ${_imgui}/imgui_draw.cpp
    ${_imgui}/imgui_tables.cpp
    ${_imgui}/imgui_widgets.cpp)
  target_include_directories(delta_host PRIVATE ${_imgui}
    ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/profiler/src)

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
    target_include_directories(delta_host PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()
