# Host window, input, and audio. Desktop UI is provided by delta_ui.
set(_desktop_only audio_output_sdl.cc window_sdl.cc)
if(ANDROID)
  if(DELTA_ANDROID_APP)
    add_delta_module(host DEPS delta_ui vulkan android log)
    set_source_files_properties(window_headless.cc PROPERTIES HEADER_FILE_ONLY ON)
  else()
    add_delta_module(host DEPS delta_ui)
    set_source_files_properties(window_android.cc PROPERTIES HEADER_FILE_ONLY ON)
  endif()
  set_source_files_properties(${_desktop_only} PROPERTIES HEADER_FILE_ONLY ON)
else()
  find_package(Vulkan REQUIRED)
  find_package(SDL3 REQUIRED CONFIG)
  add_delta_module(host DEPS delta_ui Vulkan::Vulkan SDL3::SDL3)
  set_source_files_properties(window_android.cc window_headless.cc
                              audio_output_stub.cc PROPERTIES HEADER_FILE_ONLY ON)

  target_include_directories(delta_host PRIVATE
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
