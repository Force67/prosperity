# Desktop menus and overlays. Compiler activity remains available on Android.
if(ANDROID)
  add_delta_module(ui)
  set_source_files_properties(
    pause_menu.cc home_screen.cc home_background_vk.cc image.cc input_sdl.cc overlay.cc overlay_busy.cc
    overlay_log.cc overlay_theme.cc overlay_vk.cc
    PROPERTIES HEADER_FILE_ONLY ON)
else()
  find_package(Vulkan REQUIRED)
  find_package(SDL3 REQUIRED CONFIG)
  add_delta_module(ui DEPS Vulkan::Vulkan SDL3::SDL3)

  set(_imgui ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/imgui)
  target_sources(delta_ui PRIVATE
    ${_imgui}/imgui.cpp
    ${_imgui}/imgui_draw.cpp
    ${_imgui}/imgui_tables.cpp
    ${_imgui}/imgui_widgets.cpp)
  target_include_directories(delta_ui PRIVATE ${_imgui})
  target_include_directories(delta_ui PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/profiler/src)
endif()
