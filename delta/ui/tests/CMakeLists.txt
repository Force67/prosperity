add_delta_test(shader_activity_test
  SOURCES shader_activity_test.cc
  DEPS delta_ui)

add_delta_test(home_background_test
  SOURCES home_background_test.cc
  DEPS delta_ui)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
  target_include_directories(home_background_test PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/imgui)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
  add_delta_test(settings_menu_test
    SOURCES settings_menu_test.cc
    DEPS delta_ui)
  target_include_directories(settings_menu_test PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/equilibrium/external/tracy/imgui)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
  add_delta_test(mouse_look_test
    SOURCES mouse_look_test.cc
    DEPS delta_ui)
endif()
