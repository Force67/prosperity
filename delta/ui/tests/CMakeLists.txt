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
