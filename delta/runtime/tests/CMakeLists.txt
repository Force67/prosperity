add_delta_test(code_lift_test
  SOURCES code_lift_test.cc
  DEPS delta_runtime)

add_delta_test(fiber_shim_test
  SOURCES fiber_shim_test.cc
  DEPS delta_runtime)

add_delta_test(pad_test
  SOURCES pad_test.cc
  DEPS delta_runtime)

add_delta_test(display_timing_test
  SOURCES display_timing_test.cc
  DEPS delta_runtime)
set_tests_properties(display_timing_test PROPERTIES TIMEOUT 15 ENVIRONMENT
  "DELTA_GPU_NOPRESENT=1;SDL_VIDEODRIVER=offscreen")

add_delta_test(save_data_test
  SOURCES save_data_test.cc
  DEPS delta_runtime)

if(AVCODEC_FOUND)
  add_delta_test(videodec_test
    SOURCES videodec_test.cc
    DEPS delta_runtime)
endif()
