add_delta_test(code_lift_test
  SOURCES code_lift_test.cc
  DEPS delta_runtime)

add_delta_test(fiber_shim_test
  SOURCES fiber_shim_test.cc
  DEPS delta_runtime)

if(AVCODEC_FOUND)
  add_delta_test(videodec_test
    SOURCES videodec_test.cc
    DEPS delta_runtime)
endif()
