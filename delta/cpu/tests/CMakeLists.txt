if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND DELTA_BACKEND STREQUAL "NATIVE")
  add_delta_test(guest_pause_test SOURCES guest_pause_test.cc DEPS delta_cpu)
  set_tests_properties(guest_pause_test PROPERTIES TIMEOUT 15)
endif()
