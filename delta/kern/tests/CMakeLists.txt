add_delta_test(object_table_test
  SOURCES object_table_test.cpp
  DEPS delta_kern)

add_delta_test(sys_info_test
  SOURCES sys_info_test.cpp
  DEPS delta_kern)
set_tests_properties(sys_info_test PROPERTIES ENVIRONMENT "DELTA_PS4_NEO=0")
add_test(NAME sys_info_default_test
  COMMAND ${CMAKE_COMMAND} -E env --unset=DELTA_PS4_NEO $<TARGET_FILE:sys_info_test>)
add_test(NAME sys_info_neo_test COMMAND sys_info_test)
set_tests_properties(sys_info_neo_test PROPERTIES ENVIRONMENT "DELTA_PS4_NEO=2")

add_delta_test(vfs_test
  SOURCES vfs_test.cpp
  DEPS delta_kern)

add_delta_test(umtx_op_test
  SOURCES umtx_op_test.cpp
  DEPS delta_kern)
