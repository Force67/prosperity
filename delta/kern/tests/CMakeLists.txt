add_delta_test(object_table_test
  SOURCES object_table_test.cpp
  DEPS delta_kern)

add_delta_test(audio_queue_test
  SOURCES audio_queue_test.cpp
  DEPS delta_deps)

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

# A device now takes the object table it registers with rather than the whole
# process, so these two build the unit under test into the test binary and link
# no emulator at all -- which is the acceptance test for the boundary.
add_delta_test(console_dev_test
  SOURCES console_dev_test.cpp
    ../ps4/dev/console_dev.cpp ../ps4/dev/file_dev.cpp
    ../object.cpp ../object_table.cpp
  DEPS delta_deps)

add_delta_test(hid_dev_test
  SOURCES hid_dev_test.cpp
    ../ps4/dev/hid_dev.cpp ../ps4/dev/file_dev.cpp
    ../object.cpp ../object_table.cpp
  DEPS delta_deps)
