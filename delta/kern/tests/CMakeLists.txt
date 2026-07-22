add_delta_test(object_table_test
  SOURCES object_table_test.cpp
  DEPS delta_kern)

add_delta_test(sys_info_test
  SOURCES sys_info_test.cpp
  DEPS delta_kern)

add_delta_test(vfs_test
  SOURCES vfs_test.cpp
  DEPS delta_kern)

add_delta_test(umtx_op_test
  SOURCES umtx_op_test.cpp
  DEPS delta_kern)
