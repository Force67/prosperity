add_delta_test(archive_rar_test
  SOURCES archive_rar_test.cc
  DEPS delta_formats)

add_delta_test(archive_zip_test
  SOURCES archive_zip_test.cc
  DEPS delta_formats)

add_delta_test(fself_test
  SOURCES fself_test.cc
  DEPS delta_formats)
