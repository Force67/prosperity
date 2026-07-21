add_delta_test(gcn_detile_test
  SOURCES
    gcn_detile_test.cpp
    ../ps4/gcn/gcn_detile.cpp
  DEPS delta_deps)

add_delta_test(gcn_decode_test
  SOURCES
    gcn_decode_test.cpp
    ../ps4/gcn/gcn_decode.cpp
  DEPS delta_deps)
