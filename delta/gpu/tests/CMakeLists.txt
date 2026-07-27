# Machine-enforced layering (a scaled-down Chromium DEPS check): directory
# dependency rules + the module's public include surface.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_test(NAME gpu_layering
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_layering.py
            ${CMAKE_SOURCE_DIR})
endif()

add_delta_test(gcn_detile_test
  SOURCES
    gcn_detile_test.cc
    ../ps4/gcn/gcn_detile.cc
  DEPS delta_deps)

add_delta_test(gcn_decode_test
  SOURCES
    gcn_decode_test.cc
    ../ps4/gcn/gcn_decode.cc
  DEPS delta_deps)

add_delta_test(vk_memory_span_test
  SOURCES
    vk_memory_span_test.cc
    ../vulkan/vk_memory_span.cc
  DEPS delta_deps)

add_delta_test(vk_compute_hazard_test
  SOURCES
    vk_compute_hazard_test.cc
  DEPS delta_deps)

add_delta_test(vk_index_upload_test
  SOURCES
    vk_index_upload_test.cc
    ../vulkan/vk_index_upload.cc
  DEPS delta_deps)
