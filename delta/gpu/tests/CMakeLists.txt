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
    ../gcn/gcn_detile.cc
  DEPS delta_deps)

add_delta_test(gcn_decode_test
  SOURCES
    gcn_decode_test.cc
    ../gcn/gcn_decode.cc
  DEPS delta_deps)

add_delta_test(rdna_decode_test
  SOURCES
    rdna_decode_test.cc
    ../ps5/rdna/rdna_decode.cc
    ../gcn/gcn_decode.cc
  DEPS delta_deps)

add_delta_test(gcn_disasm_test
  SOURCES
    gcn_disasm_test.cc
    ../gcn/gcn_disasm.cc
    ../gcn/gcn_decode.cc
  DEPS delta_deps)

add_delta_test(gcn_audit_test
  SOURCES
    gcn_audit_test.cc
    ../gcn/gcn_audit.cc
    ../gcn/gcn_disasm.cc
    ../gcn/gcn_decode.cc
  DEPS delta_deps)

add_delta_test(pm4_test
  SOURCES pm4_test.cc
  DEPS delta_deps)

if(SPIRVTOOLS_FOUND AND SPIRVHEADERS_FOUND)
  add_delta_test(gcn_spirv_test
    SOURCES gcn_spirv_test.cc
    DEPS delta_gpu)
endif()

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
