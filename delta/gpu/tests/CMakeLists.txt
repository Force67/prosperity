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

add_delta_test(agc_queue_test
  SOURCES agc_queue_test.cc
  DEPS delta_gpu)

if(SPIRVTOOLS_FOUND AND SPIRVHEADERS_FOUND)
  add_delta_test(gcn_spirv_test
    SOURCES gcn_spirv_test.cc
    DEPS delta_gpu)
  add_delta_test(rdna_compute_image_test
    SOURCES rdna_compute_image_test.cc
    DEPS delta_gpu)
  add_delta_test(rdna_global_test
    SOURCES rdna_global_test.cc
    DEPS delta_gpu)
  add_delta_test(rdna_bvh_test
    SOURCES rdna_bvh_test.cc
    DEPS delta_gpu)
  add_delta_test(vk_draw_test
    SOURCES vk_draw_test.cc
    DEPS delta_gpu)
  target_include_directories(vk_draw_test PRIVATE ${SPIRVHEADERS_INCLUDE_DIRS})
  set_tests_properties(vk_draw_test PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=offscreen;DELTA_GPU_NOPRESENT=1;DELTA_GPU_SNAPEVERY=1000000;DELTA_GPU_SNAPEVERY_MAX=0")
endif()

add_delta_test(vk_memory_span_test
  SOURCES
    vk_memory_span_test.cc
    ../vulkan/vk_memory_span.cc
  DEPS delta_deps)

add_delta_test(vk_color_mask_test
  SOURCES vk_color_mask_test.cc
  DEPS delta_gpu)

add_delta_test(bc6_texture_test
  SOURCES bc6_texture_test.cc
  DEPS delta_gpu)

add_delta_test(vk_compute_hazard_test
  SOURCES
    vk_compute_hazard_test.cc
  DEPS delta_deps)

add_delta_test(vk_index_upload_test
  SOURCES
    vk_index_upload_test.cc
    ../vulkan/vk_index_upload.cc
  DEPS delta_deps)
