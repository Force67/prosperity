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
  add_delta_test(render_draw_test
    SOURCES render_draw_test.cc
    DEPS delta_gpu)
  target_include_directories(render_draw_test PRIVATE ${SPIRVHEADERS_INCLUDE_DIRS})
  set_tests_properties(render_draw_test PROPERTIES
    ENVIRONMENT "SDL_VIDEODRIVER=offscreen;DELTA_GPU_NOPRESENT=1;DELTA_GPU_SNAPEVERY=1000000;DELTA_GPU_SNAPEVERY_MAX=0")
endif()

add_delta_test(vk_memory_span_test
  SOURCES
    vk_memory_span_test.cc
    ../vulkan/vk_memory_span.cc
  DEPS delta_deps)

add_delta_test(render_color_mask_test
  SOURCES render_color_mask_test.cc
  DEPS delta_gpu)

add_delta_test(bc6_texture_test
  SOURCES bc6_texture_test.cc
  DEPS delta_gpu)

add_delta_test(render_compute_hazard_test
  SOURCES
    render_compute_hazard_test.cc
  DEPS delta_deps)

add_delta_test(render_index_upload_test
  SOURCES
    render_index_upload_test.cc
    ../render/index_upload.cc
  DEPS delta_deps)

# Needs a GPU: every backend without a usable device is skipped.
add_delta_test(rhi_conformance_test
  SOURCES rhi_conformance_test.cc
  DEPS delta_gpu)

# Developer tool (not a test): lowers the recompiler's SPIR-V cache through the
# OpenGL backend's GLSL path and compiles it with the driver.
if(DELTA_GPU_OPENGL)
  add_executable(rhi_lowering_corpus rhi_lowering_corpus.cc)
  target_link_libraries(rhi_lowering_corpus PRIVATE delta_gpu spirv-cross-glsl
                        ${EPOXY_LIBRARY} OpenGL::EGL OpenGL::OpenGL)
  target_include_directories(rhi_lowering_corpus PRIVATE ${EPOXY_INCLUDE_DIR})
endif()
