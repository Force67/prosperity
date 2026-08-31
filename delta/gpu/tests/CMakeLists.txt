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

# The RHI's decisions, asserted against the recording device: no GPU needed, so
# this one runs everywhere.
add_delta_test(rhi_null_test
  SOURCES
    rhi_null_test.cc
    ../rhi/null_device.cc
    ../rhi/types.cc
  DEPS delta_deps)

# The RHI's backends, compared against each other on a real device. The shaders
# are one HLSL file compiled two ways, so glslc has to be present for the
# SPIR-V half; without it the test is not registered at all rather than
# silently testing one backend.
find_program(GLSLC glslc)
if(GLSLC)
  # The HLSL as a C string, so the D3D12 backend compiles the same text glslc
  # was given.
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/rhi_conformance.hlsl _rhi_hlsl)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rhi_conformance_hlsl.h
    "// Generated from rhi_conformance.hlsl. Do not edit.\n"
    "#pragma once\n"
    "static const char kConformanceHlsl[] = R\"HLSL(\n${_rhi_hlsl})HLSL\";\n")

  set(_rhi_spv)
  foreach(_entry IN ITEMS
      "vs_solid:vertex" "ps_solid:fragment" "ps_mrt:fragment"
      "vs_tex:vertex" "ps_tex:fragment" "cs_fill:compute")
    string(REPLACE ":" ";" _parts ${_entry})
    list(GET _parts 0 _name)
    list(GET _parts 1 _stage)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_name}.spv.inc
      COMMAND ${GLSLC} -x hlsl -DTARGET_VULKAN=1 -fshader-stage=${_stage}
              -fentry-point=${_name} -mfmt=c
              ${CMAKE_CURRENT_SOURCE_DIR}/rhi_conformance.hlsl
              -o ${CMAKE_CURRENT_BINARY_DIR}/${_name}.spv.inc
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rhi_conformance.hlsl
      COMMENT "glslc ${_name} (HLSL -> SPIR-V)")
    list(APPEND _rhi_spv ${CMAKE_CURRENT_BINARY_DIR}/${_name}.spv.inc)
  endforeach()
  add_custom_target(rhi_conformance_shaders DEPENDS ${_rhi_spv})

  add_delta_test(rhi_conformance_test
    SOURCES rhi_conformance_test.cc
    DEPS delta_gpu)
  add_dependencies(rhi_conformance_test rhi_conformance_shaders)
  target_include_directories(rhi_conformance_test
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else()
  message(STATUS "glslc not found; the RHI conformance test is not built")
endif()
