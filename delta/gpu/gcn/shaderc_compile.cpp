/*
 * PS4Delta : PS4 emulation and research project
 *
 * Runtime GLSL -> SPIR-V compilation. See shaderc_compile.h.
 */

#include "shaderc_compile.h"

#include <shaderc/shaderc.h>

#include <cstdio>
#include <mutex>

namespace gpu::gcn {
namespace {

// One compiler instance for the process. shaderc_compile_into_spv is reentrant,
// so a single compiler is safe to share across threads once initialised.
shaderc_compiler_t compiler() {
  static shaderc_compiler_t c = shaderc_compiler_initialize();
  return c;
}

}  // namespace

std::vector<uint32_t> compileGlsl(bool vertexStage, const std::string &src,
                                  const char *tag) {
  shaderc_compilation_result_t res = shaderc_compile_into_spv(
      compiler(), src.c_str(), src.size(),
      vertexStage ? shaderc_glsl_vertex_shader : shaderc_glsl_fragment_shader,
      tag, "main", nullptr);

  std::vector<uint32_t> out;
  if (shaderc_result_get_compilation_status(res) ==
      shaderc_compilation_status_success) {
    const char *bytes = shaderc_result_get_bytes(res);
    size_t len = shaderc_result_get_length(res);  // bytes
    out.assign(reinterpret_cast<const uint32_t *>(bytes),
               reinterpret_cast<const uint32_t *>(bytes) + len / 4);
  } else {
    std::fprintf(stderr, "[glslc] %s: %s\n%s\n", tag,
                 shaderc_result_get_error_message(res), src.c_str());
  }
  shaderc_result_release(res);
  return out;
}

bool shadercSelfTest() {
  const std::string src = "#version 450\nvoid main(){ gl_Position = vec4(0); }";
  return !compileGlsl(true, src, "selftest").empty();
}

}  // namespace gpu::gcn
