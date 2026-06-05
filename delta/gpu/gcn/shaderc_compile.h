#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Runtime GLSL -> SPIR-V compilation via shaderc. The GCN recompiler emits GLSL
 * that we hand to the driver as SPIR-V; this is the back end of that pipeline.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace gpu::gcn {

// Compile a GLSL shader to SPIR-V words. `tag` is a name used in diagnostics.
// Returns the SPIR-V words, or an empty vector on failure (the shaderc error
// message and the source are logged to stderr with a [glslc] prefix).
std::vector<uint32_t> compileGlsl(bool vertexStage, const std::string &src,
                                  const char *tag);

// Self-test: compile a trivial valid vertex shader; true if it produced words.
bool shadercSelfTest();

}  // namespace gpu::gcn
