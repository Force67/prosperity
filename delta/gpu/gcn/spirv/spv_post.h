#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * SPIRV-Tools post-processing for the direct GCN->SPIR-V backend: the optimize
 * pass the recompiler runs over freshly-emitted (naive) SPIR-V, plus
 * validation.
 *
 * The translator (gcn_spirv) models the GCN register file as Private-storage
 * variables and emits straight load/compute/store SPIR-V. Legalization passes
 * (local-variable elimination / SSA rewrite = mem2reg) turn that into SSA, then
 * performance passes fold and prune it. This is the "emit SPIR-V then optimize"
 * pipeline (vs. GCN->GLSL->shaderc).
 */

#include "base/arch.h"
#include <string>
#include <vector>

namespace gpu::gcn::spirv {

// Legalize + optimize a module. Returns the optimized binary; on failure
// returns the input unchanged (the naive SPIR-V is still valid, just
// unoptimized).
std::vector<u32> Optimize(const std::vector<u32>& spv);

// Validate against the Vulkan 1.1 environment. On failure fills *err (if
// given).
bool Validate(const std::vector<u32>& spv, std::string* err = nullptr);

// Validate + optimize, with a DISK cache keyed by the content of the incoming
// module. Optimization is ~65% of the recompiler's cost (SotC: 3110 ms of
// shader time a run, 1074 ms with the pass disabled) and its output is a pure
// function of its input, so a hit is indistinguishable from a miss except in
// time; a hit also skips validation, since nothing is stored that did not
// validate when it was produced.
// Returns false only when the module fails validation, filling *err as
// Validate does. The cache lives in DELTA_GPU_SHADER_CACHE_DIR, or
// $XDG_CACHE_HOME/ps4delta/spirv, and is disabled by DELTA_GPU_SHADER_CACHE=0.
bool Finalize(const std::vector<u32>& spv,
              std::vector<u32>* out,
              std::string* err = nullptr);

}  // namespace gpu::gcn::spirv
