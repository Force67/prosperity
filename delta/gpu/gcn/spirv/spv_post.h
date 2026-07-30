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

#include <cstdint>
#include <string>
#include <vector>

namespace gpu::gcn::spirv {

// Legalize + optimize a module. Returns the optimized binary; on failure
// returns the input unchanged (the naive SPIR-V is still valid, just
// unoptimized).
std::vector<uint32_t> Optimize(const std::vector<uint32_t>& spv);

// Validate against the Vulkan 1.1 environment. On failure fills *err (if
// given).
bool Validate(const std::vector<uint32_t>& spv, std::string* err = nullptr);

}  // namespace gpu::gcn::spirv
