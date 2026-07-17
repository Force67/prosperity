/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN shader recompiler entry point. Recompilation goes GCN -> SPIR-V directly
 * (delta/gpu/gcn/spirv), with a SPIRV-Tools optimize pass; there is no GLSL
 * intermediate. This file is just the public recompile() facade over that backend.
 */

#include "gcn_translate.h"
#include "spirv/gcn_spirv.h"

namespace gpu::gcn {

Recompiled recompile(const uint32_t *vsCode, const uint32_t *psCode,
                     const uint32_t *vsUserData, const uint32_t *psUserData) {
  Recompiled r;
  if (!vsCode || !vsUserData || !psUserData) return r;
  recompileSpirv(vsCode, psCode, vsUserData, psUserData, r);  // fills r, sets r.ok
  return r;
}

RecompiledCs recompileCompute(const uint32_t *csCode, uint32_t numThreadX,
                              uint32_t numThreadY, uint32_t numThreadZ,
                              uint32_t userSgpr, uint32_t tgidEnable) {
  RecompiledCs r;
  if (!csCode) return r;
  recompileComputeSpirv(csCode, numThreadX, numThreadY, numThreadZ, userSgpr,
                        tgidEnable, r);  // fills r, sets r.ok
  return r;
}

}  // namespace gpu::gcn
