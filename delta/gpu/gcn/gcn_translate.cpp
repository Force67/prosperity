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
  if (!vsCode || !psCode || !vsUserData || !psUserData) return r;
  recompileSpirv(vsCode, psCode, vsUserData, psUserData, r);  // fills r, sets r.ok
  return r;
}

}  // namespace gpu::gcn
