#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN (GFX6/7 "Liverpool") -> SPIR-V translator. Direct backend: emits SPIR-V
 * from the GCN bytecode (no GLSL/shaderc), then a spirv-opt pass legalises and
 * optimises it (see spv_post). Produces the same resource-binding plan and the
 * same pipeline interface (push-constant cbuffer, set-0 samplers, vertex inputs
 * by location) as the GLSL backend, so it is a drop-in alternative.
 *
 * Models the GCN register file as Private-storage uint arrays (sgpr[128],
 * vgpr[256]); float ops go through OpBitcast, exactly mirroring the GLSL Ff/Uf
 * helpers. spirv-opt's SSA rewrite promotes the register file out of memory.
 */

#include "../gcn_translate.h"  // Recompiled / ShaderAttr / ShaderCbuf / ShaderTex

namespace gpu::gcn {

// Recompile a VS+PS pair directly to SPIR-V. Fills r.vsSpirv/r.fsSpirv +
// attrs/vsCbufs/psTexs/numParams and sets r.ok. Returns r.ok. On any failure the
// caller can fall back to the GLSL backend.
bool recompileSpirv(const uint32_t *vsCode, const uint32_t *psCode,
                    const uint32_t *vsUserData, const uint32_t *psUserData,
                    Recompiled &r);

}  // namespace gpu::gcn
