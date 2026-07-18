#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) shader recompiler facade. Translates PS5 guest shaders
 * (decoded by rdna_decode) directly to SPIR-V and returns the same
 * gpu::gcn::Recompiled binding plan the shared Vulkan renderer consumes, so the
 * whole gpu/vk_render path is reused unchanged.
 *
 * The translator reuses the shared gpu::gcn SPIR-V backend: the register-file
 * model (gpu::gcn::Translator), the scalar/vector ALU emitters, exports, and
 * constant-buffer plumbing. Only the RDNA2-specific per-instruction dispatch
 * (field layouts + opcode remap) and the SMEM constant-buffer decode live in
 * rdna_translate.cpp; everything downstream is the GFX7 path's code.
 */

#include <cstdint>

#include "ps4/gcn/gcn_translate.h"

namespace gpu::rdna {

// Recompile an RDNA2 VS+PS pair. vs_code/ps_code are guest pointers to the
// RDNA2 bytecode; the user-data arrays are the shader-stage user SGPRs (used to
// read the fetch-shader pointer during translation). On gfx10.3 the "VS" is the
// merged ES/GS NGG vertex program (read from the GS SH block). Returns a
// gpu::gcn::Recompiled (r.ok == false when a required feature is unsupported).
gpu::gcn::Recompiled Recompile(const uint32_t* vs_code, const uint32_t* ps_code,
                               const uint32_t* vs_user_data,
                               const uint32_t* ps_user_data);

}  // namespace gpu::rdna
