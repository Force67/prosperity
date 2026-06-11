#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN (GFX6/7 "Liverpool") shader recompiler. Translates a guest vertex + pixel
 * shader pair directly to SPIR-V (a register-VM model cleaned up by spirv-opt), plus
 * a resource binding plan the renderer uses to wire the real vertex buffers /
 * constant buffers / textures from the guest at draw time. Replaces the old heuristic
 * quad path with the shaders the game actually runs.
 *
 * Scope: the straight-line VS/PS patterns 2D titles like Isaac use (vertex fetch +
 * MVP transform + export; interpolate + texture sample + modulate + export). Control
 * flow is not reconstructed (single basic block); unhandled ops degrade gracefully.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace gpu::gcn {

// A vertex attribute recovered from the VS fetch shader, in semantic/location order.
struct ShaderAttr {
  uint32_t location = 0;    // GLSL `in` location == semantic index
  uint32_t numComps = 0;    // 1..4 (from the buffer_load_format opcode)
  uint32_t tableSgpr = 0;   // VS user-data dword index of the vertex-buffer-table ptr
  uint32_t vbufDwordOff = 0;// dword offset of this attribute's V# within that table
};

// A constant buffer the VS reads (s_buffer_load). Bound as a UBO.
struct ShaderCbuf {
  uint32_t binding = 0;
  uint32_t udSgpr = 0;      // VS user-data dword index of the 4-dword V# (cbuffer ptr)
  uint32_t numDwords = 0;   // highest dword index read + 1 (UBO size)
};

// A texture the PS samples (image_sample). Bound as a sampler2D.
struct ShaderTex {
  uint32_t binding = 0;
  uint32_t udSgpr = 0;      // PS user-data dword index of the 8-dword T#
};

struct Recompiled {
  bool ok = false;
  std::vector<uint32_t> vsSpirv;  // emitted directly from GCN (empty on failure)
  std::vector<uint32_t> fsSpirv;
  std::vector<ShaderAttr> attrs;   // vertex inputs
  std::vector<ShaderCbuf> vsCbufs; // VS UBOs (set 0, binding = .binding)
  std::vector<ShaderTex> psTexs;   // PS samplers (set 0, binding = .binding)
  uint32_t numParams = 0;          // VS->PS interpolants (locations 0..numParams-1)
  uint8_t psMrtMask = 0;           // bit n set = PS exports to MRT color target n (0..7)
};

// Recompile a VS+PS pair. vsCode/psCode are guest pointers to the GCN code; the
// user-data arrays are the 16 user SGPRs for each stage (used only to read the
// fetch-shader pointer during translation, not the live resources).
Recompiled recompile(const uint32_t *vsCode, const uint32_t *psCode,
                     const uint32_t *vsUserData, const uint32_t *psUserData);

}  // namespace gpu::gcn
