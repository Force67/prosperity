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

// A constant buffer a shader stage reads (s_buffer_load). Bound as a UBO.
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
  std::vector<ShaderCbuf> vsCbufs; // VS UBOs (set 1, binding = .binding)
  std::vector<ShaderCbuf> psCbufs; // PS UBOs (set 1, binding = .binding)
  std::vector<ShaderTex> psTexs;   // PS samplers (set 0, binding = .binding)
  uint32_t numParams = 0;          // VS->PS interpolants (locations 0..numParams-1)
  uint8_t psMrtMask = 0;           // bit n set = PS exports to MRT color target n (0..7)
};

// Recompile a VS+PS pair. vsCode/psCode are guest pointers to the GCN code; the
// user-data arrays are the 16 user SGPRs for each stage (used only to read the
// fetch-shader pointer during translation, not the live resources).
Recompiled recompile(const uint32_t *vsCode, const uint32_t *psCode,
                     const uint32_t *vsUserData, const uint32_t *psUserData);

// A memory resource a compute shader touches (source/dest buffer or image). The
// descriptor (V#/T#) lives in the seeded register file at `baseSgpr` (user data);
// the renderer resolves its live guest base/size from the dispatch's COMPUTE_USER_DATA
// and binds a storage buffer for it. The recompiled CS accesses it by `binding`,
// computing offsets relative to the descriptor base (the storage buffer aliases the
// guest range [base, base+size), so base maps to buffer offset 0).
struct CsResource {
  uint32_t baseSgpr = 0;  // SGPR index the descriptor sits at (in user data)
  uint32_t binding = 0;   // storage-buffer binding (set 0)
  uint8_t kind = 0;       // 0 = buffer (V#/pointer), 1 = image (T#)
  bool written = false;   // dispatch writes it -> copy the storage buffer back to guest
  uint32_t minBytes = 0;  // lower bound on size from immediate-offset accesses
};

// A recompiled compute shader: the GLCompute SPIR-V + its resource-binding plan +
// the workgroup size. Cached by CS address (the plan/SPIR-V depend only on the code,
// not the per-dispatch user data, which is passed to the shader as push constants).
struct RecompiledCs {
  bool ok = false;
  std::vector<uint32_t> spirv;
  std::vector<CsResource> resources;
  uint32_t localSize[3] = {1, 1, 1};  // threads per workgroup (COMPUTE_NUM_THREAD_*)
};

// Recompile a compute shader to a Vulkan compute pipeline (GLCompute SPIR-V). csCode
// is a guest pointer to the GCN code; numThread* the workgroup size; userSgpr the
// number of user-data SGPRs seeded into s0.. (COMPUTE_PGM_RSRC2.user_sgpr); tgidEnable
// which workgroup-id dims land in the SGPRs after the user data. Returns ok=false when
// the shader uses a feature the compute backend does not implement (caller skips the
// dispatch loudly rather than corrupting memory).
RecompiledCs recompileCompute(const uint32_t *csCode, uint32_t numThreadX,
                              uint32_t numThreadY, uint32_t numThreadZ,
                              uint32_t userSgpr, uint32_t tgidEnable);

}  // namespace gpu::gcn
