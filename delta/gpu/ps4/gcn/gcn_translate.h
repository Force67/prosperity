#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN (GFX7 "Liverpool") shader recompiler. Translates guest shaders directly
 * to SPIR-V (a register-VM model cleaned up by spirv-opt), plus a resource
 * binding plan the renderer uses to wire the real vertex buffers / constant
 * buffers / textures from the guest at draw time. This is the only shader
 * execution path: VS+PS pairs become Vulkan graphics pipelines, compute
 * shaders become Vulkan compute pipelines. Branchy shaders are lowered to a
 * while/switch state machine over basic blocks; unhandled ops degrade
 * gracefully (graphics) or decline the recompile (compute).
 */

#include <cstdint>
#include <vector>

namespace gpu::gcn {

// A vertex attribute recovered from the VS fetch shader, in semantic order.
struct ShaderAttr {
  uint32_t location = 0;        // GLSL `in` location == semantic index
  uint32_t num_comps = 0;       // 1..4 (from the buffer_load_format opcode)
  uint32_t table_sgpr = 0;      // VS user-data dword of the vertex-table ptr
  uint32_t vbuf_dword_off = 0;  // dword offset of this attr's V# in the table
};

// A constant buffer a shader stage reads (s_buffer_load). Bound as a UBO.
struct ShaderCbuf {
  uint32_t binding = 0;
  uint32_t ud_sgpr = 0;     // user-data dword index of the 4-dword V# / chain root
  uint32_t num_dwords = 0;  // highest dword index read + 1 (UBO size)
  // Descriptor pointer chain (RDNA2 SMEM): when the descriptor is not directly in
  // user data but s_load'd from a chain of user-data root pointers. chain_len == 0
  // means direct (the V# is inline at ud_sgpr). Otherwise ud_sgpr is the root
  // user-data SGPR (a pointer pair) and chain_off[0..len-1] are the byte offsets
  // dereferenced at each level; the last one addresses the final 4-dword V#. The
  // GFX7 path leaves this 0 (direct), so its behavior is unchanged.
  uint32_t chain_len = 0;
  uint32_t chain_off[3] = {};
};

// A texture the PS references (MIMG). Bound as a combined image sampler at
// set 0; binding order == MIMG order (matches TrackTextures).
struct ShaderTex {
  uint32_t binding = 0;
  uint32_t ud_sgpr = 0;  // PS user-data dword index of the 8-dword T#
};

struct Recompiled {
  bool ok = false;
  std::vector<uint32_t> vs_spirv;  // emitted directly from GCN
  std::vector<uint32_t> gs_spirv;  // fixed RECTLIST expansion stage
  std::vector<uint32_t> fs_spirv;
  std::vector<ShaderAttr> attrs;      // vertex inputs
  std::vector<ShaderCbuf> vs_cbufs;   // VS UBOs (set 1, binding = .binding)
  std::vector<ShaderCbuf> ps_cbufs;   // PS UBOs (set 1, binding = .binding)
  std::vector<ShaderTex> ps_texs;     // PS samplers (set 0, binding = .binding)
  uint32_t num_params = 0;            // VS->PS interpolants (locations 0..n-1)
  uint8_t ps_mrt_mask = 0;            // bit n set = PS exports MRT color n
};

// Recompile a VS+PS pair. vs_code/ps_code are guest pointers to the GCN code;
// the user-data arrays are the 16 user SGPRs for each stage (used only to read
// the fetch-shader pointer during translation, not the live resources).
Recompiled Recompile(const uint32_t* vs_code, const uint32_t* ps_code,
                     const uint32_t* vs_user_data,
                     const uint32_t* ps_user_data);

// A memory resource a compute shader touches (source/dest buffer or image).
// The descriptor (V#/T#) lives in the seeded register file at `base_sgpr`
// (user data); the renderer resolves its live guest base/size from the
// dispatch's COMPUTE_USER_DATA and binds a storage buffer for it. The
// recompiled CS accesses it by `binding`, computing offsets relative to the
// descriptor base (the storage buffer aliases the guest range
// [base, base + size), so base maps to buffer offset 0).
struct CsResource {
  uint32_t base_sgpr = 0;  // SGPR index the descriptor sits at (in user data)
  uint32_t binding = 0;    // storage-buffer binding (set 0)
  uint8_t kind = 0;        // 0 = buffer (V#/pointer), 1 = image (T#)
  bool written = false;    // dispatch writes it -> copy back to guest
  uint32_t min_bytes = 0;  // lower bound on size from immediate offsets
};

// A recompiled compute shader: the GLCompute SPIR-V + its resource-binding
// plan + the workgroup shape. Cache key must include the workgroup shape and
// RSRC2-derived state, not just the code address (they are baked into the
// module).
struct RecompiledCs {
  bool ok = false;
  std::vector<uint32_t> spirv;
  std::vector<CsResource> resources;
  uint32_t local_size[3] = {1, 1, 1};  // threads per workgroup
};

// Recompile a compute shader to a Vulkan compute pipeline (GLCompute SPIR-V).
// cs_code is a guest pointer to the GCN code; num_thread_* the workgroup size;
// user_sgpr the number of user-data SGPRs seeded into s0..
// (COMPUTE_PGM_RSRC2.user_sgpr); tgid_enable which workgroup-id dims land in
// the SGPRs after the user data; lds_dwords the raw RSRC2 LDS_SIZE field (in
// 128-dword granules). Returns ok=false when the shader uses a feature the
// compute backend does not implement (caller skips the dispatch loudly rather
// than corrupting memory).
RecompiledCs RecompileCompute(const uint32_t* cs_code, uint32_t num_thread_x,
                              uint32_t num_thread_y, uint32_t num_thread_z,
                              uint32_t user_sgpr, uint32_t tgid_enable,
                              uint32_t lds_dwords);

}  // namespace gpu::gcn
