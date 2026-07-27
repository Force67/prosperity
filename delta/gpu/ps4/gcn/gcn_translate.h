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

// Upper bound used by the compute resource planner and Vulkan staging path.
// The renderer additionally checks the selected device's descriptor limits.
inline constexpr uint32_t kMaxCsResources = 32;

// A vertex attribute recovered from the VS fetch shader, in semantic order.
struct ShaderAttr {
  uint32_t location = 0;        // GLSL `in` location == semantic index
  uint32_t num_comps = 0;       // 1..4 (from the buffer_load_format opcode)
  uint32_t table_sgpr = 0;      // fetch-table pointer or direct V# base SGPR
  uint32_t vbuf_dword_off = 0;  // dword offset of this attr's V# in the table
  bool direct_fetch = false;    // MUBUF is in the main VS, not a fetch shader
  // gfx10 unified buffer format carried by a TYPED fetch
  // (tbuffer_load_format_*), which overrides the V#'s own format. 0 = untyped
  // fetch, use the V#.
  uint32_t inst_format = 0;
  uint32_t use_pc = ~0u;  // direct/inline fetch MUBUF pc for scalar replay
};

// Set-1 UBO bindings shared by VS + PS. A shader pair whose constant buffers
// exceed this gets planned only up to the cap, and every s_buffer_load from a
// dropped base then emits nothing, leaving its destination SGPRs zero.
constexpr uint32_t kMaxCbufBindings = 16;
constexpr uint32_t kCbufDwords = 4096;

// A constant buffer a shader stage reads (s_buffer_load). Bound as a UBO.
struct ShaderCbuf {
  uint32_t binding = 0;
  uint32_t ud_sgpr = 0;  // user-data dword index of the 4-dword V# / chain root
  uint32_t num_dwords = 0;  // highest dword index read + 1 (UBO size)
  // Descriptor pointer chain (RDNA2 SMEM): when the descriptor is not directly
  // in user data but s_load'd from a chain of user-data root pointers.
  // chain_len == 0 means direct (the V# is inline at ud_sgpr). Otherwise
  // ud_sgpr is the root user-data SGPR (a pointer pair) and chain_off[0..len-1]
  // are the byte offsets dereferenced at each level; the last one addresses the
  // final 4-dword V#. The GFX7 path leaves this 0 (direct), so its behavior is
  // unchanged.
  uint32_t chain_len = 0;
  uint32_t chain_off[3] = {};
  uint32_t use_pc = ~0u;  // RDNA consumer used for draw-time scalar replay
  // The descriptor is a 2-dword flat pointer read with s_load, not a 4-dword
  // V# read with s_buffer_load. Engines that keep their constants in a shader
  // resource table (Shadow of the Colossus loads every constant as
  // s_load_dword from a table pointer chained off user data) produce these;
  // ud_sgpr is then the SGPR pair holding the pointer, which draw-time scalar
  // evaluation resolves, and num_dwords alone gives the window size.
  bool pointer = false;
};

// Set-2 storage-buffer bindings shared by VS + PS, and the window of each one
// the renderer stages. A raw MUBUF load addresses its resource with a per-lane
// index, so unlike a constant buffer there is no static bound on what it
// reads; the window is what the renderer can afford to copy per draw, and the
// shader clamps into it.
constexpr uint32_t kMaxGfxBuffers = 4;
constexpr uint32_t kGfxBufferDwords = 262144;  // 1 MB

// A raw (non-format) buffer a graphics stage reads with MUBUF: vertex data the
// VS fetches by hand rather than through the vertex-input state, a skinning
// palette, an instance table. Bound as a storage buffer at set 2, aliasing
// [V#.base, V#.base + window). The V# lives in `srsrc_sgpr` at `use_pc`, where
// draw-time scalar evaluation reads it -- it may have arrived there by s_load
// through an SRT chain, so user data alone does not name it.
struct ShaderBuffer {
  uint32_t binding = 0;
  uint32_t srsrc_sgpr = 0;
  uint32_t use_pc = 0;
};

// A texture the PS references (MIMG). Bound as a combined image sampler at
// set 0; binding order == MIMG order (matches TrackTextures).
struct ShaderTex {
  uint32_t binding = 0;
  uint32_t ud_sgpr = 0;  // PS user-data dword index of the 8-dword T#
  bool storage = false;  // image_store binding rather than a sampled image
};

struct Recompiled {
  bool ok = false;
  std::vector<uint32_t> vs_spirv;  // emitted directly from GCN
  std::vector<uint32_t> gs_spirv;  // fixed RECTLIST expansion stage
  std::vector<uint32_t> fs_spirv;
  std::vector<ShaderAttr> attrs;     // vertex inputs
  std::vector<ShaderCbuf> vs_cbufs;  // VS UBOs (set 1, binding = .binding)
  std::vector<ShaderCbuf> ps_cbufs;  // PS UBOs (set 1, binding = .binding)
  std::vector<ShaderBuffer> vs_bufs;  // VS raw buffers (set 2, = .binding)
  std::vector<ShaderBuffer> ps_bufs;  // PS raw buffers (set 2, = .binding)
  std::vector<ShaderTex> ps_texs;    // PS samplers (set 0, binding = .binding)
  uint32_t num_params = 0;           // VS->PS interpolants (locations 0..n-1)
  uint8_t ps_mrt_mask = 0;           // bit n set = PS exports MRT color n
};

// Recompile a VS+PS pair. vs_code/ps_code are guest pointers to the GCN code;
// the user-data arrays are the 16 user SGPRs for each stage (used only to read
// the fetch-shader pointer during translation, not the live resources).
Recompiled Recompile(const uint32_t* vs_code,
                     const uint32_t* ps_code,
                     const uint32_t* vs_user_data,
                     const uint32_t* ps_user_data);

// A memory resource a compute shader touches. The descriptor may be inline in
// user data or loaded through an SRT chain; `base_sgpr` names its live location
// at `use_pc`, where the command processor resolves it before dispatch. The
// recompiled CS accesses it by `binding`, computing offsets relative to the
// descriptor base (the storage buffer aliases [base, base + size)).
struct CsResource {
  uint32_t base_sgpr = 0;  // SGPR index of the live descriptor at use_pc
  uint32_t use_pc = 0;     // representative instruction consuming it
  uint32_t binding = 0;    // storage-buffer binding (set 0)
  uint8_t kind = 0;        // 0 = buffer V#, 1 = image T#, 2 = scalar pointer
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
RecompiledCs RecompileCompute(const uint32_t* cs_code,
                              uint32_t num_thread_x,
                              uint32_t num_thread_y,
                              uint32_t num_thread_z,
                              uint32_t user_sgpr,
                              uint32_t tgid_enable,
                              uint32_t lds_dwords);

// Print the instruction listing of the shader at a guest code address, tagged
// with `tag`. Diagnostic only: a renderer that has caught a target in a bad
// state (a NaN-poisoned attachment) knows the producing shader's address but
// not how to decode it, and guest shader addresses move between runs, so the
// listing has to be produced in the run that observed the problem.
void DisassembleAt(uint64_t code_address, const char* tag);

}  // namespace gpu::gcn
