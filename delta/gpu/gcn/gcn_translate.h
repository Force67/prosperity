#pragma once

/*
 * PS4Delta: GCN (GFX7) shader recompiler. Guest shaders become SPIR-V plus a
 * resource-binding plan the renderer wires at draw time. Branchy shaders
 * lower to a while/switch state machine; unhandled ops decline the recompile.
 */

#include "base/arch.h"
#include <vector>

#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

// Compute planner bound; the renderer also checks device descriptor limits.
inline constexpr u32 kMaxCsResources = 128;

// Wave blocks in the shared-LDS storage buffer; the wave id wraps into it.
inline constexpr u32 kLdsWaves = 1024;
inline constexpr u32 kLdsMaxDwords = 4096;  // GraphicsLdsDwords' own cap
// DELTA_GPU_DSMARK trace area, one slot per DS instruction pc.
inline constexpr u32 kLdsTraceBase = kLdsWaves * kLdsMaxDwords - 1024;
// Sink for stores from EXEC-masked (inactive) lanes.
inline constexpr u32 kLdsTrashDword = kLdsWaves * kLdsMaxDwords - 1;

// A vertex attribute recovered from the VS fetch shader, in semantic order.
struct ShaderAttr {
  u32 location = 0;        // GLSL `in` location == semantic index
  u32 num_comps = 0;       // 1..4 (from the buffer_load_format opcode)
  u32 table_sgpr = 0;      // fetch-table pointer or direct V# base SGPR
  u32 vbuf_dword_off = 0;  // dword offset of this attr's V# in the table
  bool direct_fetch = false;    // MUBUF is in the main VS, not a fetch shader
  u32 inst_format = 0;  // gfx10 typed-fetch buffer format; 0 = use the V#'s
  u32 use_pc = ~0u;  // direct/inline fetch MUBUF pc for scalar replay
  u32 inst_offset = 0;  // field separator when RDNA packs attrs into one V#
  u32 inst_dfmt = 0;  // MTBUF typed-fetch format; 0 = untyped, use the V#
  u32 inst_nfmt = 0;
};

// Set-1 dynamic UBOs shared by VS + PS, bounded by
// maxDescriptorSetUniformBufferDynamic; loads past the cap emit nothing.
constexpr u32 kMaxCbufBindings = 15;
// Mesh pipelines address staged windows through one storage buffer and a
// small offset table, avoiding the limit on dynamic UBO descriptors.
constexpr u32 kIndirectCbufBindings = 32;
constexpr u32 kIndirectGsUserDataAddr = kIndirectCbufBindings + 2 * 32;
constexpr u32 kIndirectDrawDwords = kIndirectGsUserDataAddr + 4;
constexpr u32 kCbufDwords = 4096;

// A constant buffer a shader stage reads (s_buffer_load). Bound as a UBO.
struct ShaderCbuf {
  u32 binding = 0;
  u32 ud_sgpr = 0;  // user-data dword index of the 4-dword V# / chain root
  u32 num_dwords = 0;  // dwords the window spans (UBO size)
  u32 first_dword = 0;  // window start, for staging one cbuf wider than kCbufDwords
  u32 chain_len = 0;  // RDNA2 SRT chain: 0 = direct V#, else root ptr + chain_off derefs
  u32 chain_off[3] = {};
  u32 use_pc = ~0u;  // RDNA consumer used for draw-time scalar replay
  bool pointer = false;  // 2-dword flat pointer via s_load (SRT), not a 4-dword V#
  bool from_gs = false;  // split NGG stage used for scalar descriptor replay
};

// Set-2 dynamic SSBOs for raw MUBUF loads; the usable count is a device
// property via SetMaxGfxBuffers(), planned at the floor until then.
constexpr u32 kMaxGfxBuffers = 16;
constexpr u32 kMinGfxBuffers = 4;  // the Vulkan floor
constexpr u32 kGfxBufferDwords = 4 * 1024 * 1024;  // 16 MiB (Astro Bot scene buffers)

// Planner-visible cap, in [kMinGfxBuffers, kMaxGfxBuffers].
u32 MaxGfxBuffers();
void SetMaxGfxBuffers(u32 n);

// Push constants carry the guest code address, so the cache can key by content.
bool PushCodeBase();
void SetPushBudget(u32 bytes);

// Waves split across host subgroups narrower than this; 64 until reported.
constexpr u32 kGcnWave = 64;
u32 HostSubgroupSize();
void SetHostSubgroupSize(u32 lanes);
inline bool WaveSplitsAcrossSubgroups() {
  return HostSubgroupSize() < kGcnWave;
}

// Barriers a 64-thread group can omit on GCN but a split host subgroup needs.
struct LdsBarrierPlan {
  std::vector<u32> at;  // barrier points in straight-line shaders
  bool lockstep = false;  // branchy shaders: barrier per dispatch-loop iteration
};
LdsBarrierPlan PlanLdsBarriers(const Program& program,
                               const u8* reachable,
                               u32 threads_per_group);

// Per instruction: does every invocation reach it on the same iteration?
std::vector<u8> UniformPoints(const Program& program);

// Raw MUBUF buffer (hand-fetched vertex data, skinning, instance tables) at
// set 2; the V# may have arrived via SRT, hence srsrc_sgpr/use_pc.
struct ShaderBuffer {
  u32 binding = 0;
  u32 srsrc_sgpr = 0;
  u32 use_pc = 0;
  bool from_gs = false;
};

// A texture the PS references (MIMG). Bound as a combined image sampler at
// set 0; binding order == MIMG order (matches TrackTextures).
struct ShaderTex {
  u32 binding = 0;
  u32 ud_sgpr = 0;  // PS user-data dword index of the 8-dword T#
  bool storage = false;  // image_store binding rather than a sampled image
  bool is_3d = false;    // volume image (T# type SQ_RSRC_IMG_3D)
  bool is_1d = false;    // 1D image (T# type SQ_RSRC_IMG_1D[_ARRAY])
  bool is_uint = false;  // integer sampled type; even the fallback binding must match
};

struct Recompiled {
  bool ok = false;
  std::vector<u32> vs_spirv;  // emitted directly from GCN
  std::vector<u32> mesh_spirv;  // merged NGG geometry, replaces VS/GS
  u32 mesh_input_primitives = 1;  // input primitives consumed per workgroup
  u32 mesh_threads = 0, mesh_shared_bytes = 0;
  u32 mesh_vertices = 0, mesh_primitives = 0;
  bool indirect_cbufs = false;
  std::vector<u32> gs_spirv;  // fixed RECTLIST expansion stage
  std::vector<u32> fs_spirv;
  std::vector<ShaderAttr> attrs;     // vertex inputs
  std::vector<ShaderCbuf> vs_cbufs;  // VS UBOs (set 1, binding = .binding)
  std::vector<ShaderCbuf> ps_cbufs;  // PS UBOs (set 1, binding = .binding)
  std::vector<ShaderBuffer> vs_bufs;  // VS raw buffers (set 2, = .binding)
  std::vector<ShaderBuffer> ps_bufs;  // PS raw buffers (set 2, = .binding)
  std::vector<ShaderTex> ps_texs;    // PS samplers (set 0, binding = .binding)
  std::vector<ShaderTex> vs_texs;  // vertex texture fetch, numbered after ps_texs
  u32 num_params = 0;           // VS->PS interpolants (locations 0..n-1)
  u8 ps_mrt_mask = 0;           // bit n set = PS exports MRT color n
  bool shared_lds = false;  // VS LDS lives in the set-3 per-wave buffer
};

// Recompile cost since the last FPS reset; invisible in draw timings.
extern u64 g_ns_recomp;
extern u32 g_recomp_n;

// The same window split into SPIRV-Tools validate/opt and optimizer-cache hits.
extern u64 g_ns_spv_val, g_ns_spv_opt;
extern u32 g_spv_hit_n, g_spv_miss_n;

// Recompile a VS+PS pair. The masks name descriptor shapes the MIMG encoding
// cannot see (3D/1D/uint); they change emitted types, so they key the cache.
Recompiled Recompile(const u32* vs_code,
                      const u32* ps_code,
                      const u32* vs_user_data,
                      const u32* ps_user_data,
                      u32 ps_input_ena = 0,
                      const u32* ps_in_cntl = nullptr,
                      u32 ps_num_interp = 0,
                      u32 tex_3d_mask = 0,
                      u32 tex_1d_mask = 0,
                      u32 tex_uint_mask = 0,
                      u32 mrt_uint_mask = 0,
                      u32 mrt_bound_mask = 0xFF,  // bit n = pass binds colour n; rest dropped
                      bool gl_clip_space = false);

// A CS memory resource; base_sgpr/use_pc locate the possibly-SRT-chained
// descriptor for the command processor to resolve at dispatch.
struct CsResource {
  u32 base_sgpr = 0;  // SGPR index of the live descriptor at use_pc
  u32 use_pc = 0;     // representative instruction consuming it
  u32 binding = 0;    // storage-buffer binding (set 0)
  u8 kind = 0;  // 0 = buffer V#, 1 = image T#, 2 = scalar pointer, 3 = BVH T#
  bool written = false;    // dispatch writes it -> copy back to guest
  bool read = false;  // written-only resources skip the staging upload (SotC fills)
  u32 min_bytes = 0;  // lower bound on size from immediate offsets
  bool runtime_address = false;  // base resolved on GPU via the guest-address map
  bool runtime_image = false;  // linear integer storage shared across views
  u32 image_table_pc = ~0u;  // s_buffer_load_dwordx8 selecting this image
  bool inline_user_data = false;
  bool base_mip_only = false;
};

// A recompiled compute shader; the cache key includes workgroup shape + RSRC2 state.
struct RecompiledCs {
  bool ok = false;
  std::vector<u32> spirv;
  std::vector<CsResource> resources;
  u32 local_size[3] = {1, 1, 1};  // threads per workgroup
  int gds_binding = -1;  // GDS scratchpad (ds_append/consume), not guest memory
  int guest_memory_binding = -1;
  bool guest_memory_written = false;
};

// Recompile a compute shader; ok=false on unimplemented features, which the
// caller skips loudly rather than corrupting memory.
RecompiledCs RecompileCompute(const u32* cs_code,
                              u32 num_thread_x,
                              u32 num_thread_y,
                              u32 num_thread_z,
                              u32 user_sgpr,
                              u32 tgid_enable,
                              u32 lds_dwords);

// Image-layout shader params; tiled terms in bytes, linear in dwords.
struct ImageTilingParams {
  u32 width, height, words, table, mask;
  u32 tiled_offset, tiled_stride, linear_offset, pitch, linear_stride;
  u32 detile, elem_bytes;
  u32 packed;  // narrow texels: linear side in bytes, one texel per dword lane
};

std::vector<u32> BuildImageTilingShader();

// Diagnostic: disassemble the shader at a guest address (they move between runs).
void DisassembleAt(u64 code_address, const char* tag);

}  // namespace gpu::gcn
