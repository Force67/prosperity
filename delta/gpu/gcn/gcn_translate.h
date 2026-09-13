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
 * while/switch state machine over basic blocks; unhandled ops decline the
 * recompile rather than silently emitting approximate shaders.
 */

#include "base/arch.h"
#include <vector>

#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

// Upper bound used by the compute resource planner and Vulkan staging path.
// The renderer additionally checks the selected device's descriptor limits.
inline constexpr u32 kMaxCsResources = 128;

// Waves a shared-LDS block is allocated for. A graphics stage cannot declare
// Workgroup storage, so an NGG vertex program's LDS lives in one storage
// buffer with a block per wave; this is how many blocks that buffer holds, and
// the wave id wraps into it.
inline constexpr u32 kLdsWaves = 1024;
inline constexpr u32 kLdsMaxDwords = 4096;  // GraphicsLdsDwords' own cap
// DELTA_GPU_DSMARK's trace area: the tail of the scratch, one slot per DS
// instruction pc, so a dump says which stores actually ran.
inline constexpr u32 kLdsTraceBase = kLdsWaves * kLdsMaxDwords - 1024;
// Where a store from an inactive lane goes. LDS writes are per-lane on the
// hardware and EXEC gates them, so a lane the wave has masked off must not
// reach the block the active lanes share.
inline constexpr u32 kLdsTrashDword = kLdsWaves * kLdsMaxDwords - 1;

// A vertex attribute recovered from the VS fetch shader, in semantic order.
struct ShaderAttr {
  u32 location = 0;        // GLSL `in` location == semantic index
  u32 num_comps = 0;       // 1..4 (from the buffer_load_format opcode)
  u32 table_sgpr = 0;      // fetch-table pointer or direct V# base SGPR
  u32 vbuf_dword_off = 0;  // dword offset of this attr's V# in the table
  bool direct_fetch = false;    // MUBUF is in the main VS, not a fetch shader
  // gfx10 unified buffer format carried by a TYPED fetch
  // (tbuffer_load_format_*), which overrides the V#'s own format. 0 = untyped
  // fetch, use the V#.
  u32 inst_format = 0;
  u32 use_pc = ~0u;  // direct/inline fetch MUBUF pc for scalar replay
  // Byte offset immediate on the fetch instruction. Sony's RDNA compiler packs
  // every attribute into one V# and separates them with this, so without it all
  // of them land on the first field. Only the RDNA parser fills it in; the PS4
  // fetch shaders give each attribute its own V#.
  u32 inst_offset = 0;
  // The GCN spelling of the same thing: a typed fetch (MTBUF
  // tbuffer_load_format_*) carries dfmt/nfmt in the instruction and the
  // hardware ignores the V#'s. dfmt 0 is not a data format, so 0 means an
  // untyped (MUBUF) fetch whose format comes from the V#.
  u32 inst_dfmt = 0;
  u32 inst_nfmt = 0;
};

// Set-1 UBO bindings shared by VS + PS. A shader pair whose constant buffers
// exceed this gets planned only up to the cap, and every s_buffer_load from a
// dropped base then emits nothing, leaving its destination SGPRs zero.
// Bounded by maxDescriptorSetUniformBuffersDynamic, not by what the ISA can
// address: set 1 binds every cbuffer as a dynamic UBO, and declaring more of
// those than the device allows is an out-of-spec layout the driver is free to
// mishandle silently. 15 is what current NVIDIA parts report; the check in
// vk_upload_ring.cc still reports a device below that rather than assuming.
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
  // Dword the window STARTS at. A constant buffer larger than kCbufDwords is
  // still readable when every load into it sits within one window's span: the
  // renderer stages from base + first_dword*4 and the shader indexes relative
  // to that. Left 0 by the GFX7 planner, so the PS4 path is unchanged.
  u32 first_dword = 0;
  // Descriptor pointer chain (RDNA2 SMEM): when the descriptor is not directly
  // in user data but s_load'd from a chain of user-data root pointers.
  // chain_len == 0 means direct (the V# is inline at ud_sgpr). Otherwise
  // ud_sgpr is the root user-data SGPR (a pointer pair) and chain_off[0..len-1]
  // are the byte offsets dereferenced at each level; the last one addresses the
  // final 4-dword V#. The GFX7 path leaves this 0 (direct), so its behavior is
  // unchanged.
  u32 chain_len = 0;
  u32 chain_off[3] = {};
  u32 use_pc = ~0u;  // RDNA consumer used for draw-time scalar replay
  // The descriptor is a 2-dword flat pointer read with s_load, not a 4-dword
  // V# read with s_buffer_load. Engines that keep their constants in a shader
  // resource table (Shadow of the Colossus loads every constant as
  // s_load_dword from a table pointer chained off user data) produce these;
  // ud_sgpr is then the SGPR pair holding the pointer, which draw-time scalar
  // evaluation resolves, and num_dwords alone gives the window size.
  bool pointer = false;
  bool from_gs = false;  // split NGG stage used for scalar descriptor replay
};

// Set-2 storage-buffer bindings shared by VS + PS, and the window of each one
// the renderer stages. A raw MUBUF load addresses its resource with a per-lane
// index, so unlike a constant buffer there is no static bound on what it
// reads; the window is what the renderer can afford to copy per draw, and the
// shader clamps into it.
// kMaxGfxBuffers is the compile-time ceiling: it sizes the descriptor-layout
// and per-draw arrays, and nothing may plan a binding at or above it. The
// binding these live in is a DYNAMIC storage buffer, and
// maxDescriptorSetStorageBuffersDynamic has a Vulkan floor of only 4 -- so the
// count actually usable is a device property, not a constant. The renderer
// calls SetMaxGfxBuffers() once it knows the limit; until then the planner
// stays at the floor, which is valid everywhere.
//
// This matters because a shader is planned against the cap: SotC's deferred
// pixel shaders reference 5+ distinct raw buffers, and at a cap of 4 every
// load past the fourth was left unplanned, warned as "mubuf.ps", and took the
// whole shader down with it.
constexpr u32 kMaxGfxBuffers = 16;
constexpr u32 kMinGfxBuffers = 4;  // the Vulkan floor
// Astro Bot binds scene buffers up to 16 MiB. Truncating these to 1 MiB
// loses records even when their descriptors and indices are valid. This is
// the per-binding range; the upload ring retains its fixed total allocation.
constexpr u32 kGfxBufferDwords = 4 * 1024 * 1024;  // 16 MiB

// Planner-visible cap, in [kMinGfxBuffers, kMaxGfxBuffers].
u32 MaxGfxBuffers();
void SetMaxGfxBuffers(u32 n);

// True when the push-constant budget (>= 144 bytes) has room for each graphics
// stage's own guest code address after the two 64-byte user-data halves. The
// emitted module then reads the address for s_getpc_b64 from the push range
// per draw instead of baking it in, which is what lets the recompile cache key
// VS/PS by code CONTENT while a title streams the same shader to a fresh
// address every use. At the 128-byte Vulkan floor the address stays baked and
// programs containing s_getpc_b64 must be keyed by address. Defaults to the
// floor until the renderer reports the device limit, like MaxGfxBuffers.
bool PushCodeBase();
void SetPushBudget(u32 bytes);

// A GCN wave is 64 lanes. One lane maps to one invocation here, so a host
// subgroup narrower than that splits a wave across several of them, and the
// lockstep a wave64 shader is entitled to assume no longer holds. Reported by
// the renderer at device init; 64 (assume no split) until then.
constexpr u32 kGcnWave = 64;
u32 HostSubgroupSize();
void SetHostSubgroupSize(u32 lanes);
inline bool WaveSplitsAcrossSubgroups() {
  return HostSubgroupSize() < kGcnWave;
}

// Where a compute shader needs workgroup barriers its guest compiler was
// entitled to omit: a 64-thread threadgroup is exactly one wave on GCN, so LDS
// is coherent across it without an s_barrier, and one lane per invocation
// breaks that. Empty unless the shader is one of those AND the host subgroup
// is narrower than a wave.
struct LdsBarrierPlan {
  // Instruction indices to emit a barrier before (straight-line shaders, where
  // every point is reached by every invocation exactly once).
  std::vector<u32> at;
  // Branchy shaders instead barrier once per dispatch-loop iteration, which
  // separates accesses in different blocks. Costs a workgroup-wide "is anyone
  // still running" reduction, so it is only turned on where it is needed.
  bool lockstep = false;
};
LdsBarrierPlan PlanLdsBarriers(const Program& program,
                               const u8* reachable,
                               u32 threads_per_group);

// Per instruction: is this a point every invocation of the group reaches on
// the same dynamic iteration, so a barrier there is defined? Straight-line
// shaders are uniform throughout; under the dispatch loop only the entry
// block is, since a later block can be reached on differing iterations.
std::vector<u8> UniformPoints(const Program& program);

// A raw (non-format) buffer a graphics stage reads with MUBUF: vertex data the
// VS fetches by hand rather than through the vertex-input state, a skinning
// palette, an instance table. Bound as a storage buffer at set 2, aliasing
// [V#.base, V#.base + window). The V# lives in `srsrc_sgpr` at `use_pc`, where
// draw-time scalar evaluation reads it -- it may have arrived there by s_load
// through an SRT chain, so user data alone does not name it.
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
  // Declared with an integer sampled type. A UNORM view cannot satisfy one, so
  // even the fallback a binding takes when nothing resolves has to match.
  bool is_uint = false;
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
  // VS samplers (a vertex texture fetch: displacement, per-vertex lookup).
  // Set 0 is shared, so these are numbered after ps_texs.
  std::vector<ShaderTex> vs_texs;
  u32 num_params = 0;           // VS->PS interpolants (locations 0..n-1)
  u8 ps_mrt_mask = 0;           // bit n set = PS exports MRT color n
  // The VS backs its LDS with the shared per-wave buffer at set 3, so the
  // pipeline layout needs that set and the draw has to bind it.
  bool shared_lds = false;
};

// Time spent recompiling shaders (GCN -> SPIR-V + spirv-opt) and the number of
// recompiles, both since the last FPS report reset them. A title that streams
// its shader code re-recompiles the same shaders, and that cost shows up
// nowhere else: it is not inside a draw.
extern u64 g_ns_recomp;
extern u32 g_recomp_n;

// The same window, split by what inside a recompile spent it: SPIRV-Tools
// validation, SPIRV-Tools optimization, and how often the on-disk cache of
// optimizer output answered instead. Everything not named here is our own
// decode + emit.
extern u64 g_ns_spv_val, g_ns_spv_opt;
extern u32 g_spv_hit_n, g_spv_miss_n;

// Recompile a VS+PS pair. vs_code/ps_code are guest pointers to the GCN code;
// the user-data arrays are the 16 user SGPRs for each stage (used only to read
// the fetch-shader pointer during translation, not the live resources).
// tex_3d_mask has bit i set when PS sampler binding i is a 3D image. A 3D
// resource is invisible in the MIMG encoding (the DA bit stays 0), so it has to
// come from the caller's decoded T#s, and it must be part of the cache key: the
// same code sampled through a 2D and a 3D descriptor is two different modules.
// tex_1d_mask is the same for 1D[_ARRAY] descriptors, whose address body
// carries one fewer coordinate than the 2D case the DA bit alone suggests.
// tex_uint_mask / mrt_uint_mask mark integer-format sampled images and colour
// targets: those change the SPIR-V types (uvec4 in and out), so like the two
// above they are part of the module's identity, not per-draw state.
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
                      // Bit n set = the pass binds colour attachment n. A
                      // shader routinely exports targets the pass does not
                      // bind (a depth prepass binds none, and a single-target
                      // pass still meets exports to MRT1); declaring an Output
                      // Vulkan has nowhere to put is a write into nothing, so
                      // the module drops those outputs and keeps only the
                      // export's effect on the discard lowering. 0xFF = assume
                      // every target is bound, i.e. the old behaviour.
                      u32 mrt_bound_mask = 0xFF,
                      bool gl_clip_space = false);

// A memory resource a compute shader touches. The descriptor may be inline in
// user data or loaded through an SRT chain; `base_sgpr` names its live location
// at `use_pc`, where the command processor resolves it before dispatch. The
// recompiled CS accesses it by `binding`, computing offsets relative to the
// descriptor base (the storage buffer aliases [base, base + size)).
struct CsResource {
  u32 base_sgpr = 0;  // SGPR index of the live descriptor at use_pc
  u32 use_pc = 0;     // representative instruction consuming it
  u32 binding = 0;    // storage-buffer binding (set 0)
  u8 kind = 0;  // 0 = buffer V#, 1 = image T#, 2 = scalar pointer, 3 = BVH T#
  bool written = false;    // dispatch writes it -> copy back to guest
  // Does the dispatch READ it? A resource that is written and never read does
  // not have to be staged in from guest memory before the dispatch -- and
  // SotC's material fills are whole 4 MiB arenas of exactly that shape, so
  // uploading them is pure cost. Tracked per access and OR'd, so a
  // read-modify-write (an atomic, or a load and a store to the same buffer)
  // still reports read.
  bool read = false;
  u32 min_bytes = 0;  // lower bound on size from immediate offsets
  // Bases that depend on lane results or traversal iterations are resolved
  // from the live SGPRs through the checked guest-address map on the GPU.
  bool runtime_address = false;
  // Plain image loads/stores can share native linear integer storage across
  // differently formatted views. Other layouts keep the staged image path.
  bool runtime_image = false;
  u32 image_table_pc = ~0u;  // s_buffer_load_dwordx8 selecting this image
  bool inline_user_data = false;
  bool base_mip_only = false;
};

// A recompiled compute shader: the GLCompute SPIR-V + its resource-binding
// plan + the workgroup shape. Cache key must include the workgroup shape and
// RSRC2-derived state, not just the code address (they are baked into the
// module).
struct RecompiledCs {
  bool ok = false;
  std::vector<u32> spirv;
  std::vector<CsResource> resources;
  u32 local_size[3] = {1, 1, 1};  // threads per workgroup
  // GDS: a small global scratchpad the ds_append/ds_consume counters live in.
  // It is not guest memory, so it gets a binding of its own past the resources.
  int gds_binding = -1;
  int guest_memory_binding = -1;
  bool guest_memory_written = false;
};

// Recompile a compute shader to a Vulkan compute pipeline (GLCompute SPIR-V).
// cs_code is a guest pointer to the GCN code; num_thread_* the workgroup size;
// user_sgpr the number of user-data SGPRs seeded into s0..
// (COMPUTE_PGM_RSRC2.user_sgpr); tgid_enable which workgroup-id dims land in
// the SGPRs after the user data; lds_dwords the raw RSRC2 LDS_SIZE field (in
// 128-dword granules). Returns ok=false when the shader uses a feature the
// compute backend does not implement (caller skips the dispatch loudly rather
// than corrupting memory).
RecompiledCs RecompileCompute(const u32* cs_code,
                              u32 num_thread_x,
                              u32 num_thread_y,
                              u32 num_thread_z,
                              u32 user_sgpr,
                              u32 tgid_enable,
                              u32 lds_dwords);

// Parameters shared by the image-layout utility shader and its Vulkan caller.
// Tiled offsets, strides and table terms are bytes; linear offsets and strides
// are dwords. Pitch is in texels, words is the expanded size of one texel.
struct ImageTilingParams {
  u32 width, height, words, table, mask;
  u32 tiled_offset, tiled_stride, linear_offset, pitch, linear_stride;
  u32 detile, elem_bytes;
  // Narrow texels stay packed on the linear side: linear_offset and
  // linear_stride are then in bytes and each texel is a lane of a dword.
  u32 packed;
};

std::vector<u32> BuildImageTilingShader();

// Print the instruction listing of the shader at a guest code address, tagged
// with `tag`. Diagnostic only: a renderer that has caught a target in a bad
// state (a NaN-poisoned attachment) knows the producing shader's address but
// not how to decode it, and guest shader addresses move between runs, so the
// listing has to be produced in the run that observed the problem.
void DisassembleAt(u64 code_address, const char* tag);

}  // namespace gpu::gcn
