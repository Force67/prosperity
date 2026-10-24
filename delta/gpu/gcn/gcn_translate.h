#pragma once

#include "base/arch.h"

#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

bool DirectGcnEnabled();
// Address of the vertex shader DELTA_GPU_GCN_VSVGPR_VS probes, or 0.
u64 VsProbeShader();
// Vertex index bits of a probe slot (DELTA_GPU_GCN_VSVGPR_VBITS); the rest of
// the 19 slot bits are the instance.
u32 VsProbeVertexBits();

void ResetShaderWork();
void PostShaderWork(base::Function<void()> job);

inline constexpr u32 kMaxCsResources = 128;
inline constexpr u32 kCsTexBindingBase = 192;

inline constexpr u32 kLdsWaves = 1024;
inline constexpr u32 kLdsMaxDwords = 4096;
inline constexpr u32 kLdsTraceBase = kLdsWaves * kLdsMaxDwords - 1024;
inline constexpr u32 kLdsTrashDword = kLdsWaves * kLdsMaxDwords - 1;

struct ShaderAttr {
  u32 location = 0;
  u32 num_comps = 0;
  u32 table_sgpr = 0;
  u32 vbuf_dword_off = 0;
  bool direct_fetch = false;
  u32 inst_format = 0;
  u32 use_pc = ~0u;
  u32 inst_offset = 0;
  u32 inst_dfmt = 0;
  u32 inst_nfmt = 0;
  bool per_instance = false;
};

constexpr u32 kMaxCbufBindings = 15;
constexpr u32 kIndirectCbufBindings = 64;
constexpr u32 kIndirectGsUserDataAddr = kIndirectCbufBindings + 2 * 32;
constexpr u32 kIndirectDrawDwords = kIndirectGsUserDataAddr + 4;
constexpr u32 kCbufDwords = 4096;

struct ShaderCbuf {
  u32 binding = 0;
  u32 ud_sgpr = 0;
  u32 num_dwords = 0;
  u32 first_dword = 0;
  u32 chain_len = 0;
  u32 chain_off[3] = {};
  u32 use_pc = ~0u;
  bool pointer = false;
  u32 version = ~0u;
  bool from_gs = false;
};

constexpr u32 kMaxGfxBuffers = 16;
constexpr u32 kMinGfxBuffers = 4;
constexpr u32 kGfxBufferDwords = 4 * 1024 * 1024;

u32 MaxGfxBuffers();
void SetMaxGfxBuffers(u32 n);

bool PushCodeBase();
void SetPushBudget(u32 bytes);

constexpr u32 kGcnWave = 64;
u32 HostSubgroupSize();
void SetHostSubgroupSize(u32 lanes);
inline bool WaveSplitsAcrossSubgroups() {
  return HostSubgroupSize() < kGcnWave;
}

struct LdsBarrierPlan {
  base::Vector<u32> at;
  bool lockstep = false;
  bool orders_memory = false;
};
LdsBarrierPlan PlanLdsBarriers(const Program& program,
                               const u8* reachable,
                               u32 threads_per_group,
                               const base::HashMap<u32, u32>* bindings =
                                   nullptr,
                               u64 shared_mask = 0);

base::Vector<u8> UniformPoints(const Program& program);

struct ShaderBuffer {
  u32 binding = 0;
  u32 srsrc_sgpr = 0;
  u32 use_pc = 0;
  bool from_gs = false;
};

struct ShaderTex {
  u32 binding = 0;
  u32 ud_sgpr = 0;
  bool storage = false;
  bool is_3d = false;
  bool is_1d = false;
  bool is_uint = false;
};

struct GsPipeline {
  const u32* es_code = nullptr;
  const u32* gs_code = nullptr;
  const u32* es_user_data = nullptr;
  u32 es_user_sgprs = 0;
  u32 gs_user_sgprs = 0;
  u32 input_prim = 4;
  u32 out_prim = 2;
  u32 max_vert_out = 0;
  u32 esgs_dwords = 0;
  u32 gsvs_dwords = 0;
  u32 instances = 1;
};

struct Recompiled {
  bool ok = false;
  bool guest_gs = false;
  bool writes_layer = false;
  base::Vector<u32> vs_spirv;
  base::Vector<u32> mesh_spirv;
  u32 mesh_input_primitives = 1;
  u32 mesh_threads = 0, mesh_shared_bytes = 0;
  u32 mesh_vertices = 0, mesh_primitives = 0;
  bool indirect_cbufs = false;
  bool direct = false;
  bool direct_vs = false;
  bool bindless = false;
  bool direct_reads = false;
  base::Vector<u32> gs_spirv;
  base::Vector<u32> fs_spirv;
  base::Vector<ShaderAttr> attrs;
  base::Vector<ShaderCbuf> vs_cbufs;
  base::Vector<ShaderCbuf> gs_cbufs;
  base::Vector<ShaderCbuf> ps_cbufs;
  base::Vector<ShaderBuffer> vs_bufs;
  base::Vector<ShaderBuffer> ps_bufs;
  base::Vector<ShaderTex> ps_texs;
  base::Vector<ShaderTex> vs_texs;
  u32 num_params = 0;
  u8 ps_mrt_mask = 0;
  bool shared_lds = false;
};

constexpr u32 kColFormatUnknown = 0xFFFFFFFFu;

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
                     u32 mrt_bound_mask = 0xFF,
                     bool gl_clip_space = false,
                     const GsPipeline* gs = nullptr,
                     u64 int_attr_mask = 0,
                     u32 col_format = kColFormatUnknown,
                     u32 tex_cube_mask = 0);

// In `mrt_bound_mask`: target 0 blends with a second source, so the MRT1
// export is that source (location 0, index 1) instead of a target of its own.
constexpr u32 kMrtDualSource = 1u << 8;

constexpr u32 kUnknownStoreExtent = ~0u;

struct CsResource {
  u32 base_sgpr = 0;
  u32 use_pc = 0;
  u32 binding = 0;
  u8 kind = 0;
  bool written = false;
  bool read = false;
  u32 min_bytes = 0;
  u32 store_extent = 0;
  bool runtime_address = false;
  bool runtime_image = false;
  u32 image_table_pc = ~0u;
  bool inline_user_data = false;
  bool base_mip_only = false;
};

struct RecompiledCs {
  bool ok = false;
  base::Vector<u32> spirv;
  base::Vector<CsResource> resources;
  u32 local_size[3] = {1, 1, 1};
  int gds_binding = -1;
  int guest_memory_binding = -1;
  bool guest_memory_written = false;
  u64 native_candidates = 0;
  // The candidates sampled as arrays (MIMG DA), which need an arrayed view.
  u64 native_arrayed = 0;
  // One wave split over host subgroups, so a dispatch whose bindings alias
  // needs the variant compiled with those bindings as shared_mask.
  bool orders_shared_memory = false;
  base::Vector<int> textures;
  u64 textures_3d = 0;
};

RecompiledCs RecompileCompute(const u32* cs_code,
                              u32 num_thread_x,
                              u32 num_thread_y,
                              u32 num_thread_z,
                              u32 user_sgpr,
                              u32 tgid_enable,
                              u32 lds_dwords,
                              u64 native_mask = 0,
                              u64 uint_mask = 0,
                              u64 shared_mask = 0,
                              u64 cube_mask = 0,
                              const base::Vector<u32>& formats = {});

struct ImageTilingParams {
  u32 width, height, words, table, mask;
  u32 tiled_offset, tiled_stride, linear_offset, pitch, linear_stride;
  u32 detile, elem_bytes;
  u32 packed;
  u32 depth16;
  u32 pack11;
};

base::Vector<u32> BuildImageTilingShader();

base::Vector<u32> BuildMeshIndirectShader();

enum class IndexConversionMode { kDirect, kIndirect, kIndirectStrided };
constexpr u32 kIndexConversionGroups = 2048;
base::Vector<u32> BuildIndexConversionShader(
    u32 index_type,
    bool quads,
    IndexConversionMode mode = IndexConversionMode::kDirect);

base::Vector<u32> BuildClearPredicateShader();

// Publishes a dispatch's result word by word: where result != shadow (the
// staged bytes before the dispatch) the word goes to guest, elsewhere the
// result adopts the guest's word; the shadow then holds what guest holds.
// Push constants: the word offsets of result, shadow and guest, then the
// word count.
constexpr u32 kWordMergeGroup = 64;
base::Vector<u32> BuildWordMergeShader();

void DisassembleAt(u64 code_address, const char* tag);
base::String ListingAt(u64 code_address, u32 max_dwords);

}  // namespace gpu::gcn
