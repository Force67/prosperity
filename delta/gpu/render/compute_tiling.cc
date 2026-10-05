/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The GPU tiling shader: converts gfx10 images between their tiled guest
// layout and the linear layout dispatches bind.

#include "gpu/render/compute_internal.h"

#include <cstdio>
#include <cstring>

#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/renderer_state.h"
#include "gpu/render/tiling.h"
#include "guest/session.h"

namespace gpu::render::cs {

#ifdef DELTA_HAVE_SPIRV_BACKEND
base::HashMap<u64, TileTable> g_tile_tables;

TilingState g_tiling;

bool InitTiling() {
  auto& s = g_tiling;
  if (s.pipe)
    return true;
  static bool attempted = false;
  if (attempted)
    return false;
  attempted = true;
  const auto code = gcn::BuildImageTilingShader();
  if (code.empty())
    return false;
  rhi::BindGroupLayoutDesc set;
  for (u32 i = 0; i < 3; i++)
    set.bindings.push_back(
        {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
  set.push = true;
  s.set_layout = Device().CreateBindGroupLayout(set);
  if (!s.set_layout)
    return false;
  rhi::PipelineLayoutDesc layout;
  layout.groups = {s.set_layout};
  layout.push_constant_bytes = sizeof(gcn::ImageTilingParams);
  layout.push_constant_stages = rhi::kStageCompute;
  s.layout = Device().CreatePipelineLayout(layout);
  if (!s.layout)
    return false;
  rhi::ComputePipelineDesc pipe;
  pipe.layout = s.layout;
  pipe.code = rhi::Code(code);
  s.pipe = Device().CreateComputePipeline(pipe);
  return s.pipe != nullptr;
}

bool ConvertGfx10ImageImpl(const gcn::TextureLayout32& tiled,
                           const gcn::TextureLayout32& linear,
                           const void* src,
                           void* dst,
                           bool detile,
                           rhi::Buffer* linear_buffer = nullptr,
                           bool pack11 = false) {
  const bool expanded = (tiled.elem_bytes == 1 || tiled.elem_bytes == 2) &&
                        linear.elem_bytes == 4;
  if (!g_backend.device || g_cs_failed ||
      (pack11 && (tiled.elem_bytes != 4 || linear.elem_bytes != 16)))
    return false;
  const u64 max_range = Device().caps().max_storage_buffer_range;
  if ((!expanded && !pack11 &&
       (tiled.elem_bytes < 4 || tiled.elem_bytes != linear.elem_bytes)) ||
      linear.tiling_idx != 8 || tiled.layers != linear.layers ||
      tiled.mip_levels != linear.mip_levels || tiled.size > max_range ||
      linear.size > max_range || tiled.size > UINT32_MAX ||
      linear.size > UINT32_MAX)
    return false;
  base::Vector<u32> table, terms;
  base::Vector<gcn::ImageTilingParams> params;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& t = tiled.mips[mip];
    const auto& l = linear.mips[mip];
    u32 mask;
    u64 stride;
    if (t.width != l.width || t.height != l.height ||
        !gcn::BuildSeparableAddressTable(tiled, mip, terms, mask, stride))
      return false;
    params.push_back({t.width, t.height, pack11 ? 1u : linear.elem_bytes / 4,
                      static_cast<u32>(table.size()), mask,
                      static_cast<u32>(t.offset), static_cast<u32>(stride),
                      static_cast<u32>(l.offset / 4), l.pitch,
                      l.pitch * l.stored_height * (linear.elem_bytes / 4),
                      detile ? 1u : 0u, tiled.elem_bytes, 0u, 0u,
                      pack11 ? 1u : 0u});
    table.insert(table.end(), terms.begin(), terms.end());
  }
  if (params.empty() || !InitTiling())
    return false;
  auto& s = g_tiling;
  const u64 sizes[] = {tiled.size, linear.size, table.size() * 4};
  rhi::BindingWrite writes[3];
  for (u32 i = 0; i < 3; i++) {
    const bool bound = i == 1 && linear_buffer;
    if (!bound &&
        (!CsRangeEnsureBuffer(s.buffers[i], sizes[i]) || !s.buffers[i].map))
      return false;
    writes[i].binding = i;
    writes[i].buffer = bound ? linear_buffer : s.buffers[i].buf;
    writes[i].range = sizes[i];
  }
  const u32 input = detile ? 0 : 1, output = detile ? 1 : 0;
  // Whichever half is bound to memory the shader reaches directly needs no
  // copy: the linear side is the caller's own buffer, the tiled side is guest
  // memory.
  const bool input_bound =
      input == 1 && linear_buffer;
  const bool output_bound =
      output == 1 && linear_buffer;
  const u64 t_hin = NowNs();
  if (!input_bound) {
    ParallelCopy(s.buffers[input].map, src, sizes[input]);
    g_tile_hin_bytes += sizes[input];
  }
  if (!output_bound && !s.buffers[output].device_local)
    std::memset(s.buffers[output].map, 0, sizes[output]);
  std::memcpy(s.buffers[2].map, table.data(), sizes[2]);
  g_tile_hin_ns += NowNs() - t_hin;
  rhi::CommandList* c = BeginImmediate();
  if (!c)
    return false;
  c->Barrier(kAccessAll, rhi::kAccessCopyWrite);
  if (!input_bound)
    RecordStagingCopy(c, s.buffers[input], sizes[input], true);
  RecordStagingCopy(c, s.buffers[2], sizes[2], true);
  if (output_bound || s.buffers[output].device_local)
    c->FillBuffer(writes[output].buffer, writes[output].offset, sizes[output],
                  0);
  c->Barrier(rhi::kAccessAllWrite, kAccessComputeRW);
  c->SetPipeline(s.pipe);
  c->PushBindGroup(0, s.set_layout, writes, 3);
  for (const auto& p : params) {
    c->SetPushConstants(0, sizeof(p), &p);
    c->Dispatch((p.width * p.words + 63) / 64, p.height, tiled.layers);
  }
  if (!output_bound) {
    RecordStagingCopy(c, s.buffers[output], sizes[output], false);
  }
  c->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
             rhi::kAccessHostRead | kAccessComputeRW);
  const u64 t_sync = NowNs();
  const bool ok = EndImmediate(c, "imm.image_tiling");
  g_tile_sync_ns += NowNs() - t_sync;
  if (!ok) {
    BASE_LOGI("gpuvk", "image conversion failed");
    g_cs_failed = true;
    return false;
  }
  const u64 t_hout = NowNs();
  if (detile && !linear_buffer) {
    ParallelCopy(dst, s.buffers[output].map, sizes[output]);
    g_tile_hout_bytes += sizes[output];
  } else if (!detile) {
    gcn::CopyImageContents(tiled, s.buffers[output].map, dst);
    g_tile_hout_bytes += tiled.size;
  }
  g_tile_hout_ns += NowNs() - t_hout;
  return true;
}

// ---- Tiled-truth ranges (DELTA_GPU_CS_TRUTH) ----

u64 TileTableKey(const gcn::TextureLayout32& t,
                 const gcn::TextureLayout32& l,
                 bool packed,
                 bool pack11) {
  u64 h = 1469598103934665603ull;
  const u64 words[] = {t.tiling_idx,    t.mips[0].width,
                       t.mips[0].height, t.mips[0].pitch,
                       t.layers,        t.mip_levels,
                       t.elem_bytes,    t.size,
                       l.elem_bytes,    l.size,
                       l.mips[0].pitch, (packed ? 1u : 0u) | (pack11 ? 2u : 0u)};
  for (u64 w : words)
    h = (h ^ w) * 1099511628211ull;
  return h;
}

// The address terms of one surface layout, built once. The shader reads
// three of them per texel, so they live where it can reach them quickly.
const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed,
                              bool pack11) {
  const bool expanded = !packed &&
                        (tiled.elem_bytes == 1 || tiled.elem_bytes == 2) &&
                        linear.elem_bytes == 4;
  // 4-byte texels staged as 16: 11/11/10 floats, unpacked and packed again.
  pack11 = pack11 || (!packed && tiled.elem_bytes == 4 && linear.elem_bytes == 16);
  if (pack11 && (packed || tiled.elem_bytes != 4 || linear.elem_bytes != 16))
    return nullptr;
  if (packed &&
      (tiled.elem_bytes >= 4 || tiled.elem_bytes != linear.elem_bytes))
    return nullptr;
  const u64 max_range = Device().caps().max_storage_buffer_range;
  if (!InitTiling() ||
      (!expanded && !packed && !pack11 &&
       (tiled.elem_bytes < 4 || tiled.elem_bytes != linear.elem_bytes)) ||
      linear.tiling_idx != 8 || tiled.layers != linear.layers ||
      tiled.mip_levels != linear.mip_levels || tiled.size > max_range ||
      linear.size > max_range || tiled.size > UINT32_MAX ||
      linear.size > UINT32_MAX)
    return nullptr;
  const u64 key = TileTableKey(tiled, linear, packed, pack11);
  auto it = g_tile_tables.find(key);
  if (it != g_tile_tables.end())
    return it->second.buf ? &it->second : nullptr;
  if (g_tile_tables.size() >= 1024)
    return nullptr;
  TileTable& t = g_tile_tables[key];  // a failed build stays empty: no retry
  base::Vector<u32> table, terms;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& tm = tiled.mips[mip];
    const auto& lm = linear.mips[mip];
    u32 mask;
    u64 stride;
    if (tm.width != lm.width || tm.height != lm.height ||
        !gcn::BuildSeparableAddressTable(tiled, mip, terms, mask, stride))
      return nullptr;
    const u32 words = packed || pack11 ? 1u : linear.elem_bytes / 4;
    t.params.push_back(
        {tm.width, tm.height, words, static_cast<u32>(table.size()), mask,
         static_cast<u32>(tm.offset), static_cast<u32>(stride),
         static_cast<u32>(packed ? lm.offset : lm.offset / 4), lm.pitch,
         packed ? lm.pitch * lm.stored_height * linear.elem_bytes
                : lm.pitch * lm.stored_height * (linear.elem_bytes / 4),
         1u, tiled.elem_bytes, packed ? 1u : 0u, 0u, pack11 ? 1u : 0u});
    table.insert(table.end(), terms.begin(), terms.end());
  }
  if (table.empty())
    return nullptr;
  t.layers = tiled.layers;
  t.expanded = expanded;
  t.bytes = table.size() * 4;
  // Written once by the host, read by every conversion: device-local
  // host-visible memory where the heap offers it, plain host memory otherwise.
  t.buf = CreateCsBuffer(t.bytes, rhi::kBufferStorage,
                         rhi::MemoryKind::kUploadDevice);
  if (t.buf && !t.buf->mapped()) {
    Device().Destroy(t.buf);
    t.buf = nullptr;
  }
  if (!t.buf) {
    t.buf = CreateCsBuffer(t.bytes, rhi::kBufferStorage,
                           rhi::MemoryKind::kUpload);
    if (t.buf && t.buf->mapped())
      BASE_LOGI("csgpu", "tiling table {} bytes uses host memory", t.bytes);
  }
  if (t.buf && !t.buf->mapped()) {
    Device().Destroy(t.buf);
    t.buf = nullptr;
  }
  if (!t.buf)
    return nullptr;
  std::memcpy(t.buf->mapped(), table.data(), t.bytes);
  return &t;
}

// Record one layout conversion between two device buffers: `tiled` holds the
// guest bytes (a range's truth), `linear` the shader's view of them. `mips`
// and `layers` limit the pass to the leading levels and slices (a bridge
// copies mip 0 of layer 0 only); 0 means all of them.
void RecordImageTiling(rhi::CommandList* c,
                       const TileTable& t,
                       rhi::Buffer* tiled,
                       u64 tiled_off,
                       u64 tiled_bytes,
                       rhi::Buffer* linear,
                       u64 linear_bytes,
                       bool detile,
                       u32 mips,
                       u32 layers,
                       bool depth16) {
  struct Span {
    rhi::CommandList* c;
    ~Span() { c->ProfileEnd(); }
  } span{c};
  c->ProfileBegin(4);
  auto& s = g_tiling;
  const u32 mip_count =
      mips ? base::Min<u32>(mips, t.params.size()) : t.params.size();
  const u32 layer_count = layers ? base::Min(layers, t.layers) : t.layers;
  c->Barrier(kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostWrite,
             kAccessComputeRW | rhi::kAccessCopyWrite);
  c->SetPipeline(s.pipe);
  rhi::BindingWrite writes[3];
  writes[0].binding = 0;
  writes[0].buffer = tiled;
  writes[0].offset = tiled_off;
  writes[0].range = tiled_bytes;
  writes[1].binding = 1;
  writes[1].buffer = linear;
  writes[1].range = linear_bytes;
  writes[2].binding = 2;
  writes[2].buffer = t.buf;
  writes[2].range = t.bytes;
  c->PushBindGroup(0, s.set_layout, writes, 3);
  for (u32 m = 0; m < mip_count; m++) {
    gcn::ImageTilingParams p = t.params[m];
    p.detile = detile ? 1u : 0u;
    p.depth16 = depth16 ? 1u : 0u;
    c->SetPushConstants(0, sizeof(p), &p);
    c->Dispatch((p.width * p.words + 63) / 64, p.height, layer_count);
  }
  c->Barrier(rhi::kAccessComputeWrite,
             kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostRead);
}

base::Vector<CsScratch> g_cs_scratch;
u64 g_truth_view_hits = 0;

rhi::Buffer* AcquireView(u64 bytes,
                         u64 base,
                         u64 off,
                         const TileTable* table,
                         u64 seq,
                         bool* fresh) {
  for (auto& s : g_cs_scratch) {
    if (s.view_base == base && s.view_off == off && s.view_table == table &&
        s.view_seq == seq && s.cap >= bytes) {
      s.owner = g_cs_scratch_owner;
      s.last_frame = g_frame.num;
      g_truth_view_hits++;
      *fresh = false;
      return s.buf;
    }
  }
  *fresh = true;
  return AcquireScratch(bytes);
}

void StampView(rhi::Buffer* view,
               u64 base,
               u64 off,
               const TileTable* table,
               u64 seq) {
  for (auto& s : g_cs_scratch)
    if (s.buf == view) {
      s.view_base = base;
      s.view_off = off;
      s.view_table = table;
      s.view_seq = seq;
      return;
    }
}

rhi::Buffer* AcquireScratch(u64 bytes) {
  int best = -1;
  for (size_t i = 0; i < g_cs_scratch.size(); i++) {
    const auto& s = g_cs_scratch[i];
    if (s.owner == g_cs_scratch_owner || s.cap < bytes)
      continue;
    if (best < 0 || s.cap < g_cs_scratch[best].cap ||
        (s.cap == g_cs_scratch[best].cap &&
         s.last_frame < g_cs_scratch[best].last_frame))
      best = static_cast<int>(i);
  }
  if (best >= 0) {
    auto& s = g_cs_scratch[best];
    s.owner = g_cs_scratch_owner;
    s.last_frame = g_frame.num;
    s.view_base = 0;
    s.view_table = nullptr;
    return s.buf;
  }
  // One dispatch's views are all that has to coexist; everything colder is
  // recycled before the ring is allowed to grow past its budget.
  constexpr u64 kScratchBudget = 512ull << 20;
  CsScratch s;
  s.cap = (bytes + 0x3FFFFF) & ~u64(0x3FFFFF);
  while (!g_cs_scratch.empty() &&
         (g_cs_scratch.size() >= 24 ||
          g_cs_scratch_bytes + s.cap > kScratchBudget)) {
    size_t coldest = SIZE_MAX;
    for (size_t i = 0; i < g_cs_scratch.size(); i++)
      if (g_cs_scratch[i].owner != g_cs_scratch_owner &&
          (coldest == SIZE_MAX ||
           g_cs_scratch[i].last_frame < g_cs_scratch[coldest].last_frame))
        coldest = i;
    if (coldest == SIZE_MAX)
      break;
    g_cs_frame_retired.push_back(g_cs_scratch[coldest].buf);
    g_cs_scratch_bytes -= g_cs_scratch[coldest].cap;
    g_cs_scratch.erase(g_cs_scratch.begin() + coldest);
  }
  s.buf = CreateCsBuffer(
      s.cap, rhi::kBufferStorage | rhi::kBufferCopySrc | rhi::kBufferCopyDst,
      SplitVram() ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback);
  if (!s.buf)
    return nullptr;
  s.owner = g_cs_scratch_owner;
  s.last_frame = g_frame.num;
  if (Device().caps().debug_labels) {
    char name[48];
    std::snprintf(name, sizeof(name), "cs scratch %llu KB",
                  (unsigned long long)(s.cap / 1024));
    Device().SetName(s.buf, name);
  }
  if (s.cap >= (64u << 20)) {
    static int n = 0;
    if (n++ < 32)
      BASE_LOGI("csgpu", "scratch {} MB for a {} MB view (ring {} x {} MB)",
                (unsigned long long)(s.cap >> 20),
                (unsigned long long)(bytes >> 20), g_cs_scratch.size(),
                (unsigned long long)(g_cs_scratch_bytes >> 20));
  }
  g_cs_scratch_bytes += s.cap;
  g_cs_scratch.push_back(s);
  return s.buf;
}

bool TruthAvailable() {
  return InitTiling();
}
#else
const TileTable* GetTileTable(const gcn::TextureLayout32&,
                              const gcn::TextureLayout32&,
                              bool,
                              bool) {
  return nullptr;
}
rhi::Buffer* AcquireView(u64, u64, u64, const TileTable*, u64, bool* fresh) {
  *fresh = true;
  return nullptr;
}
void StampView(rhi::Buffer*, u64, u64, const TileTable*, u64) {}
void RecordImageTiling(rhi::CommandList*,
                       const TileTable&,
                       rhi::Buffer*,
                       u64,
                       u64,
                       rhi::Buffer*,
                       u64,
                       bool,
                       u32,
                       u32,
                       bool) {}
rhi::Buffer* AcquireScratch(u64) {
  return nullptr;
}
bool TruthAvailable() {
  return false;
}
#endif

// The tiling shader handles every gfx10 swizzle, and a Liverpool layout once
// its address terms are shown to be separable (4-byte and wider texels).
bool WidensLinear(const ComputeInfo::Res& r) {
  return gcn::TilingIsLinear(r.tiling_idx) &&
         (r.elem_bytes == 1 || r.elem_bytes == 2) && r.stage_elem_bytes == 4;
}

bool GpuTileable(const ComputeInfo::Res& r) {
  if (r.tiling_idx >= gcn::kGfx10TilingBase)
    return true;
  if ((r.elem_bytes < 4 || r.elem_bytes != r.stage_elem_bytes) &&
      !WidensLinear(r))
    return false;
  static base::HashMap<u64, bool> known;
  static const guest::SessionReset reset_known(
      [] { guest::ResetResource(known); });
  u64 key = 1469598103934665603ull;
  for (u64 v : {u64(r.tiling_idx), u64(r.width), u64(r.height), u64(r.pitch),
                u64(r.layers), u64(r.mip_levels), u64(r.elem_bytes),
                u64(r.dfmt), u64(r.pow2_pad), r.guest_size, r.size})
    key = (key ^ v) * 1099511628211ull;
  auto [it, inserted] = known.try_emplace(key, false);
  if (!inserted)
    return it->second;
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(r, tiled, linear))
    return false;
  base::Vector<u32> terms;
  u32 mask;
  u64 stride;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++)
    if (!gcn::BuildSeparableAddressTable(tiled, mip, terms, mask, stride))
      return false;
  it->second = true;
  return true;
}

// A resource whose range can be a tiled truth: a surface the tiling shader
// handles, with the shader reading its texels directly or expanded.
bool TruthEligible(const ComputeInfo::Res& r) {
  return kCsTruth && kCsVram && r.image_staging && !r.zero_fill && r.size &&
         r.guest_size &&
         ((r.elem_bytes >= 4 && r.elem_bytes == r.stage_elem_bytes) ||
          ((r.elem_bytes == 1 || r.elem_bytes == 2) &&
           r.stage_elem_bytes == 4) ||
          (r.dfmt == 6 && r.elem_bytes == 4 && r.stage_elem_bytes == 16)) &&
         r.dfmt != 35 && r.dfmt != 40 && !gcn::TilingIsLinear(r.tiling_idx) &&
         r.guest_size <= Device().caps().max_storage_buffer_range &&
         r.size <= Device().caps().max_storage_buffer_range &&
         r.guest_size <= (768ull << 20) && r.size <= (768ull << 20) &&
         TruthAvailable() && GpuTileable(r);
}

bool CanGpuTile(const ComputeInfo::Res& res, const CsRange& e) {
  return kCsGpuTiling && !kDetileDump && !kCsWbAudit && e.device_local &&
         res.image_staging && res.size >= 256 * 1024 &&
         ((res.elem_bytes >= 4 && res.elem_bytes == res.stage_elem_bytes) ||
          ((res.elem_bytes == 1 || res.elem_bytes == 2) &&
           res.stage_elem_bytes == 4)) &&
         res.dfmt != 35 && res.dfmt != 40 &&
         // A linear surface only needs the GPU when its texels widen: the
         // video decoder's 8/16-bit planes, staged on the CPU per dispatch.
         (!gcn::TilingIsLinear(res.tiling_idx) || WidensLinear(res)) &&
         GpuTileable(res);
}

bool ConvertCsRangeImage(const ComputeInfo::Res& res, CsRange& e, bool detile) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  gcn::TextureLayout32 tiled, linear;
  if (!CanGpuTile(res, e) || !BuildCsImageLayouts(res, tiled, linear))
    return false;
  const u64 started = NowNs();
  if (!ConvertGfx10ImageImpl(tiled, linear, reinterpret_cast<void*>(res.base),
                             reinterpret_cast<void*>(res.base), detile, e.buf))
    return false;
  if (CsSyncReporting()) {
    g_gpu_tiling_ns[detile] += NowNs() - started;
    g_gpu_tiling_n[detile]++;
  }
  e.mirror_current = false;
  return true;
#else
  return false;
#endif
}

}  // namespace gpu::render::cs

namespace gpu::render {

bool ConvertGfx10Image(const gcn::TextureLayout32& tiled,
                       const gcn::TextureLayout32& linear,
                       const void* src,
                       void* dst,
                       bool detile,
                       bool pack11) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  const u64 started = NowNs();
  const bool ok = ConvertGfx10ImageImpl(tiled, linear, src, dst, detile,
                                        nullptr, pack11);
  if (ok && CsSyncReporting()) {
    g_gpu_tiling_ns[detile] += NowNs() - started;
    g_gpu_tiling_n[detile]++;
  }
  return ok;
#else
  return false;
#endif
}

}  // namespace gpu::render
