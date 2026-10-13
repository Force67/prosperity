/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

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

u64 TileTableKey(const gcn::TextureLayout32& t,
                 const gcn::TextureLayout32& l,
                 bool packed,
                 bool pack11) {
  u64 h = 1469598103934665603ull;
  const u64 words[] = {
      t.tiling_idx,     t.mips[0].width,
      t.mips[0].height, t.mips[0].pitch,
      t.layers,         t.mip_levels,
      t.elem_bytes,     t.size,
      l.elem_bytes,     l.size,
      l.mips[0].pitch,  (packed ? 1u : 0u) | (pack11 ? 2u : 0u)};
  for (u64 w : words)
    h = (h ^ w) * 1099511628211ull;
  return h;
}

const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed,
                              bool pack11) {
  const bool expanded = !packed &&
                        (tiled.elem_bytes == 1 || tiled.elem_bytes == 2) &&
                        linear.elem_bytes == 4;
  pack11 =
      pack11 || (!packed && tiled.elem_bytes == 4 && linear.elem_bytes == 16);
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
  TileTable& t = g_tile_tables[key];
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
  t.buf = CreateCsBuffer(t.bytes, rhi::kBufferStorage,
                         rhi::MemoryKind::kUploadDevice);
  if (t.buf && !t.buf->mapped()) {
    Device().Destroy(t.buf);
    t.buf = nullptr;
  }
  if (!t.buf) {
    t.buf =
        CreateCsBuffer(t.bytes, rhi::kBufferStorage, rhi::MemoryKind::kUpload);
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

bool WidensLinear(const ComputeInfo::Res& r) {
  return gcn::TilingIsLinear(r.tiling_idx) &&
         (r.elem_bytes == 1 || r.elem_bytes == 2) && r.stage_elem_bytes == 4;
}

bool GpuTileable(const ComputeInfo::Res& r) {
  if (r.tiling_idx >= gcn::kGfx10TilingBase)
    return true;
  const bool expanded =
      (r.elem_bytes == 1 || r.elem_bytes == 2) && r.stage_elem_bytes == 4;
  if (!expanded && (r.elem_bytes < 4 || r.elem_bytes != r.stage_elem_bytes) &&
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

#ifdef DELTA_HAVE_SPIRV_BACKEND
bool StageBcImageOnGpu(const ComputeInfo::Res& res,
                       CsRange& e,
                       const gcn::TextureLayout32& tiled,
                       const gcn::TextureLayout32& linear) {
  const bool bc6 = res.dfmt == 40;
  const rhi::Format block_format =
      bc6 ? (res.nfmt == 1 ? rhi::Format::kBC6HSfloat
                           : rhi::Format::kBC6HUfloat)
          : rhi::Format::kBC1Unorm;
  const rhi::Format plain_format =
      bc6 ? rhi::Format::kRGBA32Float : rhi::Format::kRGBA8Unorm;
  gcn::TextureLayout32 blocks;
  if (!Device().SupportsBlit(block_format, plain_format) ||
      !gcn::BuildTextureLayout32(blocks, tiled.mips[0].width,
                                 tiled.mips[0].height, tiled.mips[0].pitch,
                                 tiled.layers, tiled.mip_levels, 8, false,
                                 tiled.elem_bytes))
    return false;
  const TileTable* table = GetTileTable(tiled, blocks);
  if (!table)
    return false;
  struct Images {
    rhi::Texture* block = nullptr;
    rhi::Texture* plain = nullptr;
  };
  static base::HashMap<u64, Images> images;
  static const guest::SessionReset reset_images(
      [] { guest::ResetResource(images); });
  u64 key = 1469598103934665603ull;
  for (u64 v : {u64(block_format), u64(res.width), u64(res.height),
                u64(res.layers), u64(res.mip_levels)})
    key = (key ^ v) * 1099511628211ull;
  Images& im = images[key];
  if (!im.block || !im.plain) {
    rhi::TextureDesc desc;
    desc.width = res.width;
    desc.height = res.height;
    desc.layers = res.layers;
    desc.mips = res.mip_levels;
    desc.usage = rhi::kTextureCopySrc | rhi::kTextureCopyDst;
    desc.format = block_format;
    desc.name = "cs bc blocks";
    im.block = Device().CreateTexture(desc);
    desc.format = plain_format;
    desc.name = "cs bc decoded";
    im.plain = Device().CreateTexture(desc);
    if (!im.block || !im.plain)
      return false;
  }
  const u64 raw_bytes = (tiled.size + 3) & ~u64(3);
  const u64 block_bytes = (blocks.size + 3) & ~u64(3);
  rhi::Buffer* raw = AcquireScratch(raw_bytes);
  rhi::Buffer* linear_blocks = raw ? AcquireScratch(block_bytes) : nullptr;
  if (!linear_blocks || !RecordGuestRead(g_cs_list, res.base, tiled.size, raw))
    return false;
  rhi::CommandList* const c = g_cs_list;
  RecordImageTiling(c, *table, raw, 0, raw_bytes, linear_blocks, block_bytes,
                    true);
  rhi::BufferTextureCopy in[16]{}, out[16]{};
  rhi::TextureRegion regions[16];
  for (u32 mip = 0; mip < res.mip_levels; mip++) {
    const u32 w = base::Max(res.width >> mip, 1u);
    const u32 h = base::Max(res.height >> mip, 1u);
    rhi::TextureRegion& r = regions[mip];
    r.mip = mip;
    r.layers = res.layers;
    r.width = w;
    r.height = h;
    in[mip].buffer_offset = blocks.mips[mip].offset;
    in[mip].row_length = blocks.mips[mip].pitch * 4;
    in[mip].image_height = blocks.mips[mip].stored_height * 4;
    in[mip].region = r;
    out[mip].buffer_offset = linear.mips[mip].offset;
    out[mip].row_length = linear.mips[mip].pitch;
    out[mip].image_height = linear.mips[mip].stored_height;
    out[mip].region = r;
  }
  rhi::TextureRange all;
  all.mips = res.mip_levels;
  all.layers = res.layers;
  rhi::TextureBarrier b[2] = {{im.block, rhi::TextureState::kUndefined,
                               rhi::TextureState::kCopyDst, all},
                              {im.plain, rhi::TextureState::kUndefined,
                               rhi::TextureState::kCopyDst, all}};
  c->Barrier(kAccessComputeRW, rhi::kAccessCopyRead, b, 2);
  c->CopyBufferToTexture(im.block, linear_blocks, in, res.mip_levels);
  b[0].before = rhi::TextureState::kCopyDst;
  b[0].after = rhi::TextureState::kCopySrc;
  c->Barrier(0, 0, b, 1);
  for (u32 mip = 0; mip < res.mip_levels; mip++)
    c->BlitTexture(im.plain, regions[mip], im.block, regions[mip],
                   rhi::Filter::kNearest);
  b[1].before = rhi::TextureState::kCopyDst;
  b[1].after = rhi::TextureState::kCopySrc;
  c->Barrier(0, 0, &b[1], 1);
  c->FillBuffer(e.buf, 0, (res.size + 3) & ~u64(3), 0);
  c->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyWrite);
  c->CopyTextureToBuffer(e.buf, im.plain, out, res.mip_levels);
  return true;
}
#endif

bool StageImageOnGpu(const ComputeInfo::Res& res, CsRange& e) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  if (res.dfmt == 35 || res.dfmt == 40)
    return StageBcImageOnGpu(res, e, tiled, linear);
  const TileTable* table = GetTileTable(
      tiled, linear,
      tiled.elem_bytes < 4 && linear.elem_bytes == tiled.elem_bytes);
  rhi::Buffer* raw =
      table ? AcquireScratch((tiled.size + 3) & ~u64(3)) : nullptr;
  if (!raw || !RecordGuestRead(g_cs_list, res.base, tiled.size, raw))
    return false;
  RecordImageTiling(g_cs_list, *table, raw, 0, (tiled.size + 3) & ~u64(3),
                    e.buf, (linear.size + 3) & ~u64(3), true);
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
  if (!g_backend.device || g_cs_failed)
    return false;
  const TileTable* table = GetTileTable(tiled, linear, false, pack11);
  if (!table)
    return false;
  const u64 tiled_bytes = (tiled.size + 3) & ~u64(3);
  const u64 linear_bytes = (linear.size + 3) & ~u64(3);
  const u64 in_bytes = detile ? tiled.size : linear.size;
  const u64 out_bytes = detile ? linear_bytes : tiled_bytes;
  const auto make = [](u64 size, rhi::MemoryKind memory, u32 usage) {
    rhi::BufferDesc desc;
    desc.size = size;
    desc.usage = usage;
    desc.memory = memory;
    return Device().CreateBuffer(desc);
  };
  constexpr u32 kStorage =
      rhi::kBufferStorage | rhi::kBufferCopySrc | rhi::kBufferCopyDst;
  rhi::Buffer* tiled_buf =
      make(tiled_bytes, rhi::MemoryKind::kDevice, kStorage);
  rhi::Buffer* linear_buf =
      make(linear_bytes, rhi::MemoryKind::kDevice, kStorage);
  rhi::Buffer* upload =
      make(in_bytes, rhi::MemoryKind::kUpload, rhi::kBufferCopySrc);
  rhi::Buffer* readback =
      make(out_bytes, rhi::MemoryKind::kReadback, rhi::kBufferCopyDst);
  rhi::CommandList* c = tiled_buf && linear_buf && upload && readback &&
                                upload->mapped() && readback->mapped()
                            ? BeginImmediate()
                            : nullptr;
  bool ok = c != nullptr;
  if (ok) {
    std::memcpy(upload->mapped(), src, in_bytes);
    rhi::Buffer* in = detile ? tiled_buf : linear_buf;
    rhi::Buffer* out = detile ? linear_buf : tiled_buf;
    c->FillBuffer(out, 0, out_bytes, 0);
    c->CopyBuffer(in, 0, upload, 0, in_bytes);
    RecordImageTiling(c, *table, tiled_buf, 0, tiled_bytes, linear_buf,
                      linear_bytes, detile);
    c->CopyBuffer(readback, 0, out, 0, out_bytes);
    c->Barrier(rhi::kAccessCopyWrite, rhi::kAccessHostRead);
    ok = EndImmediate(c, "imm.image_tiling");
  }
  if (ok && detile)
    std::memcpy(dst, readback->mapped(), linear.size);
  else if (ok)
    gcn::CopyImageContents(tiled, readback->mapped(), dst);
  for (rhi::Buffer* b : {tiled_buf, linear_buf, upload, readback})
    Device().Destroy(b);
  return ok;
#else
  return false;
#endif
}

}  // namespace gpu::render
