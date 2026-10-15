/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute/internal.h"

#include <cstring>

#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_page_table.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/guest_format.h"
#include "gpu/render/hash.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/image/texture_cache.h"
#include "guest/session.h"

namespace gpu::render {

bool PreserveCsDepthBeforeClear(u64 base) {
  auto depth_it = g_depths.find(base);
  auto range_it = g_cs_ranges.find(base);
  if (!g_frame.recording || depth_it == g_depths.end() ||
      range_it == g_cs_ranges.end())
    return false;

  DepthTarget& depth = depth_it->second;
  CsRange& range = range_it->second;
  if (!depth.texture || depth.layout == rhi::TextureState::kUndefined ||
      !range.buf || range.pending_batch || range.gpu_dirty ||
      !range.image_staging)
    return false;

  const rhi::TextureState old_state = depth.layout;
  CsAliasedImage image{depth.texture,     depth.w,   depth.h, 4,
                       rhi::kAspectDepth, old_state, true};
  AliasedCopyPlan plan;
  if (!PlanAliasedCopy(image, range.res, "preserves", plan) || plan.unpack ||
      plan.depth16 || range.res.layers > 1 || depth.layers > 1 || range.truth)
    return false;

  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(range.res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  const u64 copy_bytes =
      static_cast<u64>(level.pitch) * plan.h * range.res.stage_elem_bytes;
  if (level.offset + copy_bytes > range.cap)
    return false;

  rhi::CommandList* const list = g_frame.list;
  AliasedImageBarrier(list, image, old_state, rhi::TextureState::kCopySrc);
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.region.aspect = rhi::kAspectDepth;
  copy.region.width = plan.w;
  copy.region.height = plan.h;
  list->CopyTextureToBuffer(range.buf, depth.texture, &copy, 1);
  list->Barrier(rhi::kAccessCopyWrite,
                rhi::kAccessComputeRead | rhi::kAccessHostRead);
  AliasedImageBarrier(list, image, rhi::TextureState::kCopySrc, old_state);

  range.rt_sourced = true;
  range.last_rt_frame = static_cast<int>(g_frame.num) + 1;
  range.rt_serial = g_render_serial + 1;
  if (kCsRtTrace) {
    static int logged = 0;
    if (logged++ < 16)
      BASE_LOGI("csrt", "f{} preserved depth {:#x} before clear", g_frame.num,
                (unsigned long)base);
  }
  return true;
}

bool CsSupplyTextureParts(u64 base,
                          const gcn::TextureLayout32& tiled,
                          u32 w,
                          u32 h,
                          rhi::Texture* img,
                          rhi::TextureState state,
                          u64* seq) {
  if (tiled.layer_stride || tiled.mip_levels < 2 ||
      (img &&
       (img->desc().dim != rhi::TextureDim::k2D ||
        !rhi::GetFormatInfo(img->desc().format).compressed ||
        rhi::GetFormatInfo(img->desc().format).bytes != tiled.elem_bytes)))
    return false;
  struct Memo {
    u64 base = 0, layout = 0, epoch = 0, revision = 0;
    int frame = -1;
  };
  static Memo memo[256];
  Memo& checked = memo[((base >> 9) ^ (base >> 18) ^ tiled.size) & 255];
  const u64 key =
      HashWord(HashWord(TileTableKey(tiled, tiled, false, false), w), h);
  const u64 epoch = TextureEpoch();
  if (seq && checked.base == base && checked.layout == key &&
      checked.epoch == epoch && checked.frame == g_frame.num &&
      checked.revision == *seq)
    return true;
  const auto parts = DirtyRangesOverlapping(base, tiled.size);
  if (parts.size() < 2)
    return false;
  u64 cursor = base;
  u64 revision = 14695981039346656037ull;
  CollectGuestWrites();
  for (u64 part : parts) {
    const auto& e = g_cs_ranges.find(part)->second;
    const u64 lo = base::Max(base, part);
    const u64 hi = base::Min(base + tiled.size, part + e.guest_bytes);
    if (lo != cursor || !e.buf || e.imported || (e.image_staging && !e.truth) ||
        !e.cpu_writes.empty() || !e.fill_kept.empty() || hi - part > e.size ||
        !e.staged_clean)
      return false;
    cursor = hi;
    revision = HashWord(HashWord(revision, part), e.write_seq);
  }
  if (cursor != base + tiled.size)
    return false;
  gcn::TextureLayout32 linear;
  if (!gcn::BuildTextureLayout32(linear, tiled.mips[0].width,
                                 tiled.mips[0].height, tiled.mips[0].pitch,
                                 tiled.layers, tiled.mip_levels, 8, false,
                                 tiled.elem_bytes))
    return false;
  rhi::BufferTextureCopy copies[16]{};
  for (u32 mip = 0; mip < linear.mip_levels; ++mip) {
    const auto& level = linear.mips[mip];
    const u32 mw = base::Max(w >> mip, 1u), mh = base::Max(h >> mip, 1u);
    if (level.width != (mw + 3) / 4 || level.height != (mh + 3) / 4)
      return false;
    auto& copy = copies[mip];
    copy.buffer_offset = level.offset;
    copy.row_length = level.pitch * 4;
    copy.image_height = level.stored_height * 4;
    copy.region.mip = mip;
    copy.region.layers = tiled.layers;
    copy.region.width = mw;
    copy.region.height = mh;
  }
  const TileTable* table = GetTileTable(tiled, linear);
  if (!table)
    return false;
  if (seq && *seq == revision)
    return true;
  if (!img)
    return true;
  g_cs_scratch_owner++;
  bool fresh = true;
  rhi::Buffer* pixels =
      AcquireView(linear.size, base, 0, table, revision, &fresh);
  rhi::Buffer* assembled = fresh ? AcquireScratch(tiled.size) : nullptr;
  if (!pixels || (fresh && !assembled))
    return false;
  EndRegion();
  auto* list = g_frame.list;
  list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
  if (fresh) {
    for (u64 part : parts) {
      const auto& e = g_cs_ranges.find(part)->second;
      const u64 lo = base::Max(base, part);
      const u64 hi = base::Min(base + tiled.size, part + e.guest_bytes);
      list->CopyBuffer(assembled, lo - base, e.buf, lo - part, hi - lo);
    }
    list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessComputeRead);
    RecordImageTiling(list, *table, assembled, 0, tiled.size, pixels,
                      linear.size, true);
    StampView(pixels, base, 0, table, revision);
  }
  rhi::TextureBarrier barrier{img, state, rhi::TextureState::kCopyDst};
  barrier.range.mips = tiled.mip_levels;
  barrier.range.layers = tiled.layers;
  list->Barrier(0, 0, &barrier, 1);
  list->CopyBufferToTexture(img, pixels, copies, linear.mip_levels);
  barrier.before = rhi::TextureState::kCopyDst;
  barrier.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &barrier, 1);
  for (u64 part : parts) {
    auto& e = g_cs_ranges.find(part)->second;
    e.last_used_frame = e.frame_ref = g_frame.num;
    e.chunk_ref = g_frame.chunk_seq;
    if (e.pending_batch)
      g_cs_chunk_needs_batch = true;
  }
  if (kCsTexBridgeTrace)
    BASE_LOGI("texbridge", "{:#x}: composed {} ranges, {} mips, {} bytes", base,
              parts.size(), tiled.mip_levels, tiled.size);
  checked = {base, key, epoch, revision, g_frame.num};
  *seq = revision;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += tiled.size;
  return true;
}

bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     rhi::Texture* img,
                     rhi::TextureState state,
                     u64* seq,
                     bool compressed,
                     u32* supplied_mips) {
  if (!kCsTexBridge || !kCsVram || g_cs_failed)
    return false;
  if (compressed && CsSupplyTextureParts(base, layout, w, h, img, state, seq)) {
    if (supplied_mips)
      *supplied_mips = layout.mip_levels;
    return true;
  }
  auto it = g_cs_ranges.find(base);
  if (it == g_cs_ranges.end())
    return false;
  CsRange& e = it->second;
  const ComputeInfo::Res& r = e.res;
  if (!e.buf || !e.gpu_dirty || !e.image_staging || e.imported)
    return false;
  auto declined = [&](const char* why) {
    if (kCsTexBridgeTrace) {
      static base::HashSet<u64> seen;
      static const guest::SessionReset reset_seen(
          [] { guest::ResetResource(seen); });
      if (seen.size() < 256 && seen.insert(base).second)
        BASE_LOGI("texbridge",
                  "{:#x} {}: cs {}x{} l={} m={} e={}/{} d={} t={} | tex "
                  "{}x{} l={} m={} e={} t={}",
                  (unsigned long)base, why, r.width, r.height, r.layers,
                  r.mip_levels, r.elem_bytes, r.stage_elem_bytes, r.dfmt,
                  r.tiling_idx, w, h, layout.layers, layout.mip_levels,
                  layout.elem_bytes, layout.tiling_idx);
    }
    return false;
  };
  const u32 width = compressed ? (w + 3) / 4 : w;
  const u32 height = compressed ? (h + 3) / 4 : h;
  if (r.width != width || r.height != height)
    return declined("extent");
  if (compressed &&
      ((r.nfmt != 4 && r.nfmt != 5) ||
       (r.dfmt != 11 && r.dfmt != 12 && r.dfmt != 14) ||
       (img && (!rhi::GetFormatInfo(img->desc().format).compressed ||
                rhi::GetFormatInfo(img->desc().format).bytes != r.elem_bytes))))
    return declined("block format");
  if (r.layers < layout.layers || r.mip_levels > layout.mip_levels ||
      (!supplied_mips && r.mip_levels != layout.mip_levels))
    return declined("layers/mips");
  const bool narrow = e.truth && r.elem_bytes < 4 && r.stage_elem_bytes == 4;
  const bool unpacked = r.dfmt == 6 && !e.truth && r.stage_elem_bytes == 16 &&
                        r.mip_levels == 1 && r.layers == 1;
  if ((r.dfmt == 6 && !unpacked) || r.dfmt == 35 || r.dfmt == 40 ||
      (!narrow && !unpacked && r.elem_bytes != r.stage_elem_bytes))
    return declined("converted staging");
  if (r.elem_bytes != layout.elem_bytes)
    return declined("texel size");
  if (unpacked &&
      ((img && !BlitsBetween(rhi::Format::kRGBA32Float, img->desc().format)) ||
       !GetBridgeFloatImage(w, h)))
    return declined("no blit");
  if (LiveTargetOfSize(base, w, h))
    return declined("live target");
  for (u64 other : DirtyRangesOverlapping(base, layout.size, base)) {
    const CsRange& o = g_cs_ranges.find(other)->second;
    if (o.dirty_stamp > e.dirty_stamp || o.guest_bytes != e.guest_bytes ||
        o.res.width != r.width || o.res.height != r.height ||
        o.res.elem_bytes != r.elem_bytes || o.res.tiling_idx != r.tiling_idx)
      return declined("dirty sibling");
  }
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(r, tiled, linear) || linear.layer_stride)
    return declined("layout");
  if (tiled.tiling_idx != layout.tiling_idx ||
      tiled.layer_stride != layout.layer_stride)
    return declined("image layout");
  for (u32 mip = 0; mip < tiled.mip_levels; ++mip)
    if (tiled.mips[mip].offset != layout.mips[mip].offset ||
        tiled.mips[mip].pitch != layout.mips[mip].pitch ||
        tiled.mips[mip].stored_height != layout.mips[mip].stored_height)
      return declined("mip layout");
  if (narrow &&
      !gcn::BuildTextureLayout32(linear, r.width, r.height, r.pitch, r.layers,
                                 r.mip_levels, 8, r.pow2_pad, r.elem_bytes))
    return declined("packed layout");
  const TileTable* table =
      e.truth ? GetTileTable(tiled, linear, narrow) : nullptr;
  if (e.truth && !table)
    return declined("tile table");
  rhi::BufferTextureCopy copies[16];
  for (u32 mip = 0; mip < linear.mip_levels; mip++) {
    const auto& level = linear.mips[mip];
    const u32 mip_w = base::Max(w >> mip, 1u);
    const u32 mip_h = base::Max(h >> mip, 1u);
    if (level.offset % base::Max<u64>(4, r.elem_bytes) ||
        level.width != (compressed ? (mip_w + 3) / 4 : mip_w) ||
        level.height != (compressed ? (mip_h + 3) / 4 : mip_h))
      return declined("mip geometry");
    copies[mip].buffer_offset = level.offset;
    copies[mip].row_length = level.pitch * (compressed ? 4 : 1);
    copies[mip].image_height = level.stored_height * (compressed ? 4 : 1);
    copies[mip].region.mip = mip;
    copies[mip].region.layers = layout.layers;
    copies[mip].region.width = mip_w;
    copies[mip].region.height = mip_h;
  }
  if (supplied_mips)
    *supplied_mips = linear.mip_levels;
  if (!img)
    return true;
  const bool volume = img->desc().dim == rhi::TextureDim::k3D;
  if (volume) {
    if (linear.mip_levels != 1)
      return false;
    copies[0].region.depth = copies[0].region.layers;
    copies[0].region.layers = 1;
  }
  e.last_used_frame = g_frame.num;
  if (*seq == e.write_seq)
    return true;
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  rhi::Buffer* src = e.buf;
  if (e.truth) {
    g_cs_scratch_owner++;
    bool fresh = true;
    src = AcquireView(linear.size, base, 0, table, e.write_seq, &fresh);
    if (!src)
      return false;
    if (fresh) {
      RecordImageTiling(list, *table, e.buf, 0, e.guest_bytes, src, linear.size,
                        true);
      StampView(src, base, 0, table, e.write_seq);
    }
  }
  rhi::TextureBarrier b;
  b.texture = img;
  b.before = state;
  b.after = rhi::TextureState::kCopyDst;
  b.range.mips = linear.mip_levels;
  b.range.layers = volume ? 1 : layout.layers;
  list->Barrier(0, 0, &b, 1);
  if (unpacked) {
    rhi::Texture* float_image = GetBridgeFloatImage(w, h);
    rhi::TextureBarrier fb;
    fb.texture = float_image;
    fb.before = rhi::TextureState::kUndefined;
    fb.after = rhi::TextureState::kCopyDst;
    list->Barrier(0, 0, &fb, 1);
    list->CopyBufferToTexture(float_image, src, copies, 1);
    fb.before = rhi::TextureState::kCopyDst;
    fb.after = rhi::TextureState::kCopySrc;
    list->Barrier(0, 0, &fb, 1);
    rhi::TextureRegion region;
    region.width = w;
    region.height = h;
    list->BlitTexture(img, region, float_image, region, rhi::Filter::kNearest);
  } else {
    list->CopyBufferToTexture(img, src, copies, linear.mip_levels);
  }
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  *seq = e.write_seq;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += e.size;
  return true;
}

DELTA_OPTION(bool, kCsBufBridge, "DELTA_GPU_CS_BUF_BRIDGE", true);
DELTA_OPTION(bool, kCsVtxBridge, "DELTA_GPU_CS_VTX_BRIDGE", true);
DELTA_OPTION(bool, kBufMemo, "DELTA_GPU_BUFMEMO", true);

}  // namespace gpu::render

namespace gpu::render::cs {

CsRange* BufferSourceUncached(u64 base, u64 bytes, u64* range_base);

CsRange* BufferSource(u64 base, u64 bytes, u64* range_base) {
  if (!kCsBufBridge || !kCsVram || g_cs_failed || !bytes || !g_frame.recording)
    return nullptr;
  struct Memo {
    u64 base = 0, bytes = 0, epoch = 0, range_base = 0;
    CsRange* range = nullptr;
  };
  static Memo memo[512];
  Memo& m = memo[((base >> 4) ^ (base >> 16) ^ bytes) & 511];
  const u64 epoch = TextureEpoch();
  if (!kBufMemo || m.epoch != epoch || m.base != base || m.bytes != bytes) {
    u64 found_base = 0;
    CsRange* found = BufferSourceUncached(base, bytes, &found_base);
    m = {base, bytes, epoch, found_base, found};
  }
  *range_base = m.range_base;
  return m.range;
}

CsRange* BufferSourceUncached(u64 base, u64 bytes, u64* range_base) {
  u64 only = 0;
  const bool several = AnyDirtyOverlapping(base, bytes, [&](u64 other) {
    if (only && other != only)
      return true;
    only = other;
    return false;
  });
  if (kCsOverlapTrace && g_frame.num >= kCsOverlapTrace && only) {
    static int shown = 0;
    if (shown++ < 30) {
      base::String line;
      for (u64 r : DirtyRangesOverlapping(base, bytes, base)) {
        const CsRange& o = g_cs_ranges.find(r)->second;
        base::FormatTo(line, " [{:#x}+{:#x} size={:#x} truth={} img={}]", r,
                       o.guest_bytes, o.size, (int)o.truth,
                       (int)o.image_staging);
      }
      BASE_LOGI("csvtx", "window {:#x}+{:#x}:{}", base, bytes, line.c_str());
    }
  }
  if (several || !only)
    return nullptr;
  CsRange& e = g_cs_ranges.find(only)->second;
  const u64 valid = base::Min<u64>(e.size, e.guest_bytes);
  if (!e.buf || e.image_staging || e.truth || e.imported || base < only ||
      base + bytes > only + valid)
    return nullptr;
  *range_base = only;
  return &e;
}

}  // namespace gpu::render::cs

namespace gpu::render {

u64 PendingGuestWriteBatch(u64 base, u64 bytes) {
  if (GuestPages().PendingWriteClear(base, base + bytes))
    return 0;
  u64 batch = 0;
  for (const PendingGuestWrite& w : g_pending_guest_writes)
    if (w.base < base + bytes && base < w.end)
      batch = base::Max(batch, w.batch);
  return batch;
}

u64 CsBufferRevision(u64 base, u64 bytes) {
  u64 range_base;
  const CsRange* e = BufferSource(base, bytes, &range_base);
  if (e)
    return e->write_seq;
  const u64 mirror = MirrorRevision(base, bytes);
  if (mirror)
    return mirror | (1ull << 63);
  const u64 pending =
      g_frame.recording ? PendingGuestWriteBatch(base, bytes) : 0;
  return pending ? pending | (1ull << 62) : 0;
}

bool CsSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off) {
  u64 range_base;
  CsRange* const found = BufferSource(base, bytes, &range_base);
  if (!found && dst && MirrorRevision(base, bytes))
    return MirrorSupplyBuffer(base, bytes, dst, dst_off);
  if (!found && dst && g_frame.recording &&
      PendingGuestWriteBatch(base, bytes)) {
    EndRegion();
    rhi::CommandList* const list = g_frame.list;
    if (!RecordGuestRead(list, base, bytes, dst, dst_off))
      return false;
    list->Barrier(rhi::kAccessCopyWrite, kAccessAll);
    g_buf_bridge_n++;
    g_buf_bridge_bytes += bytes;
    return true;
  }
  if (!found || !dst)
    return false;
  CsRange& e = *found;
  const u64 off = base - range_base;
  CollectGuestWrites();
  if (!e.staged_clean) {
    g_buf_bridge_declined++;
    return false;
  }
  e.last_used_frame = g_frame.num;
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  list->CopyBuffer(dst, dst_off, e.buf, off, bytes);
  list->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_buf_bridge_n++;
  g_buf_bridge_bytes += bytes;
  return true;
}

bool CsSupplyTextureFromBuffer(u64 base,
                               const gcn::TextureLayout32& tiled,
                               const gcn::TextureLayout32& linear,
                               u32 w,
                               u32 h,
                               rhi::Texture* img,
                               rhi::TextureState state,
                               u64* seq) {
  u64 range_base;
  CsRange* const found = BufferSource(base, tiled.size, &range_base);
  if (linear.mip_levels > 16 || linear.mips[0].width != w ||
      linear.mips[0].height != h ||
      (img && img->desc().dim == rhi::TextureDim::k3D))
    return false;
  if (!found)
    return !CsRangeDirtyOverlapping(base, tiled.size) &&
           MirrorSupplyTexture(base, tiled, linear, img, state, seq);
  CsRange& e = *found;
  const TileTable* table = GetTileTable(tiled, linear, tiled.elem_bytes < 4);
  if (!table)
    return false;
  if (!img)
    return true;
  e.last_used_frame = g_frame.num;
  if (*seq == e.write_seq)
    return true;
  const u64 off = base - range_base;
  CollectGuestWrites();
  if (!e.staged_clean) {
    g_buf_bridge_declined++;
    return false;
  }
  rhi::BufferTextureCopy copies[16];
  for (u32 mip = 0; mip < linear.mip_levels; mip++) {
    const auto& level = linear.mips[mip];
    copies[mip].buffer_offset = level.offset;
    copies[mip].row_length = level.pitch;
    copies[mip].image_height = level.stored_height;
    copies[mip].region.mip = mip;
    copies[mip].region.layers = linear.layers;
    copies[mip].region.width = level.width;
    copies[mip].region.height = level.height;
  }
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  g_cs_scratch_owner++;
  bool fresh = true;
  rhi::Buffer* src =
      AcquireView(linear.size, range_base, off, table, e.write_seq, &fresh);
  if (!src)
    return false;
  if (fresh) {
    RecordImageTiling(list, *table, e.buf, off, tiled.size, src, linear.size,
                      true);
    StampView(src, range_base, off, table, e.write_seq);
  }
  rhi::TextureBarrier b;
  b.texture = img;
  b.before = state;
  b.after = rhi::TextureState::kCopyDst;
  b.range.mips = linear.mip_levels;
  b.range.layers = linear.layers;
  list->Barrier(0, 0, &b, 1);
  list->CopyBufferToTexture(img, src, copies, linear.mip_levels);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  *seq = e.write_seq;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += tiled.size;
  return true;
}

CsRange* VertexSource(u64 base, u64 bytes, u64* range_base) {
  if (!kCsBufBridge || !kCsVram || g_cs_failed || !bytes || !g_frame.recording)
    return nullptr;
  constexpr u64 kSlack = 64;
  CsRange* found = nullptr;
  bool bad = false;
  AnyDirtyOverlapping(base, bytes, [&](u64 other) {
    CsRange& e = g_cs_ranges.find(other)->second;
    const u64 valid = base::Min<u64>(e.size, e.guest_bytes);
    if (other <= base && base + bytes <= other + valid) {
      if ((found && found != &e) || !e.buf || e.image_staging || e.truth ||
          e.imported)
        bad = true;
      found = &e;
      *range_base = other;
    } else {
      const u64 lo = base::Max(base, other);
      const u64 hi = base::Min(base + bytes, other + e.guest_bytes);
      bad |= hi > lo && hi - lo > kSlack;
    }
    return bad;
  });
  return bad ? nullptr : found;
}

bool MirrorHoldsNewest(u64 base, u64 bytes) {
  if (!bytes || !MirrorWritten(base, bytes))
    return false;
  bool owned = true;
  for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
       block += kMirrorBlock) {
    if (g_own && OwnAt(block).cpu_dirty.load(base::memory_order_acquire))
      return false;
    owned &= MirrorOwned(block);
  }
  if (owned)
    return true;
  if (GuestPages().PendingWriteClear(base, base + bytes))
    return false;
  for (const PendingGuestWrite& w : g_pending_guest_writes)
    if (w.base < base + bytes && base < w.end)
      return true;
  return false;
}

bool CsHoldsGeometry(u64 base, u64 bytes) {
  u64 range_base;
  return kCsVtxBridge && bytes && g_frame.recording &&
         (VertexSource(base, bytes, &range_base) ||
          MirrorHoldsNewest(base, bytes) || MirrorOwnsAny(base, base + bytes) ||
          PendingGuestWriteBatch(base, bytes));
}

namespace {
void MakeComputeBufferReadable(u64 batch, u64 mirror_seq = 0) {
  static u64 covered_chunk = ~0ull, covered_batch = 0, covered_mirror = 0;
  static const guest::SessionReset reset_coverage([] {
    covered_chunk = ~0ull;
    covered_batch = covered_mirror = 0;
  });
  if (covered_chunk != g_frame.chunk_seq) {
    covered_chunk = g_frame.chunk_seq;
    covered_batch = covered_mirror = 0;
  }
  if (batch > covered_batch || mirror_seq > covered_mirror) {
    EndRegion();
    g_frame.list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                          kAccessAll);
    covered_batch = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id - 1;
    covered_mirror = g_mirror_seq;
  }
}

bool BindComputeBuffer(u64 base,
                       u64 bytes,
                       CsRange* found,
                       u64 range_base,
                       rhi::Buffer** buffer,
                       u64* offset) {
  if (!found && MirrorHoldsNewest(base, bytes)) {
    MakeComputeBufferReadable(PendingGuestWriteBatch(base, bytes),
                              MirrorWritten(base, bytes)->seq);
    if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame)
      g_cs_chunk_needs_batch = true;
    *buffer = g_mirror;
    *offset = MirrorOffset(base);
    MirrorFrameRead();
    return true;
  }
  if (!found)
    return false;
  CsRange& e = *found;
  const u64 off = base - range_base;
  CollectGuestWrites();
  if (!e.staged_clean) {
    g_buf_bridge_declined++;
    return false;
  }
  MakeComputeBufferReadable(e.batch_id);
  e.last_used_frame = g_frame.num;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  *buffer = e.buf;
  *offset = off;
  return true;
}

bool GatherDrawBuffer(u64 base,
                      u64 bytes,
                      rhi::Buffer** buffer,
                      u64* offset,
                      CsRange* source = nullptr,
                      u64 padded_bytes = 0) {
  padded_bytes = base::Max(bytes, padded_bytes);
  if (padded_bytes > bytes && ((bytes | padded_bytes) & 3))
    return false;
  if (!bytes || !g_frame.recording ||
      (!source && !MirrorOwnsAny(base, base + bytes) &&
       !PendingGuestWriteBatch(base, bytes)))
    return false;
  CollectGuestWrites();
  if (source && !source->staged_clean) {
    source = nullptr;
    if (!MirrorOwnsAny(base, base + bytes) &&
        !PendingGuestWriteBatch(base, bytes))
      return false;
  }
  struct Gathered {
    rhi::Buffer* buffer = nullptr;
    rhi::Buffer* source = nullptr;
    u64 bytes = 0, padded_bytes = 0, revision = 0, stamp = 0;
    int last_frame = 0;
  };
  static base::Vector<Gathered> slots[2];
  static base::HashMap<u64, size_t> latest;
  static int frame = -1;
  static size_t used = 0;
  static const guest::SessionReset reset_gathered([] {
    guest::ResetResource(slots);
    guest::ResetResource(latest);
    frame = -1;
    used = 0;
  });
  auto& pool = slots[g_frame.slot_idx];
  if (frame != g_frame.num) {
    frame = g_frame.num;
    latest.clear();
    used = 0;
    for (size_t i = 0; i < pool.size();) {
      if (g_frame.num - pool[i].last_frame < 600) {
        ++i;
        continue;
      }
      Device().Destroy(pool[i].buffer);
      pool.erase(pool.begin() + i);
    }
  }
  const u64 revision = source ? source->write_seq : MirrorRevision(base, bytes);
  const u64 stamp = source ? source->staged_stamp : 0;
  rhi::Buffer* source_buffer = source ? source->buf : nullptr;
  if (revision) {
    const auto found = latest.find(base);
    if (found != latest.end()) {
      const auto& cached = pool[found->second];
      if (cached.bytes == bytes && cached.padded_bytes == padded_bytes &&
          cached.revision == revision && cached.source == source_buffer &&
          cached.stamp == stamp) {
        *buffer = cached.buffer;
        *offset = 0;
        return true;
      }
    }
  }
  if (used == pool.size())
    pool.emplace_back();
  auto& gathered = pool[used++];
  gathered.last_frame = g_frame.num;
  if (!gathered.buffer || gathered.buffer->desc().size < padded_bytes) {
    rhi::BufferDesc desc;
    desc.size = (padded_bytes + 65535) & ~u64(65535);
    desc.usage = rhi::kBufferVertex | rhi::kBufferIndex | rhi::kBufferUniform |
                 rhi::kBufferStorage | rhi::kBufferCopyDst |
                 rhi::kBufferCopySrc;
    desc.name = "gathered draw buffer";
    auto* replacement = Device().CreateBuffer(desc);
    if (!replacement)
      return false;
    if (gathered.buffer)
      Device().Destroy(gathered.buffer);
    gathered.buffer = replacement;
  }
  if (padded_bytes > bytes) {
    EndRegion();
    g_frame.list->FillBuffer(gathered.buffer, bytes, padded_bytes - bytes, 0);
  }
  if (!(source ? CsSupplyBuffer(base, bytes, gathered.buffer, 0)
               : MirrorSupplyBuffer(base, bytes, gathered.buffer, 0)))
    return false;
  gathered.bytes = bytes;
  gathered.padded_bytes = padded_bytes;
  gathered.revision = revision;
  gathered.source = source_buffer;
  gathered.stamp = stamp;
  latest[base] = used - 1;
  *buffer = gathered.buffer;
  *offset = 0;
  return true;
}
}  // namespace

void PrepareDirectGuestRead() {
  MakeComputeBufferReadable(
      g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id - 1, g_mirror_seq);
  if (CsBatchOpenOutOfFrame())
    g_cs_chunk_needs_batch = true;
  MirrorFrameRead();
}

bool CsResidentGeometryBuffer(u64 base,
                              u64 bytes,
                              rhi::Buffer** buffer,
                              u64* offset) {
  if (!kCsVtxBridge)
    return false;
  u64 range_base = 0;
  CsRange* found = VertexSource(base, bytes, &range_base);
  const bool bound =
      BindComputeBuffer(base, bytes, found, range_base, buffer, offset);
  g_vtx_bridge_n += bound;
  return bound;
}

bool CsGeometryBuffer(u64 base, u64 bytes, rhi::Buffer** buffer, u64* offset) {
  if (CsResidentGeometryBuffer(base, bytes, buffer, offset))
    return true;
  const bool bound =
      kCsVtxBridge && GatherDrawBuffer(base, bytes, buffer, offset);
  g_vtx_bridge_n += bound;
  return bound;
}

bool CsStorageBuffer(u64 base,
                     u64 bytes,
                     rhi::Buffer** buffer,
                     u64* offset,
                     u64 alignment,
                     u64 padded_bytes) {
  if (!kCsBufBridge)
    return false;
  u64 range_base = 0;
  CsRange* found = BufferSource(base, bytes, &range_base);
  rhi::Buffer* resident = nullptr;
  u64 resident_offset = 0;
  if (bytes >= padded_bytes &&
      BindComputeBuffer(base, bytes, found, range_base, &resident,
                        &resident_offset) &&
      !(resident_offset % base::Max<u64>(alignment, 1))) {
    *buffer = resident;
    *offset = resident_offset;
    return true;
  }
  return GatherDrawBuffer(base, bytes, buffer, offset, found, padded_bytes);
}

bool RtRefreshMiss(const char* why) {
  if (CsSyncReporting()) {
    static base::Map<const char*, u64> counts;
    static const guest::SessionReset reset_counts(
        [] { guest::ResetResource(counts); });
    static u64 n = 0;
    counts[why]++;
    if (++n % 200 == 0) {
      base::String line;
      for (const auto& [k, v] : counts)
        base::FormatTo(line, " {}={}", k, v);
      BASE_LOGI("rtrefresh", "declined:{}", line.c_str());
    }
  }
  return false;
}

bool CsRefreshRtInFrame(u64 base, CsRange& e) {
  if ((!e.truth && !e.image_staging) || !e.gpu_dirty || !e.buf ||
      e.rt_seq == e.write_seq || !g_frame.recording)
    return RtRefreshMiss("state");
  const ComputeInfo::Res& r = e.res;
  const bool unpacked = r.dfmt == 6 && r.stage_elem_bytes == 16;
  const bool widened =
      (r.elem_bytes == 1 || r.elem_bytes == 2) && r.stage_elem_bytes == 4;
  if (r.mip_levels != 1 || r.layers != 1 ||
      (!unpacked && !widened && r.elem_bytes != r.stage_elem_bytes))
    return RtRefreshMiss(r.mip_levels != 1 ? "mips"
                         : r.layers != 1   ? "layers"
                                           : "elem");
  ActivateWrittenRtVariant(base, r.width, r.height);
  auto rt_it = g_rts.find(base);
  if (rt_it == g_rts.end() || !rt_it->second.texture || rt_it->second.is_depth)
    return RtRefreshMiss("no-rt");
  if (!unpacked && (rt_it->second.w != r.width || rt_it->second.h != r.height ||
                    FormatBytes(rt_it->second.fmt) != r.elem_bytes)) {
    const rhi::Format fmt = GuestTextureFormat(r.dfmt, r.nfmt);
    if (FormatBytes(fmt) == r.elem_bytes &&
        ActivateRtVariantAs(base, r.width, r.height, fmt)) {
      rt_it = g_rts.find(base);
      g_cs_rt_variants_made++;
    }
  }
  RTarget& rt = rt_it->second;
  if (rt.w + 8 < r.width || rt.h + 8 < r.height ||
      FormatBytes(rt.fmt) != r.elem_bytes) {
    static int shown = 0;
    if (CsSyncReporting() && shown++ < 12)
      BASE_LOGI("rtrefresh",
                "{:#x}: cs {}x{} e={}/{} dfmt={} t={} vs rt {}x{} fmt={}", base,
                r.width, r.height, r.elem_bytes, r.stage_elem_bytes, r.dfmt,
                r.tiling_idx, rt.w, rt.h, (int)rt.fmt);
    return RtRefreshMiss(FormatBytes(rt.fmt) != r.elem_bytes ? "rt-format"
                                                             : "rt-size");
  }
  gcn::TextureLayout32 tiled, linear, packed;
  if (!BuildCsImageLayouts(r, tiled, linear) ||
      (widened &&
       !gcn::BuildTextureLayout32(packed, r.width, r.height, r.pitch, 1, 1, 8,
                                  r.pow2_pad, r.elem_bytes)))
    return RtRefreshMiss("layout");
  const u32 w = base::Min(rt.w, r.width), h = base::Min(rt.h, r.height);
  rhi::Texture* float_image = nullptr;
  if (unpacked && (!BlitsBetween(rhi::Format::kRGBA32Float, rt.fmt) ||
                   !(float_image = GetBridgeFloatImage(w, h))))
    return RtRefreshMiss("float-blit");
  rhi::Buffer* src = e.buf;
  bool fresh = false;
  const TileTable* table = nullptr;
  const TileTable* narrow = nullptr;
  static CsRange guest_scratch, packed_scratch;
  if (widened) {
    if (!(narrow = GetTileTable(tiled, packed, true)) ||
        (!e.truth && !(table = GetTileTable(tiled, linear))))
      return RtRefreshMiss("table");
    if ((!e.truth && !CsRangeEnsureBuffer(guest_scratch, tiled.size)) ||
        !CsRangeEnsureBuffer(packed_scratch, packed.size))
      return false;
    guest_scratch.frame_ref = packed_scratch.frame_ref = g_frame.num;
    src = packed_scratch.buf;
  } else if (e.truth) {
    if (!(table = GetTileTable(tiled, linear)))
      return RtRefreshMiss("table");
    g_cs_scratch_owner++;
    fresh = true;
    if (!(src = AcquireView(linear.size, base, 0, table, e.write_seq, &fresh)))
      return false;
  }
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  if (fresh)
    RecordImageTiling(list, *table, e.buf, 0, e.guest_bytes, src, linear.size,
                      true, 1, 1);
  if (widened) {
    rhi::Buffer* guest = e.buf;
    u64 guest_bytes = e.guest_bytes;
    if (!e.truth) {
      guest = guest_scratch.buf;
      guest_bytes = (tiled.size + 3) & ~u64(3);
      RecordImageTiling(list, *table, guest, 0, guest_bytes, e.buf,
                        (linear.size + 3) & ~u64(3), false, 1, 1);
    }
    RecordImageTiling(list, *narrow, guest, 0, guest_bytes, src,
                      (packed.size + 3) & ~u64(3), true, 1, 1);
  }
  const rhi::TextureState from = rt.layout;
  TransitionImage(list, rt.texture, rt.layout, rhi::TextureState::kCopyDst);
  const auto& level = widened ? packed.mips[0] : linear.mips[0];
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.region.width = w;
  copy.region.height = h;
  if (unpacked) {
    rhi::TextureBarrier fb;
    fb.texture = float_image;
    fb.before = rhi::TextureState::kUndefined;
    fb.after = rhi::TextureState::kCopyDst;
    list->Barrier(0, 0, &fb, 1);
    list->CopyBufferToTexture(float_image, src, &copy, 1);
    fb.before = rhi::TextureState::kCopyDst;
    fb.after = rhi::TextureState::kCopySrc;
    list->Barrier(0, 0, &fb, 1);
    rhi::TextureRegion region;
    region.width = w;
    region.height = h;
    list->BlitTexture(rt.texture, region, float_image, region,
                      rhi::Filter::kNearest);
  } else {
    list->CopyBufferToTexture(rt.texture, src, &copy, 1);
  }
  if (from != rhi::TextureState::kUndefined)
    TransitionImage(list, rt.texture, rt.layout, from);
  rt.dirty_for_read = true;
  rt.ever_rendered = true;
  rt.last_frame = g_frame.num;
  e.rt_seq = e.write_seq;
  e.refreshed_serial = rt.render_serial;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  e.last_used_frame = g_frame.num;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_truth_rt_n++;
  return true;
}

bool CsRefreshRtFromTruth(u64 base) {
  auto it = g_cs_ranges.find(base);
  if (it == g_cs_ranges.end())
    return RtRefreshMiss("no-range-at-base");
  CsRange& e = it->second;
  if ((!e.truth && !e.image_staging) || !e.gpu_dirty)
    return RtRefreshMiss(e.gpu_dirty ? "buffer-range" : "clean");
  return e.rt_seq == e.write_seq || CsRefreshRtInFrame(base, e);
}

bool DetileTextureOnGpu(u64 base,
                        const gcn::TextureLayout32& tiled,
                        u32 texel_w,
                        u32 texel_h,
                        rhi::Texture* img,
                        rhi::TextureState state,
                        bool is_3d,
                        u32 first_mip) {
  const u32 elem = tiled.elem_bytes;
  const bool linear_in = gcn::TilingIsLinear(tiled.tiling_idx);
  const u32 block =
      (elem == 8 || elem == 16) && texel_w > tiled.mips[0].width ? 4u : 1u;
  gcn::TextureLayout32 linear = tiled;
  if (!g_frame.list || !elem || elem > 16 ||
      (!linear_in && !gcn::BuildTextureLayout32(
                         linear, tiled.mips[0].width, tiled.mips[0].height,
                         tiled.mips[0].pitch, tiled.layers, tiled.mip_levels, 8,
                         false, elem)))
    return false;
  if (first_mip >= linear.mip_levels)
    return false;
  rhi::BufferTextureCopy copies[16]{};
  for (u32 mip = first_mip; mip < linear.mip_levels; mip++) {
    const auto& level = linear.mips[mip];
    auto& copy = copies[mip - first_mip];
    copy.buffer_offset = level.offset;
    copy.row_length = level.pitch * block;
    copy.image_height = level.stored_height * block;
    copy.region.mip = mip;
    copy.region.layers = is_3d ? 1u : linear.layers;
    copy.region.depth = is_3d ? linear.layers : 1u;
    copy.region.width = base::Max(texel_w >> mip, 1u);
    copy.region.height = base::Max(texel_h >> mip, 1u);
    if (copy.region.width > level.width * block ||
        copy.region.height > level.height * block || (level.offset % elem))
      return false;
  }
  const TileTable* table =
      linear_in ? nullptr : GetTileTable(tiled, linear, elem < 4);
  if (!linear_in && !table)
    return false;
  g_cs_scratch_owner++;
  const u64 raw_bytes = (tiled.size + 3) & ~u64(3);
  rhi::Buffer* raw = AcquireScratch(raw_bytes);
  rhi::Buffer* pixels =
      linear_in || !raw ? raw : AcquireScratch((linear.size + 3) & ~u64(3));
  if (!pixels)
    return false;
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  if (!RecordGuestRead(list, base, tiled.size, raw))
    return false;
  if (table)
    RecordImageTiling(list, *table, raw, 0, raw_bytes, pixels,
                      (linear.size + 3) & ~u64(3), true);
  else
    list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyRead);
  rhi::TextureBarrier barrier{img, state, rhi::TextureState::kCopyDst};
  barrier.range.mips = linear.mip_levels;
  barrier.range.layers = is_3d ? 1u : linear.layers;
  list->Barrier(0, 0, &barrier, 1);
  list->CopyBufferToTexture(img, pixels, copies, linear.mip_levels - first_mip);
  barrier.before = rhi::TextureState::kCopyDst;
  barrier.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &barrier, 1);
  return true;
}

bool CsClearPredicate(u64 address,
                      u64 expand,
                      bool htile,
                      rhi::Buffer** predicate,
                      u64* offset) {
  constexpr u32 kPredicates = 1024;
  struct Pipe {
    rhi::BindGroupLayout* set = nullptr;
    rhi::PipelineLayout* layout = nullptr;
    rhi::Pipeline* pipe = nullptr;
    rhi::Buffer* buf = nullptr;
    bool tried = false;
  };
  static Pipe p;
  static int frame = -1;
  static u32 used = 0;
  const bool mark = expand != 0;
  expand = (base::Max<u64>(expand, 4) + 3) & ~u64(3);
  if (!Device().caps().conditional || !CsBindsImages() || (address & 3))
    return false;
  u64 range_base;
  CsRange* const e = BufferSource(address, expand, &range_base);
  bool pending = false;
  if (!GuestPages().PendingWriteClear(address, address + 4))
    for (const PendingGuestWrite& w : g_pending_guest_writes)
      pending |= w.base < address + 4 && address < w.end;
  if (!e && !(pending && MirrorWritten(address, 4) &&
              MirrorFillable(address, expand)))
    return false;
  struct Revision {
    u64 stamp = 0, seq = 0;
    bool operator==(const Revision&) const = default;
  };
  static base::HashMap<u64, Revision> expanded;
  static const guest::SessionReset reset_expanded(
      [] { guest::ResetResource(expanded); });
  const u64 block = address & ~(kMirrorBlock - 1);
  const auto revision = [&]() -> Revision {
    if (e)
      return Revision{e->staged_stamp, e->write_seq};
    return {0, MirrorRevision(address, 4)};
  };
  if (mark && (revision().seq || revision().stamp)) {
    const auto it = expanded.find(address);
    if (it != expanded.end() && it->second == revision()) {
      *predicate = nullptr;
      return true;
    }
  }
  if (!p.tried) {
    p.tried = true;
#ifdef DELTA_HAVE_SPIRV_BACKEND
    const auto code = gcn::BuildClearPredicateShader();
#else
    const base::Vector<u32> code;
#endif
    rhi::BindGroupLayoutDesc set;
    for (u32 i = 0; i < 2; i++)
      set.bindings.push_back(
          {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
    set.push = true;
    p.set = code.empty() ? nullptr : Device().CreateBindGroupLayout(set);
    rhi::PipelineLayoutDesc layout;
    layout.groups = {p.set};
    layout.push_constant_bytes = 12;
    layout.push_constant_stages = rhi::kStageCompute;
    p.layout = p.set ? Device().CreatePipelineLayout(layout) : nullptr;
    rhi::ComputePipelineDesc pipe;
    pipe.layout = p.layout;
    pipe.code = rhi::Code(code);
    p.pipe = p.layout ? Device().CreateComputePipeline(pipe) : nullptr;
    rhi::BufferDesc desc;
    desc.size = 2 * kPredicates * 4;
    desc.usage = rhi::kBufferStorage | rhi::kBufferPredicate;
    p.buf = p.pipe ? Device().CreateBuffer(desc) : nullptr;
  }
  if (frame != g_frame.num) {
    frame = g_frame.num;
    used = 0;
  }
  if (!p.buf || used == kPredicates)
    return false;
  const u32 index = g_frame.slot_idx * kPredicates + used++;
  rhi::Buffer* const src = e ? e->buf : g_mirror;
  const u64 at = e ? address - range_base : MirrorOffset(address);
  const u64 window = at & ~(u64(Device().caps().storage_offset_alignment) - 1);
  EndRegion();
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  rhi::CommandList* const list = g_cs_list;
  list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessComputeRead);
  list->SetPipeline(p.pipe);
  rhi::BindingWrite writes[2];
  writes[0].binding = 0;
  writes[0].buffer = src;
  writes[0].offset = window;
  writes[0].range = at + 4 - window;
  writes[1].binding = 1;
  writes[1].buffer = p.buf;
  writes[1].range = p.buf->desc().size;
  list->PushBindGroup(0, p.set, writes, 2);
  const u32 params[3] = {static_cast<u32>((at - window) / 4), index,
                         htile ? 1u : 0u};
  list->SetPushConstants(0, sizeof(params), params);
  list->Dispatch(1, 1, 1);
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessComputeRead,
                rhi::kAccessPredicateRead | rhi::kAccessCopyWrite);
  if (!mark) {
  } else if (e) {
    list->FillBuffer(e->buf, at, expand, ~0u);
    list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW | kAccessCopyRW);
    e->write_seq++;
    e->frame_ref = g_frame.num;
    e->chunk_ref = g_frame.chunk_seq;
    if (e->pending_batch)
      g_cs_chunk_needs_batch = true;
    expanded[address] = revision();
  } else {
    MirrorFill(address, expand, ~0u);
    if (revision().seq)
      expanded[address] = revision();
  }
  if (!g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_chunk_needs_batch = true;
  BumpTextureEpoch();
  *predicate = p.buf;
  *offset = u64(index) * 4;
  return true;
}

}  // namespace gpu::render
