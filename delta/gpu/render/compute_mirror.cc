/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute_internal.h"

#include <sys/mman.h>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_direct.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/upload_ring.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"
#include "profile/profile.h"

namespace gpu::render {

constexpr u64 kMirrorImport = 2ull << 20;
rhi::Buffer* g_mirror = nullptr;
bool g_mirror_tried = false;
base::HashMap<u64, rhi::Buffer*> g_mirror_imports;

rhi::Buffer* Mirror() {
  if (!g_mirror_tried) {
    g_mirror_tried = true;
    rhi::BufferDesc desc;
    desc.size = kMirrorSize;
    desc.sparse = true;
    desc.usage = rhi::kBufferStorage | rhi::kBufferUniform |
                 rhi::kBufferCopySrc | rhi::kBufferCopyDst |
                 rhi::kBufferAddress | rhi::kBufferIndirect |
                 rhi::kBufferVertex | rhi::kBufferIndex;
    desc.name = "guest mirror";
    if (Device().caps().sparse_buffer)
      g_mirror = Device().CreateBuffer(desc);
    BASE_LOGI("mirror", "guest mirror {:#x}+{:#x}, {:#x}+{:#x}: {}",
              kMirrorPs4Base, kMirrorLowWindow, kMirrorPs5Base, kMirrorWindow,
              g_mirror ? "up" : "unavailable");
  }
  return g_mirror;
}

rhi::Buffer* MirrorImport(u64 address, u64* import_base) {
  u64 offset = 0, bytes = 0;
  if (auto* direct = DirectImportAt(address, &offset, &bytes);
      direct && direct->desc().host_pointer &&
      bytes == direct->desc().size - offset) {
    *import_base = address - offset;
    return direct;
  }
  const u64 chunk = address & ~(kMirrorImport - 1);
  auto found = g_mirror_imports.find(chunk);
  if (found == g_mirror_imports.end()) {
    rhi::Buffer* buf = nullptr;
    const rhi::Caps& caps = Device().caps();
    if (caps.host_import && gpu::IsWritableRange(chunk, kMirrorImport)) {
      DELTA_ZONE("mirror.import");
      GuestWriteTracker().Release(chunk, kMirrorImport);
      rhi::BufferDesc desc;
      desc.size = kMirrorImport;
      desc.usage = rhi::kBufferCopyDst | rhi::kBufferCopySrc;
      desc.host_pointer = reinterpret_cast<void*>(chunk);
      buf = Device().CreateBuffer(desc);
    }
    found = g_mirror_imports.insert_or_assign(chunk, buf).first;
  }
  if (found->second) {
    *import_base = chunk;
    return found->second;
  }
  *import_base = address & ~(kGuestPageImport - 1);
  return ImportGuestPage(*import_base);
}

bool MirrorEligible(const ComputeInfo::Res& r, u64 bytes) {
  return kMirror && !r.zero_fill && !r.image_staging && bytes &&
         MirrorCovers(r.base, bytes) && !(r.base & 3) && Mirror();
}

bool MirrorCopy(u64 base,
                u64 bytes,
                bool to_guest,
                rhi::CommandList* list = nullptr) {
  base::Vector<base::Pair<rhi::Buffer*, base::Pair<u64, u64>>> pieces;
  for (u64 at = base; at < base + bytes;) {
    u64 import_base = 0;
    rhi::Buffer* import = MirrorImport(at, &import_base);
    if (!import)
      return false;
    const u64 end = base::Min(base + bytes, import_base + import->desc().size);
    pieces.push_back({import, {at, end}});
    at = end;
  }
  if (!list)
    list = g_cs_list;
  list->ProfileBegin(3);
  list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  for (const auto& [import, span] : pieces) {
    u64 import_base = 0;
    MirrorImport(span.first, &import_base);
    const u64 n = span.second - span.first;
    if (to_guest)
      list->CopyBuffer(import, span.first - import_base, g_mirror,
                       MirrorOffset(span.first), n);
    else
      list->CopyBuffer(g_mirror, MirrorOffset(span.first), import,
                       span.first - import_base, n);
  }
  list->Barrier(rhi::kAccessCopyWrite, kAccessAll | rhi::kAccessHostRead);
  list->ProfileEnd();
  return true;
}

bool GuestImportable(u64 base, u64 bytes) {
  for (u64 at = base; at < base + bytes;) {
    u64 import_base = 0;
    rhi::Buffer* import = MirrorImport(at, &import_base);
    if (!import)
      return false;
    at = base::Min(base + bytes, import_base + import->desc().size);
  }
  return true;
}

bool RecordGuestRead(rhi::CommandList* list,
                     u64 base,
                     u64 bytes,
                     rhi::Buffer* dst,
                     u64 dst_off) {
  const auto owned = [](u64 at) {
    return InMirror(at) && MirrorOwned(at & ~(kMirrorBlock - 1));
  };
  const auto block_end = [&](u64 at) {
    return base::Min(base + bytes, (at & ~(kMirrorBlock - 1)) + kMirrorBlock);
  };
  const auto direct = [](u64 at, u64 end) {
    for (u64 offset, n; at < end; at += n)
      if (!DirectImportAt(at, &offset, &n))
        return false;
    return true;
  };
  enum Source { kMirrorSource, kImport, kDirect, kUpload };
  const auto source = [&](u64 at) {
    if (owned(at))
      return kMirrorSource;
    if (GuestImportable(at, block_end(at) - at))
      return kImport;
    return direct(at, block_end(at)) ? kDirect : kUpload;
  };
  list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
  for (u64 at = base; at < base + bytes;) {
    const Source from = source(at);
    u64 end = block_end(at);
    while (end < base + bytes && source(end) == from)
      end = block_end(end);
    const u64 n = end - at;
    if (from == kMirrorSource) {
      list->CopyBuffer(dst, dst_off + at - base, g_mirror, MirrorOffset(at), n);
    } else if (from == kImport) {
      for (u64 piece = at; piece < end;) {
        u64 import_base = 0;
        rhi::Buffer* import = MirrorImport(piece, &import_base);
        const u64 to = base::Min(end, import_base + import->desc().size);
        list->CopyBuffer(dst, dst_off + piece - base, import,
                         piece - import_base, to - piece);
        piece = to;
      }
    } else if (from == kDirect) {
      for (u64 piece = at, offset, len; piece < end; piece += len) {
        rhi::Buffer* import = DirectImportAt(piece, &offset, &len);
        len = base::Min(len, end - piece);
        list->CopyBuffer(dst, dst_off + piece - base, import, offset, len);
      }
    } else {
      if (!gpu::IsReadableRange(at, n))
        return false;
      TextureUploadSlice upload;
      if (!AllocateTextureUpload(g_frame.slot_idx, n, 16, upload)) {
        rhi::BufferDesc desc;
        desc.size = n;
        desc.usage = rhi::kBufferCopySrc;
        desc.memory = rhi::MemoryKind::kUpload;
        rhi::Buffer* own = Device().CreateBuffer(desc);
        if (!own || !own->mapped())
          return false;
        g_cs_frame_retired.push_back(own);
        upload = {own, 0, static_cast<u8*>(own->mapped())};
      }
      std::memcpy(upload.map, reinterpret_cast<const void*>(at), n);
      list->CopyBuffer(dst, dst_off + at - base, upload.buffer, upload.offset,
                       n);
    }
    at = end;
  }
  if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_chunk_needs_batch = true;
  return true;
}

base::HashSet<u64> g_mirror_clean;
base::Vector<base::Pair<u64, u64>> g_mirror_self;

DELTA_OPTION(bool, kMirrorOwn, "DELTA_GPU_MIRROR_OWN", true);
constexpr u32 kOwnMaxFaults = 4;
OwnBlock* g_own = nullptr;
base::Vector<u64> g_own_unsubmitted;

OwnBlock& OwnAt(u64 block) {
  if (!g_own) {
    void* p =
        mmap(nullptr, kOwnBlocks * sizeof(OwnBlock), PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    g_own = static_cast<OwnBlock*>(p);
  }
  return g_own[MirrorOffset(block) / kMirrorBlock];
}

bool MirrorOwned(u64 block) {
  return g_own && InMirror(block) &&
         OwnAt(block).state.load(base::memory_order_acquire) != 0;
}

bool InMirror(u64 address) {
  return MirrorOffset(address) != kNotMirrored;
}

bool MirrorResolve(u64 first, u64 end) {
  if (!g_own)
    return false;
  bool resolved = false;
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    if (!InMirror(block))
      continue;
    OwnBlock& o = OwnAt(block);
    for (;;) {
      u64 s = o.state.load(base::memory_order_acquire);
      if (!(s & kOwnProtected))
        break;
      if ((s & kOwnBusy) || !o.state.compare_exchange_weak(
                                s, s | kOwnBusy, base::memory_order_acq_rel))
        continue;
      if (o.hi > o.lo)
        Device().CopyAfter(g_mirror, MirrorOffset(block) + o.lo, o.import,
                           o.import_offset + o.lo, o.hi - o.lo,
                           s & kOwnSubmission);
      ::mprotect(reinterpret_cast<void*>(block), kMirrorBlock,
                 PROT_READ | PROT_WRITE | PROT_EXEC);
      WriteTracker::OpenedByOwner(block, block + kMirrorBlock);
      if (!(s & kOwnPending)) {
        DirectRestore(block);
        o.lo = o.hi = 0;
      }
      o.cpu_dirty.store(1, base::memory_order_relaxed);
      o.faults.fetch_add(1, base::memory_order_relaxed);
      g_mirror_resolves.fetch_add(1, base::memory_order_relaxed);
      o.state.store(s & kOwnPending, base::memory_order_release);
      resolved = true;
      break;
    }
  }
  return resolved;
}

bool MirrorOwnsAny(u64 first, u64 end) {
  if (!g_own || first >= end ||
      (first >= kMirrorLowWindow &&
       (end <= kMirrorPs5Base || first >= kMirrorPs5Base + kMirrorWindow)))
    return false;
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock)
    if (InMirror(block) && MirrorOwned(block))
      return true;
  return false;
}

bool MirrorUsesImport(rhi::Buffer* buffer) {
  const u64 first = reinterpret_cast<u64>(buffer->desc().host_pointer);
  const u64 end = first + buffer->desc().size;
  if (!MirrorOwnsAny(first, end))
    return false;
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock)
    if (MirrorOwned(block) && OwnAt(block).import == buffer)
      return true;
  return false;
}

bool MirrorOwnable(u64 base, u64 bytes) {
  if (!kMirrorOwn || !gcn::g_direct_table || !Device().caps().copy_after ||
      !g_mirror->address())
    return false;
  for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
       block += kMirrorBlock)
    if (!gpu::IsReadableRangeCached(block, kMirrorBlock) ||
        OwnAt(block).faults.load(base::memory_order_relaxed) >= kOwnMaxFaults)
      return false;
  return true;
}

rhi::Buffer* MirrorImport(u64 address, u64* import_base);

void MirrorOwn(u64 base, u64 bytes) {
  static bool hooked = false;
  if (!hooked) {
    hooked = true;
    WriteTracker::SetOwner(&MirrorResolve, &MirrorOwnsAny);
  }
  const u64 end = base + bytes;
  for (u64 block = base & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    OwnBlock& o = OwnAt(block);
    const u32 lo = static_cast<u32>(base::Max(block, base) - block);
    const u32 hi =
        static_cast<u32>(base::Min(end, block + kMirrorBlock) - block);
    const u64 s = o.state.load(base::memory_order_acquire);
    if (!s) {
      u64 import_base = 0;
      o.import = MirrorImport(block, &import_base);
      o.import_offset = block - import_base;
      o.lo = lo;
      o.hi = hi;
      DirectRedirect(block, g_mirror->address() + MirrorOffset(block));
      g_mirror_was_owned.insert(block);
    } else {
      o.lo = base::Min(o.lo, lo);
      o.hi = base::Max(o.hi, hi);
    }
    if (!(o.state.fetch_or(kOwnPending, base::memory_order_acq_rel) &
          kOwnPending))
      g_own_unsubmitted.push_back(block);
  }
}

void MirrorProtectSubmitted(u64 submission) {
  if (g_own_unsubmitted.empty())
    return;
  DELTA_ZONE("mirror.protect");
  for (u64 block : g_own_unsubmitted) {
    OwnBlock& o = OwnAt(block);
    u64 s = o.state.load(base::memory_order_acquire);
    while ((s & kOwnBusy) || !o.state.compare_exchange_weak(
                                 s, kOwnBusy, base::memory_order_acq_rel))
      s = o.state.load(base::memory_order_acquire);
    ::mprotect(reinterpret_cast<void*>(block), kMirrorBlock, PROT_NONE);
    o.state.store(submission | kOwnProtected, base::memory_order_release);
  }
  g_own_unsubmitted.clear();
}

namespace {
struct MirrorDraw {
  int frame = -1;
  u64 chunk = 0;
} g_mirror_draw;
const guest::SessionReset reset_mirror_draw([] {
  guest::ResetResource(g_mirror_draw);
});

bool OrderMirrorAfterFrameRead() {
  if (g_mirror_draw.frame != g_frame.num ||
      g_mirror_draw.chunk != g_frame.chunk_seq)
    return true;
  if (!CsSplitFrameChunk())
    return false;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  g_cs_list->Barrier(rhi::kAccessVertexRead | rhi::kAccessIndexRead |
                         rhi::kAccessShaderRead | rhi::kAccessUniformRead |
                         rhi::kAccessCopyRead | rhi::kAccessComputeRead |
                         rhi::kAccessIndirectRead,
                     kAccessComputeRW | kAccessCopyRW);
  g_mirror_draw.frame = -1;
  return true;
}
}  // namespace

void MirrorFrameRead() {
  g_mirror_draw = {g_frame.num, g_frame.chunk_seq};
}

bool MirrorCopyIn(Renderer& renderer, u64 base, u64 bytes) {
  if (CsRangeDirtyOverlapping(base, bytes) &&
      !FlushCsWritesForGpu(renderer, base, bytes, "mirror"))
    return false;
  return MirrorCopyInGuest(base, bytes);
}

bool MirrorCopyInGuest(u64 base, u64 bytes) {
  if (!OrderMirrorAfterFrameRead())
    return false;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  const u64 page = base::Max(Device().caps().sparse_page, kMirrorBlock);
  const u64 end = base + bytes;
  u64 run = 0, run_end = 0;
  bool in_run = false;
  const auto copy_run = [&] {
    in_run = false;
    const u64 lo = MirrorOffset(run) & ~(page - 1);
    const u64 hi = (MirrorOffset(run_end - 1) + page) & ~(page - 1);
    return Device().CommitSparse(g_mirror, lo, hi - lo) &&
           MirrorCopy(run, run_end - run, false);
  };
  for (u64 block = base & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    bool fresh = MirrorOwned(block);
    if (!fresh) {
      const bool cpu_dirty = OwnAt(block).cpu_dirty.exchange(0);
      fresh = g_mirror_clean.count(block) && !cpu_dirty;
    }
    if (fresh) {
      if (in_run && !copy_run())
        return false;
      continue;
    }
    const bool whole = gpu::IsReadableRangeCached(block, kMirrorBlock);
    const u64 at = whole ? block : base::Max(block, base);
    const u64 to =
        whole ? block + kMirrorBlock : base::Min(end, block + kMirrorBlock);
    if (in_run && at != run_end && !copy_run())
      return false;
    if (!in_run)
      run = at;
    in_run = true;
    run_end = to;
    if (whole && GuestWriteTracker().Arm(block, kMirrorBlock))
      g_mirror_clean.insert(block);
  }
  return !in_run || copy_run();
}

static bool PrepareGuestWrite(u64 base, u64 bytes, bool* mirrored) {
  *mirrored = kMirror && Mirror() && MirrorCovers(base, bytes);
  if (CsTracksGuestWrites())
    CollectGuestWritesForSubmission();
  if (!OrderMirrorAfterFrameRead())
    return false;
  return *mirrored ? MirrorCopyInGuest(base, bytes)
                   : GuestImportable(base, bytes);
}

static void RecordGuestWrite(rhi::CommandList* c,
                             bool mirrored,
                             u64 base,
                             u64 n,
                             rhi::Buffer* src,
                             const base::Vector<base::Pair<u64, u64>>& skip,
                             u64 src_off = 0) {
  c->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
  const auto copy = [&](u64 lo, u64 hi) {
    if (mirrored) {
      c->CopyBuffer(g_mirror, MirrorOffset(lo), src, src_off + lo - base,
                    hi - lo);
      return;
    }
    for (u64 at = lo; at < hi;) {
      u64 import_base = 0;
      rhi::Buffer* import = MirrorImport(at, &import_base);
      const u64 to = base::Min(hi, import_base + import->desc().size);
      c->CopyBuffer(import, at - import_base, src, src_off + at - base,
                    to - at);
      at = to;
    }
  };
  u64 at = base;
  for (const auto& [lo, hi] : skip) {
    if (at < lo)
      copy(at, lo);
    at = base::Max(at, hi);
  }
  if (at < base + n)
    copy(at, base + n);
  c->Barrier(rhi::kAccessCopyWrite,
             kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostRead);
  if (mirrored) {
    MirrorWriteBack(base, n);
  } else {
    NotePendingGuestWrite(base, base + n);
    NoteGuestWrite(base, n);
  }
}

bool GpuWriteBack(u64 base, CsRange& e, bool* kept) {
  *kept = false;
  if (!e.buf || e.imported || (base & 3))
    return false;
  const bool retile = e.image_staging && !e.truth;
  gcn::TextureLayout32 tiled, linear;
  const TileTable* table = nullptr;
  if (retile) {
    if (e.res.dfmt == 35 || e.res.dfmt == 40 ||
        !BuildCsImageLayouts(e.res, tiled, linear))
      return false;
    table = GetTileTable(
        tiled, linear,
        tiled.elem_bytes < 4 && linear.elem_bytes == tiled.elem_bytes);
    if (!table)
      return false;
  }
  const u64 n = retile          ? tiled.size
                : e.guest_bytes ? base::Min(e.size, e.guest_bytes)
                                : e.size;
  bool mirrored = false;
  if (!PrepareGuestWrite(base, n, &mirrored))
    return false;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  rhi::CommandList* const c = g_cs_list;
  rhi::Buffer* src = e.buf;
  if (retile) {
    const u64 padded = (n + 3) & ~u64(3);
    src = AcquireScratch(padded);
    if (!src || !RecordGuestRead(c, base, n, src))
      return false;
    RecordImageTiling(c, *table, src, 0, padded, e.buf,
                      (linear.size + 3) & ~u64(3), false);
  }
  base::Vector<base::Pair<u64, u64>> skip;
  for (const auto* spans : {&e.cpu_writes, &e.fill_kept})
    for (const auto& [lo, hi] : *spans)
      if (base::Max(lo, base) < base::Min(hi, base + n))
        skip.emplace_back(base::Max(lo, base), base::Min(hi, base + n));
  base::Sort(skip.begin(), skip.end());
  *kept = !skip.empty();
  RecordGuestWrite(c, mirrored, base, n, src, skip);
  e.fill_kept.clear();
  MarkPending(e);
  return true;
}

bool WriteTextureToGuest(rhi::Texture* texture,
                         rhi::TextureState& state,
                         u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear) {
  const TileTable* table = GetTileTable(tiled, linear, tiled.elem_bytes < 4);
  if (!table || !g_frame.recording)
    return false;
  CsBatchBeginImpl();
  if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame) {
    if (!CsBatchSubmit())
      return false;
    CsBatchBeginImpl();
  }
  if (!g_cs_batch_open || !g_cs_batches[g_cs_batch_cur].in_frame)
    return false;
  g_cs_scratch_owner++;
  const u64 tiled_bytes = (tiled.size + 3) & ~u64(3);
  const u64 linear_bytes = (linear.size + 3) & ~u64(3);
  rhi::Buffer* raw = AcquireScratch(tiled_bytes);
  rhi::Buffer* pixels = raw ? AcquireScratch(linear_bytes) : nullptr;
  bool mirrored = false;
  if (!pixels || !PrepareGuestWrite(base, tiled.size, &mirrored))
    return false;
  rhi::CommandList* const c = g_cs_list;
  if (!RecordGuestRead(c, base, tiled.size, raw))
    return false;
  const auto& level = linear.mips[0];
  rhi::BufferTextureCopy copy;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  copy.region.width = level.width;
  copy.region.height = level.height;
  const rhi::TextureState old_state = state;
  TransitionImage(c, texture, state, rhi::TextureState::kCopySrc);
  c->CopyTextureToBuffer(pixels, texture, &copy, 1);
  TransitionImage(c, texture, state, old_state);
  RecordImageTiling(c, *table, raw, 0, tiled_bytes, pixels, linear_bytes,
                    false);
  RecordGuestWrite(c, mirrored, base, tiled.size, raw, {});
  return true;
}

base::Map<u64, MirrorWrite> g_mirror_writes;
u64 g_mirror_seq = 0;

const MirrorWrite* MirrorWritten(u64 base, u64 bytes) {
  auto it = g_mirror_writes.upper_bound(base);
  if (it == g_mirror_writes.begin())
    return nullptr;
  --it;
  return base + bytes <= it->second.end ? &it->second : nullptr;
}

namespace {
void RecordMirrorWrite(u64 first, u64 bytes) {
  u64 end = first + bytes;
  auto it = g_mirror_writes.upper_bound(first);
  if (it != g_mirror_writes.begin()) {
    --it;
    if (end <= it->second.end) {
      it->second.seq = ++g_mirror_seq;
      return;
    }
    if (it->second.end < first)
      ++it;
  }
  while (it != g_mirror_writes.end() && it->first <= end) {
    first = base::Min(first, it->first);
    end = base::Max(end, it->second.end);
    it = g_mirror_writes.erase(it);
  }
  g_mirror_writes[first] = {end, ++g_mirror_seq};
}
}  // namespace

void MirrorWriteBack(u64 base, u64 bytes) {
  if (MirrorOwnable(base, bytes)) {
    RecordMirrorWrite(base, bytes);
    MirrorOwn(base, bytes);
    g_mirror_own_bytes += bytes;
    g_mirror_self.push_back({base, base + bytes});
    NoteGuestWrite(base, bytes);
    InvalidateTexRange(base, bytes);
    return;
  }
  g_mirror_wb_bytes += bytes;
  g_mirror_wb_n++;
  {
    u64& e = g_mirror_wb_union[base];
    e = base::Max(e, base + bytes);
  }
  RecordMirrorWrite(base, bytes);
  if (!MirrorCopy(base, bytes, true))
    return;
  NotePendingGuestWrite(base, base + bytes);
  g_mirror_self.push_back({base, base + bytes});
  NoteGuestWrite(base, bytes);
  InvalidateTexRange(base, bytes);
}

bool MirrorFillable(u64 base, u64 bytes) {
  return kMirror && Mirror() && !(base & 3) && !(bytes & 3) &&
         MirrorCovers(base, bytes) &&
         (MirrorOwnable(base, bytes) || GuestImportable(base, bytes));
}

bool MirrorFill(u64 base, u64 bytes, u32 value) {
  if (!MirrorFillable(base, bytes) || !OrderMirrorAfterFrameRead())
    return false;
  if (MirrorOwnable(base, bytes)) {
    if (!MirrorCopyInGuest(base, bytes))
      return false;
  } else {
    const u64 page = base::Max(Device().caps().sparse_page, kMirrorBlock);
    const u64 lo = MirrorOffset(base) & ~(page - 1);
    const u64 hi = (MirrorOffset(base) + bytes + page - 1) & ~(page - 1);
    if (!Device().CommitSparse(g_mirror, lo, hi - lo))
      return false;
  }
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  g_cs_list->FillBuffer(g_mirror, MirrorOffset(base), bytes, value);
  g_cs_list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW | kAccessCopyRW);
  MirrorWriteBack(base, bytes);
  return true;
}

bool GpuFill(u64 base, u64 bytes, u32 lo, u32 hi) {
  if (!bytes || ((base | bytes) & (lo == hi ? 3 : 7)))
    return false;
  if (lo == hi && MirrorFill(base, bytes, lo))
    return true;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  g_cs_scratch_owner++;
  rhi::Buffer* src = AcquireScratch(bytes);
  if (!src)
    return false;
  rhi::CommandList* const c = g_cs_list;
  c->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  if (lo == hi) {
    c->FillBuffer(src, 0, bytes, lo);
  } else {
    const u32 pattern[2] = {lo, hi};
    c->UpdateBuffer(src, 0, sizeof(pattern), pattern);
    for (u64 n = sizeof(pattern); n < bytes; n *= 2) {
      c->Barrier(rhi::kAccessCopyWrite, kAccessCopyRW);
      c->CopyBuffer(src, n, src, 0, base::Min(n, bytes - n));
    }
  }
  return GpuWriteGuest(base, bytes, src, 0);
}

bool GpuCopy(u64 dst, u64 src, u64 bytes) {
  CsBatchBeginImpl();
  if (!bytes || !g_cs_batch_open)
    return false;
  g_cs_scratch_owner++;
  rhi::Buffer* tmp = AcquireScratch(bytes);
  return tmp && RecordGuestRead(g_cs_list, src, bytes, tmp) &&
         GpuWriteGuest(dst, bytes, tmp, 0);
}

bool GpuWriteGuest(u64 dst, u64 bytes, rhi::Buffer* src, u64 src_off) {
  CsBatchBeginImpl();
  bool mirrored = false;
  if (!g_cs_batch_open || !PrepareGuestWrite(dst, bytes, &mirrored))
    return false;
  RecordGuestWrite(g_cs_list, mirrored, dst, bytes, src, {}, src_off);
  return true;
}

static bool MirrorWatch(u64 first, u64 end) {
  if (GuestWriteTracker().Arm(first, end - first, true))
    return true;
  for (u64 page = first; page < end; page += kMirrorHotPage)
    if (!GuestWriteTracker().Arm(page, kMirrorHotPage, true))
      g_mirror_hot.insert(page);
  return false;
}

u64 MirrorCacheBlock(u64 block) {
  if (!kMirror || !Mirror() || !g_mirror->address() || !InMirror(block) ||
      MirrorOwned(block) || !gpu::IsReadableRangeCached(block, kMirrorBlock))
    return 0;
  const u64 address = g_mirror->address() + MirrorOffset(block);
  if (g_mirror_cached.count(block))
    return address;
  if (!Device().CommitSparse(g_mirror, MirrorOffset(block), kMirrorBlock))
    return 0;
  if (!GuestImportable(block, kMirrorBlock))
    return 0;
  const bool watched = MirrorWatch(block, block + kMirrorBlock);
  rhi::CommandList* list = nullptr;
  if (g_frame.recording) {
    EndRegion();
    list = g_frame.list;
  } else {
    CsBatchBeginImpl();
    if (!g_cs_batch_open)
      return 0;
  }
  if (!MirrorCopy(block, kMirrorBlock, false, list))
    return 0;
  if (watched)
    g_mirror_clean.insert(block);
  g_mirror_cached.insert(block);
  return address;
}

bool GpuIndirectArgs(u64 args, u64 bytes) {
  if (!kMirror || !gcn::g_direct_table || !Mirror() || !g_mirror->address() ||
      (args & 3) || !InMirror(args) ||
      (args & ~(kMirrorBlock - 1)) !=
          ((args + bytes - 1) & ~(kMirrorBlock - 1)))
    return false;
  CollectGuestWrites();
  const u64 block = args & ~(kMirrorBlock - 1);
  if (MirrorOwned(block))
    return !(OwnAt(block).state.load(base::memory_order_acquire) &
             kOwnPending) ||
           CsBatchSubmit();
  return MirrorCacheBlock(block) != 0;
}

bool CanDrawIndirect(u32 prim_type, bool indexed) {
  return Device().caps().draw_indirect && (prim_type != 19 || indexed) &&
         !(Device().caps().backend == rhi::Backend::kD3D12 && prim_type == 5);
}

bool CanDrawMeshIndirect() {
  return Device().caps().mesh_shader;
}

rhi::Buffer* IndirectArgsBuffer(u64 args, u64* offset) {
  MirrorFrameRead();
  *offset = MirrorOffset(args);
  return g_mirror;
}

constexpr u64 kIndirectRing = 4ull << 20;
constexpr u64 kIndirectEntry = 32;

static rhi::Buffer* StageArgs(rhi::CommandList* list,
                              u64 args,
                              u64 bytes,
                              u64* offset,
                              u32 dst_access) {
  if (bytes > kIndirectEntry || !gpu::IsReadableRangeCached(args, bytes))
    return nullptr;
  if (!g_indirect_ring) {
    rhi::BufferDesc desc;
    desc.size = kIndirectRing;
    desc.usage = rhi::kBufferIndirect | rhi::kBufferCopySrc |
                 rhi::kBufferCopyDst | rhi::kBufferStorage;
    if (Device().caps().conditional)
      desc.usage |= rhi::kBufferPredicate;
    desc.name = "indirect args";
    g_indirect_ring = Device().CreateBuffer(desc);
    if (!g_indirect_ring)
      return nullptr;
  }
  if (g_indirect_frame != g_frame.num) {
    g_indirect_frame = g_frame.num;
    g_indirect_frame_n = 0;
  }
  if (++g_indirect_frame_n > kIndirectRing / kIndirectEntry / 4)
    return nullptr;
  *offset = g_indirect_head;
  g_indirect_head = (g_indirect_head + kIndirectEntry) % kIndirectRing;
  if (!RecordGuestRead(list, args, bytes, g_indirect_ring, *offset))
    return nullptr;
  list->Barrier(rhi::kAccessCopyWrite, dst_access);
  g_indirect_staged_n++;
  return g_indirect_ring;
}

rhi::Buffer* StageIndirectArgs(rhi::CommandList* list,
                               u64 args,
                               u64 bytes,
                               u64* offset) {
  return StageArgs(list, args, bytes, offset, rhi::kAccessIndirectRead);
}

bool BeginGuestPredicate(u64 flag) {
  if (!Device().caps().conditional || !g_frame.recording || g_predicate ||
      !FlushCsWritesForGpu(DefaultRenderer(), flag, 4, "cond"))
    return false;
  EndRegion();
  g_predicate = StageArgs(g_frame.list, flag, 4, &g_predicate_offset,
                          rhi::kAccessPredicateRead);
  return g_predicate != nullptr;
}

void EndGuestPredicate() {
  g_predicate = nullptr;
}

rhi::Buffer* GuestPredicate(u64* offset) {
  *offset = g_predicate_offset;
  return g_predicate;
}

namespace {
bool MirrorLive(u64 at) {
  const u64 block = at & ~(kMirrorBlock - 1);
  return g_mirror_cached.count(block) && !MirrorOwned(block);
}

rhi::CommandList* MirrorRefreshList() {
  if (g_frame.recording) {
    EndRegion();
    return g_frame.list;
  }
  CsBatchBeginImpl();
  return g_cs_batch_open ? g_cs_list : nullptr;
}
}  // namespace

void MirrorRefreshWritten() {
  if (g_mirror_stale.empty())
    return;
  rhi::CommandList* list = MirrorRefreshList();
  if (!list)
    return;
  list->Barrier(kAccessAll, rhi::kAccessCopyWrite);
  for (const auto& [first, end] : g_mirror_stale) {
    if (!MirrorLive(first))
      continue;
    if (MirrorWatch(first, end))
      g_mirror_clean.insert(first & ~(kMirrorBlock - 1));
    MirrorCopy(first, end - first, false, list);
  }
  g_mirror_stale.clear();
}

void MirrorRefreshCached() {
  CollectGuestWrites();
  static int scanned_frame = -1;
  base::Vector<u64> released;
  if (scanned_frame != g_frame.num) {
    scanned_frame = g_frame.num;
    for (u64 block : g_mirror_was_owned)
      if (!MirrorOwned(block))
        released.push_back(block);
  }
  for (u64 block : released) {
    g_mirror_was_owned.erase(block);
    g_mirror_cached.erase(block);
    if (const u64 address = MirrorCacheBlock(block))
      DirectRedirect(block, address);
  }
  if (g_mirror_hot.empty())
    return;
  rhi::CommandList* list = MirrorRefreshList();
  if (!list)
    return;
  list->Barrier(kAccessAll, rhi::kAccessCopyWrite);
  base::Vector<u64> pages(g_mirror_hot.begin(), g_mirror_hot.end());
  base::Sort(pages.begin(), pages.end());
  for (size_t i = 0; i < pages.size();) {
    const u64 first = pages[i];
    u64 end = first + kMirrorHotPage;
    for (++i; i < pages.size() && pages[i] == end; ++i)
      end += kMirrorHotPage;
    const bool copy = MirrorLive(first);
    for (u64 page = first; page < end; page += kMirrorHotPage)
      if (!copy || GuestWriteTracker().Arm(page, kMirrorHotPage, true))
        g_mirror_hot.erase(page);
    if (copy)
      MirrorCopy(first, end - first, false, list);
  }
}

namespace {
void AdvanceMirrorWindows(u64 first, u64 end) {
  auto it = g_mirror_writes.upper_bound(first);
  if (it != g_mirror_writes.begin())
    --it;
  u64 revision = 0;
  for (; it != g_mirror_writes.end() && it->first < end; ++it)
    if (first < it->second.end) {
      if (!revision)
        revision = ++g_mirror_seq;
      it->second.seq = revision;
    }
}

bool WatchMirrorHostReads(u64 base, u64 bytes) {
  bool tracked = true;
  for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
       block += kMirrorBlock)
    if (!InMirror(block) || !MirrorOwned(block)) {
      const u64 lo = base::Max(base, block);
      const u64 hi = base::Min(base + bytes, block + kMirrorBlock);
      tracked &= GuestWriteTracker().Arm(lo, hi - lo);
    }
  return tracked;
}
}  // namespace

bool MirrorSupplyTexture(u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear,
                         rhi::Texture* img,
                         rhi::TextureState state,
                         u64* seq) {
  const MirrorWrite* written = MirrorWritten(base, tiled.size);
  if (!written || !g_frame.recording)
    return false;
  const TileTable* table = GetTileTable(tiled, linear, tiled.elem_bytes < 4);
  if (!table)
    return false;
  if (!img)
    return true;
  if (*seq == written->seq)
    return true;
  const u64 revision = written->seq;
  g_cs_scratch_owner++;
  rhi::Buffer* scratch = AcquireScratch(linear.size);
  const bool mirrored = MirrorHoldsNewest(base, tiled.size);
  rhi::Buffer* source = mirrored ? g_mirror : AcquireScratch(tiled.size);
  if (!scratch || !source)
    return false;
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
                rhi::kAccessCopyRead | rhi::kAccessComputeRead);
  if (!mirrored && !RecordGuestRead(list, base, tiled.size, source))
    return false;
  list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessComputeRead);
  RecordImageTiling(list, *table, source, mirrored ? MirrorOffset(base) : 0,
                    tiled.size, scratch, linear.size, true);
  MirrorFrameRead();
  rhi::TextureBarrier b;
  b.texture = img;
  b.before = state;
  b.after = rhi::TextureState::kCopyDst;
  b.range.mips = linear.mip_levels;
  b.range.layers = linear.layers;
  list->Barrier(0, 0, &b, 1);
  list->CopyBufferToTexture(img, scratch, copies, linear.mip_levels);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  *seq = revision;
  if (!mirrored && !WatchMirrorHostReads(base, tiled.size))
    AdvanceMirrorWindows(base, base + tiled.size);
  if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_chunk_needs_batch = true;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += tiled.size;
  return true;
}

u64 MirrorRevision(u64 base, u64 bytes) {
  const MirrorWrite* written = MirrorWritten(base, bytes);
  return written && g_frame.recording ? written->seq : 0;
}

bool MirrorSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off) {
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  const bool mirrored = MirrorHoldsNewest(base, bytes);
  if (mirrored)
    list->CopyBuffer(dst, dst_off, g_mirror, MirrorOffset(base), bytes);
  else if (!RecordGuestRead(list, base, bytes, dst, dst_off))
    return false;
  MirrorFrameRead();
  list->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  if (!mirrored && !WatchMirrorHostReads(base, bytes))
    AdvanceMirrorWindows(base, base + bytes);
  if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_chunk_needs_batch = true;
  g_buf_bridge_n++;
  g_buf_bridge_bytes += bytes;
  return true;
}

void MirrorNoteGuestWrites(u64 first, u64 end, bool announced) {
  if (g_mirror_clean.empty() && g_mirror_writes.empty())
    return;
  if (first >= end ||
      (first >= kMirrorLowWindow &&
       (end <= kMirrorPs5Base || first >= kMirrorPs5Base + kMirrorWindow)))
    return;
  if (announced)
    for (const auto& [lo, hi] : g_mirror_self)
      if ((lo & ~u64(0xFFF)) <= first && end <= ((hi + 0xFFF) & ~u64(0xFFF)))
        return;
  AdvanceMirrorWindows(first, end);
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    if (g_own && InMirror(block) && !MirrorOwned(block))
      OwnAt(block).cpu_dirty.store(1, base::memory_order_release);
    g_mirror_clean.erase(block);
    if (g_mirror_cached.count(block))
      g_mirror_stale.push_back(
          {base::Max(first, block), base::Min(end, block + kMirrorBlock)});
  }
}

void MirrorWritesCollected() {
  g_mirror_self.clear();
}

}  // namespace gpu::render
