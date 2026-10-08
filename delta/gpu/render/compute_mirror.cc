/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// DELTA_GPU_MIRROR: one sparse device buffer that caches guest memory for
// the plain buffers a dispatch binds.

#include "gpu/render/compute_internal.h"

#include <sys/mman.h>

#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_direct.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"

namespace gpu::render {

// Imports in chunks: a device allows only a few thousand allocations.
constexpr u64 kMirrorImport = 2ull << 20;
rhi::Buffer* g_mirror = nullptr;
bool g_mirror_tried = false;
base::HashMap<u64, rhi::Buffer*> g_mirror_imports;  // chunk -> import

rhi::Buffer* Mirror() {
  if (!g_mirror_tried) {
    g_mirror_tried = true;
    rhi::BufferDesc desc;
    desc.size = kMirrorSize;
    desc.sparse = true;
    desc.usage = rhi::kBufferStorage | rhi::kBufferCopySrc |
                 rhi::kBufferCopyDst | rhi::kBufferAddress |
                 rhi::kBufferIndirect;
    desc.name = "guest mirror";
    if (Device().caps().sparse_buffer)
      g_mirror = Device().CreateBuffer(desc);
    BASE_LOGI("mirror", "guest mirror {:#x},{:#x} +{:#x} each: {}",
              kMirrorPs4Base, kMirrorPs5Base, kMirrorWindow,
              g_mirror ? "up" : "unavailable");
  }
  return g_mirror;
}

// The import that holds the guest bytes at `address`, and its base: a 2 MiB
// chunk, or the 64 KiB page when the chunk spans unmapped memory.
rhi::Buffer* MirrorImport(u64 address, u64* import_base) {
  const u64 chunk = address & ~(kMirrorImport - 1);
  auto found = g_mirror_imports.find(chunk);
  if (found == g_mirror_imports.end()) {
    rhi::Buffer* buf = nullptr;
    const rhi::Caps& caps = Device().caps();
    if (caps.host_import && gpu::IsReadableRange(chunk, kMirrorImport)) {
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
         MirrorCovers(r.base, bytes) &&
         !(r.base & 3) && Mirror();
}

// Copies [base, base+bytes) between the guest pages and the mirror, on the
// batch list, piece by piece along the imports.
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
    const u64 end =
        base::Min(base + bytes, import_base + import->desc().size);
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
  list->Barrier(rhi::kAccessCopyWrite,
                kAccessAll | rhi::kAccessHostRead);
  list->ProfileEnd();
  return true;
}

base::HashSet<u64> g_mirror_clean;
base::Vector<base::Pair<u64, u64>> g_mirror_self;

// GPU-owned blocks (DELTA_GPU_MIRROR_OWN): a block a dispatch wrote stays in
// the mirror, which the direct table then points at, and nothing is copied
// back. Once the batch is submitted the guest pages are made inaccessible; the
// first CPU touch faults, waits for the batch, copies the written span back on
// a queue of its own and hands the block back to the host. A block the CPU
// keeps touching is not taken again.
DELTA_OPTION(bool, kMirrorOwn, "DELTA_GPU_MIRROR_OWN", true);
constexpr u32 kOwnMaxFaults = 4;
OwnBlock* g_own = nullptr;
base::Vector<u64> g_own_unsubmitted;

OwnBlock& OwnAt(u64 block) {
  if (!g_own) {
    void* p = mmap(nullptr, kOwnBlocks * sizeof(OwnBlock),
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    g_own = static_cast<OwnBlock*>(p);
  }
  return g_own[MirrorOffset(block) / kMirrorBlock];
}

bool MirrorOwned(u64 block) {
  return g_own && OwnAt(block).state.load(base::memory_order_acquire) != 0;
}

bool InMirror(u64 address) {
  return MirrorOffset(address) != kNotMirrored;
}

// Any thread, from the fault handler too: hands every protected block of
// [first, end) back to the host.
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
      if ((s & kOwnBusy) ||
          !o.state.compare_exchange_weak(s, s | kOwnBusy,
                                         base::memory_order_acq_rel))
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
  if (!g_own)
    return false;
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock)
    if (InMirror(block) && MirrorOwned(block))
      return true;
  return false;
}

// Whether a dispatch's output at [base, base+bytes) can stay on the GPU.
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
    const u32 hi = static_cast<u32>(base::Min(end, block + kMirrorBlock) - block);
    const u64 s = o.state.load(base::memory_order_acquire);
    if (!s) {
      u64 import_base = 0;
      o.import = MirrorImport(block, &import_base);
      o.import_offset = block - import_base;
      o.lo = lo;
      o.hi = hi;
      DirectRedirect(block, g_mirror->address() + MirrorOffset(block));
    } else {
      o.lo = base::Min(o.lo, lo);
      o.hi = base::Max(o.hi, hi);
    }
    if (!(o.state.fetch_or(kOwnPending, base::memory_order_acq_rel) &
          kOwnPending))
      g_own_unsubmitted.push_back(block);
  }
}

// The batch that wrote them is on the queue: the blocks become inaccessible
// to the CPU until it asks for them.
void MirrorProtectSubmitted(u64 submission) {
  if (g_own_unsubmitted.empty())
    return;
  for (u64 block : g_own_unsubmitted) {
    OwnBlock& o = OwnAt(block);
    u64 s = o.state.load(base::memory_order_acquire);
    while ((s & kOwnBusy) ||
           !o.state.compare_exchange_weak(s, kOwnBusy,
                                          base::memory_order_acq_rel))
      s = o.state.load(base::memory_order_acquire);
    ::mprotect(reinterpret_cast<void*>(block), kMirrorBlock, PROT_NONE);
    o.state.store(submission | kOwnProtected, base::memory_order_release);
  }
  g_own_unsubmitted.clear();
}

// Before a dispatch reads it: backed, and holding the guest's bytes.
bool MirrorCopyIn(Renderer& renderer, u64 base, u64 bytes) {
  // A range still owned by the staging path is written back first.
  if (CsRangeDirtyOverlapping(base, bytes) &&
      !FlushCsWritesRange(renderer, base, bytes, "mirror"))
    return false;
  const u64 page = base::Max(Device().caps().sparse_page, kMirrorBlock);
  const u64 lo = MirrorOffset(base) & ~(page - 1);
  const u64 hi = (MirrorOffset(base) + bytes + page - 1) & ~(page - 1);
  if (!Device().CommitSparse(g_mirror, lo, hi - lo))
    return false;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  // Only the blocks the CPU may have changed: runs of them, one copy each.
  // Whole blocks where they are mapped, as a block may become the GPU's. A
  // block the GPU owns holds the newest bytes already.
  const u64 end = base + bytes;
  u64 run = 0, run_end = 0;
  bool in_run = false;
  for (u64 block = base & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    const bool whole = gpu::IsReadableRangeCached(block, kMirrorBlock);
    const u64 at = whole ? block : base::Max(block, base);
    const u64 to = whole ? block + kMirrorBlock : base::Min(end, block + kMirrorBlock);
    bool fresh = MirrorOwned(block);
    if (!fresh) {
      const bool cpu_dirty = OwnAt(block).cpu_dirty.exchange(0);
      fresh = g_mirror_clean.count(block) && !cpu_dirty;
    }
    if (!fresh && in_run && at != run_end) {
      if (!MirrorCopy(run, run_end - run, /*to_guest=*/false))
        return false;
      in_run = false;
    }
    if (!fresh) {
      if (!in_run)
        run = at;
      in_run = true;
      run_end = to;
    } else if (in_run) {
      if (!MirrorCopy(run, run_end - run, /*to_guest=*/false))
        return false;
      in_run = false;
    }
    if (!fresh && whole && GuestWriteTracker().Arm(block, kMirrorBlock))
      g_mirror_clean.insert(block);
  }
  return !in_run || MirrorCopy(run, run_end - run, /*to_guest=*/false);
}

base::Map<u64, MirrorWrite> g_mirror_writes;
u64 g_mirror_seq = 0;

// The write that holds [base, base+bytes) whole, if one does.
const MirrorWrite* MirrorWritten(u64 base, u64 bytes) {
  auto it = g_mirror_writes.upper_bound(base);
  if (it == g_mirror_writes.begin())
    return nullptr;
  --it;
  return it->first <= base && base + bytes <= it->second.end ? &it->second
                                                             : nullptr;
}

// After a dispatch wrote it: back out to the guest pages, in queue order.
void MirrorWriteBack(u64 base, u64 bytes) {
  if (MirrorOwnable(base, bytes)) {
    MirrorWrite& w = g_mirror_writes[base];
    w.end = base::Max(w.end, base + bytes);
    w.seq = ++g_mirror_seq;
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
  MirrorWrite& w = g_mirror_writes[base];
  w.end = base::Max(w.end, base + bytes);
  w.seq = ++g_mirror_seq;
  if (!MirrorCopy(base, bytes, /*to_guest=*/true))
    return;
  g_pending_guest_writes.push_back({base, base + bytes, g_cs_batch_id});
  g_mirror_self.push_back({base, base + bytes});
  NoteGuestWrite(base, bytes);
  InvalidateTexRange(base, bytes);
}

// Direct reads of guest memory come from VRAM: a block is copied into the
// mirror on its first read, watched, and handed back to the host pages while
// the CPU writes it, to come back after a few quiet frames.

u64 MirrorCacheBlock(u64 block) {
  if (!kMirror || !Mirror() || !g_mirror->address() || !InMirror(block) ||
      MirrorOwned(block) || !gpu::IsReadableRangeCached(block, kMirrorBlock))
    return 0;
  const u64 address = g_mirror->address() + MirrorOffset(block);
  if (g_mirror_cached.count(block))
    return address;
  if (!Device().CommitSparse(g_mirror, MirrorOffset(block), kMirrorBlock))
    return 0;
  if (!GuestWriteTracker().Arm(block, kMirrorBlock))
    return 0;
  rhi::CommandList* list = nullptr;
  if (g_frame.recording) {
    EndRegion();
    list = g_frame.list;
  } else {
    CsBatchBeginImpl();
    if (!g_cs_batch_open)
      return 0;
  }
  if (!MirrorCopy(block, kMirrorBlock, /*to_guest=*/false, list))
    return 0;
  g_mirror_clean.insert(block);
  g_mirror_cached.insert(block);
  g_mirror_uncached.erase(block);
  return address;
}

bool GpuIndirectArgs(u64 args, u64 bytes) {
  if (!kMirror || !gcn::g_direct_table || !Mirror() || !g_mirror->address() ||
      (args & 3) || !InMirror(args) ||
      (args & ~(kMirrorBlock - 1)) != ((args + bytes - 1) & ~(kMirrorBlock - 1)))
    return false;
  const u64 block = args & ~(kMirrorBlock - 1);
  if (MirrorOwned(block))
    return !(OwnAt(block).state.load(base::memory_order_acquire) &
             kOwnPending) ||
           CsBatchSubmit();
  return MirrorCacheBlock(block) != 0;
}

rhi::Buffer* IndirectArgsBuffer(u64 args, u64* offset) {
  *offset = MirrorOffset(args);
  return g_mirror;
}

void MirrorUncache(u64 block) {
  if (!g_mirror_cached.erase(block))
    return;
  if (!MirrorOwned(block))
    DirectRestore(block);
  g_mirror_uncached[block] = g_frame.num;
}

void MirrorRecache() {
  constexpr u64 kQuietFrames = 3;
  u32 budget = 32;
  for (auto it = g_mirror_uncached.begin();
       it != g_mirror_uncached.end() && budget;) {
    const u64 block = it->first;
    if (it->second + kQuietFrames > g_frame.num) {
      ++it;
      continue;
    }
    ++it;
    budget--;
    if (const u64 address = MirrorCacheBlock(block))
      DirectRedirect(block, address);
    else
      g_mirror_uncached.erase(block);
  }
}

// A texture over bytes a dispatch wrote through the mirror: detiled from the
// mirror on the GPU, where reading the guest copy would wait for the write
// back.
bool MirrorSupplyTexture(u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear,
                         rhi::Texture* img,
                         rhi::TextureState state,
                         u64* seq) {
  const MirrorWrite* written = MirrorWritten(base, tiled.size);
  if (!written || !g_frame.recording)
    return false;
  const TileTable* table =
      GetTileTable(tiled, linear, /*packed=*/tiled.elem_bytes < 4);
  if (!table)
    return false;
  if (!img)
    return true;
  if (*seq == written->seq)
    return true;
  rhi::Buffer* scratch = AcquireScratch(linear.size);
  if (!scratch)
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
  RecordImageTiling(list, *table, g_mirror, MirrorOffset(base), tiled.size,
                    scratch, linear.size, /*detile=*/true);
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
  *seq = written->seq;
  if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_chunk_needs_batch = true;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += tiled.size;
  return true;
}

// A buffer read on the GPU (a draw's cbuffer or raw window) of bytes a
// dispatch wrote through the mirror: copied from there.
u64 MirrorRevision(u64 base, u64 bytes) {
  const MirrorWrite* written = MirrorWritten(base, bytes);
  return written && g_frame.recording ? written->seq : 0;
}

bool MirrorSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off) {
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  list->CopyBuffer(dst, dst_off, g_mirror, MirrorOffset(base), bytes);
  list->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  if (g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_chunk_needs_batch = true;
  g_buf_bridge_n++;
  g_buf_bridge_bytes += bytes;
  return true;
}

void MirrorNoteGuestWrites(u64 first, u64 end) {
  if (g_mirror_clean.empty())
    return;
  for (const auto& [lo, hi] : g_mirror_self)
    if ((lo & ~u64(0xFFF)) <= first && end <= ((hi + 0xFFF) & ~u64(0xFFF)))
      return;
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    g_mirror_clean.erase(block);
    if (!g_mirror_cached.empty())
      MirrorUncache(block);
  }
}

void MirrorWritesCollected() {
  g_mirror_self.clear();
}

}  // namespace gpu::render
