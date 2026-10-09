/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// DELTA_GPU_MIRROR: one sparse device buffer that caches guest memory for
// the plain buffers a dispatch binds.

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
                 rhi::kBufferIndirect | rhi::kBufferVertex;
    desc.name = "guest mirror";
    if (Device().caps().sparse_buffer)
      g_mirror = Device().CreateBuffer(desc);
    BASE_LOGI("mirror", "guest mirror {:#x}+{:#x}, {:#x}+{:#x}: {}",
              kMirrorPs4Base, kMirrorLowWindow, kMirrorPs5Base, kMirrorWindow,
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
    if (caps.host_import && gpu::IsWritableRange(chunk, kMirrorImport)) {
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

// Guest bytes copied by the GPU straight out of the host pages into `dst`.
// False when a page cannot be imported or a block's bytes live in the mirror.
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

// Guest bytes copied by the GPU into `dst` at `dst_off`: out of the mirror
// where a block is the mirror's, straight out of the imported pages elsewhere,
// read-only direct memory through its second view. Host pages the device
// cannot import at all go up through an upload buffer, the one place the CPU
// touches them. False when the bytes are not readable at all.
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
  // Direct memory the guest maps read-only imports through the second view.
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
      list->CopyBuffer(dst, dst_off + at - base, g_mirror, MirrorOffset(at),
                       n);
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
      // Host memory the device cannot map at all (a read-only module page):
      // it goes up through the upload ring. No GPU write reaches such pages.
      if (!gpu::IsReadableRange(at, n))
        return false;
      // The frame's upload ring, or a buffer of its own where the ring cannot
      // grow (a host that pins little memory).
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
      !FlushCsWritesForGpu(renderer, base, bytes, "mirror"))
    return false;
  return MirrorCopyInGuest(base, bytes);
}

bool MirrorCopyInGuest(u64 base, u64 bytes) {
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  // Only the blocks the CPU may have changed: runs of them, backed and copied
  // once each. Whole blocks where they are mapped, as a block may become the
  // GPU's. A block the GPU owns, or one watched since its copy, holds the
  // newest bytes already, and was backed then.
  const u64 page = base::Max(Device().caps().sparse_page, kMirrorBlock);
  const u64 end = base + bytes;
  u64 run = 0, run_end = 0;
  bool in_run = false;
  const auto copy_run = [&] {
    in_run = false;
    const u64 lo = MirrorOffset(run) & ~(page - 1);
    const u64 hi = (MirrorOffset(run_end - 1) + page) & ~(page - 1);
    return Device().CommitSparse(g_mirror, lo, hi - lo) &&
           MirrorCopy(run, run_end - run, /*to_guest=*/false);
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

// Before the GPU writes [base, base+bytes) of guest memory: the mirror blocks
// it touches hold the guest's bytes around it, or the pages import.
static bool PrepareGuestWrite(u64 base, u64 bytes, bool* mirrored) {
  *mirrored = kMirror && Mirror() && MirrorCovers(base, bytes);
  if (CsTracksGuestWrites())
    CollectGuestWritesForSubmission();
  return *mirrored ? MirrorCopyInGuest(base, bytes)
                   : GuestImportable(base, bytes);
}

// `src` copied over guest memory at `base` on the open batch, around the
// `skip` spans: into the mirror, which owns the bytes or writes them through,
// or straight into the imported guest pages where the mirror does not reach.
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

// A dirty range written back on the GPU. A label behind it waits for no
// readback. `kept`: bytes the CPU wrote since staging were left as they are,
// so the buffer no longer holds what guest memory does.
bool GpuWriteBack(u64 base, CsRange& e, bool* kept) {
  *kept = false;
  if (!e.buf || e.imported || (base & 3))
    return false;
  // An image staged in the shader's layout is tiled again on the way out.
  // Block-compressed staging holds decoded texels, which cannot go back.
  const bool retile = e.image_staging && !e.truth;
  gcn::TextureLayout32 tiled, linear;
  const TileTable* table = nullptr;
  if (retile) {
    if (e.res.dfmt == 35 || e.res.dfmt == 40 ||
        !BuildCsImageLayouts(e.res, tiled, linear))
      return false;
    table = GetTileTable(tiled, linear,
                         /*packed=*/tiled.elem_bytes < 4 &&
                             linear.elem_bytes == tiled.elem_bytes);
    if (!table)
      return false;
  }
  const u64 n = retile         ? tiled.size
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
    // Over the guest's bytes, so padding the layout skips keeps them.
    const u64 padded = (n + 3) & ~u64(3);
    src = AcquireScratch(padded);
    if (!src || !RecordGuestRead(c, base, n, src))
      return false;
    RecordImageTiling(c, *table, src, 0, padded, e.buf,
                      (linear.size + 3) & ~u64(3), /*detile=*/false);
  }
  // Bytes the CPU wrote since staging are newer than the dispatch's.
  base::Vector<base::Pair<u64, u64>> skip;
  for (const auto* spans : {&e.cpu_writes, &e.fill_kept})
    for (const auto& [lo, hi] : *spans)
      if (base::Max(lo, base) < base::Min(hi, base + n))
        skip.emplace_back(base::Max(lo, base), base::Min(hi, base + n));
  base::Sort(skip.begin(), skip.end());
  *kept = !skip.empty();
  RecordGuestWrite(c, mirrored, base, n, src, skip);
  e.fill_kept.clear();
  // The buffer is read by the copy: not recycled before the batch is done.
  MarkPending(e);
  return true;
}

bool WriteTextureToGuest(rhi::Texture* texture,
                         rhi::TextureState& state,
                         u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear) {
  const TileTable* table =
      GetTileTable(tiled, linear, /*packed=*/tiled.elem_bytes < 4);
  if (!table || !g_frame.recording)
    return false;
  // On the frame's list, behind the draws that rendered the texture.
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
  // Over the guest's bytes, so padding the layout skips keeps them.
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
                    /*detile=*/false);
  RecordGuestWrite(c, mirrored, base, tiled.size, raw, {});
  return true;
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
  NotePendingGuestWrite(base, base + bytes);
  g_mirror_self.push_back({base, base + bytes});
  NoteGuestWrite(base, bytes);
  InvalidateTexRange(base, bytes);
}

// A fill done by the GPU: into the mirror, then written through like a
// dispatch's output, with no CPU touching the bytes.
bool MirrorFillable(u64 base, u64 bytes) {
  return kMirror && Mirror() && !(base & 3) && !(bytes & 3) &&
         MirrorCovers(base, bytes) &&
         (MirrorOwnable(base, bytes) || GuestImportable(base, bytes));
}

bool MirrorFill(u64 base, u64 bytes, u32 value) {
  if (!MirrorFillable(base, bytes))
    return false;
  // A block the fill makes the GPU's holds the guest's bytes around it.
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
  // Elsewhere the pattern is built in scratch and copied over the pages.
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

bool CanDrawIndirect() {
  return Device().caps().draw_indirect;
}

rhi::Buffer* IndirectArgsBuffer(u64 args, u64* offset) {
  *offset = MirrorOffset(args);
  return g_mirror;
}

// Indirect arguments the mirror does not hold, copied by the GPU out of guest
// memory, in queue order, into a ring the draw or dispatch reads. At most two
// frames are in flight, so a quarter of the ring per frame is never reused
// under a reader.
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
    desc.usage = rhi::kBufferIndirect | rhi::kBufferCopyDst |
                 rhi::kBufferStorage;
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

// A COND_EXEC flag a GPU write is still on its way to: copied behind it, and
// every draw until EndGuestPredicate runs only when it is non-zero.
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

void MirrorNoteGuestWrites(u64 first, u64 end, bool announced) {
  if (g_mirror_clean.empty())
    return;
  // Our own write-through, announced. A CPU write the tracker saw in the
  // same pages is not, and the mirror misses its bytes.
  if (announced)
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
