/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute/internal.h"

#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/memory/guest_direct.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/render/memory/upload_ring.h"
#include "gpu/rhi/write_journal.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"
#include "profile/profile.h"

namespace gpu::render {

constexpr u64 kMirrorImport = 2ull << 20;

rhi::Buffer* Mirror() {
  if (!g_cs.mirror_tried) {
    g_cs.mirror_tried = true;
    rhi::BufferDesc desc;
    desc.size = kMirrorSize;
    desc.sparse = true;
    desc.usage = rhi::kBufferStorage | rhi::kBufferUniform |
                 rhi::kBufferCopySrc | rhi::kBufferCopyDst |
                 rhi::kBufferAddress | rhi::kBufferIndirect |
                 rhi::kBufferVertex | rhi::kBufferIndex;
    desc.name = "guest mirror";
    if (Device().caps().sparse_buffer)
      g_cs.mirror = Device().CreateBuffer(desc);
    BASE_LOGI("mirror", "guest mirror {:#x}+{:#x}, {:#x}+{:#x}: {}",
              kMirrorPs4Base, kMirrorLowWindow, kMirrorPs5Base, kMirrorWindow,
              g_cs.mirror ? "up" : "unavailable");
  }
  return g_cs.mirror;
}

namespace {
// Ranges the guest remapped, queued by its threads for the render thread.
base::Mutex g_remapped_lock;
base::Vector<base::Pair<u64, u64>> g_remapped;
base::Atomic<bool> g_remapped_any{false};

// A cached import still pins the pages the range had when it was created: a
// remap gives the guest new pages (and an unmap's discard waits on the pin),
// so the import is dropped and the next use imports what is there now. A
// chunk an owned block still copies through keeps its import.
void RetireRemappedImports() {
  if (!g_remapped_any.exchange(false, base::memory_order_acq_rel))
    return;
  base::Vector<base::Pair<u64, u64>> remapped;
  {
    base::LockGuard<base::Mutex> lock(g_remapped_lock);
    remapped.swap(g_remapped);
  }
  for (const auto& [base, end] : remapped) {
    for (u64 chunk = base & ~(kMirrorImport - 1); chunk < end;
         chunk += kMirrorImport) {
      const auto found = g_cs.mirror_imports.find(chunk);
      if (found == g_cs.mirror_imports.end())
        continue;
      bool owned = false;
      for (u64 block = chunk; block < chunk + kMirrorImport && !owned;
           block += kMirrorBlock)
        owned = InMirror(block) && OwnAt(block).state.load() != 0;
      if (owned)
        continue;
      if (found->second)
        g_cs.frame_retired.push_back(found->second);
      g_cs.mirror_imports.erase(found);
    }
  }
}
}  // namespace

void MirrorNoteRemapped(u64 base, u64 bytes) {
  if (!bytes)
    return;
  base::LockGuard<base::Mutex> lock(g_remapped_lock);
  if (g_remapped.size() < 65536)
    g_remapped.push_back({base, base + bytes});
  g_remapped_any.store(true, base::memory_order_release);
}

rhi::Buffer* MirrorImport(u64 address, u64* import_base) {
  RetireRemappedImports();
  u64 offset = 0, bytes = 0;
  if (auto* direct = DirectImportAt(address, &offset, &bytes);
      direct && direct->desc().host_pointer &&
      bytes == direct->desc().size - offset) {
    *import_base = address - offset;
    return direct;
  }
  const u64 chunk = address & ~(kMirrorImport - 1);
  auto found = g_cs.mirror_imports.find(chunk);
  if (found == g_cs.mirror_imports.end()) {
    rhi::Buffer* buf = nullptr;
    const rhi::Caps& caps = Device().caps();
    if (caps.host_import && gpu::IsWritableRange(chunk, kMirrorImport)) {
      DELTA_ZONE("mirror.import");
      GuestWriteTracker().Release(chunk, kMirrorImport);
      rhi::BufferDesc desc;
      desc.size = kMirrorImport;
      desc.usage =
          rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc;
      desc.host_pointer = reinterpret_cast<void*>(chunk);
      buf = Device().CreateBuffer(desc);
    }
    found = g_cs.mirror_imports.insert_or_assign(chunk, buf).first;
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
    list = g_cs.list;
  list->ProfileBegin(to_guest ? 3 : 16);
  list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  for (const auto& [import, span] : pieces) {
    u64 import_base = 0;
    MirrorImport(span.first, &import_base);
    const u64 n = span.second - span.first;
    if (to_guest)
      list->CopyBuffer(import, span.first - import_base, g_cs.mirror,
                       MirrorOffset(span.first), n);
    else
      list->CopyBuffer(g_cs.mirror, MirrorOffset(span.first), import,
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
  // kHole: an unmapped block of a partly resident (sparse) resource.
  enum Source { kMirrorSource, kImport, kDirect, kUpload, kHole };
  const auto source = [&](u64 at) {
    if (owned(at))
      return kMirrorSource;
    if (!gpu::IsReadableRangeCached(at, block_end(at) - at))
      return kHole;
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
    if (from == kHole) {
      const u64 lo = (dst_off + at - base + 3) & ~u64(3);
      const u64 hi = (dst_off + end - base) & ~u64(3);
      if (hi > lo)
        list->FillBuffer(dst, lo, hi - lo, 0);
    } else if (from == kMirrorSource) {
      list->CopyBuffer(dst, dst_off + at - base, g_cs.mirror, MirrorOffset(at),
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
        g_cs.frame_retired.push_back(own);
        upload = {own, 0, static_cast<u8*>(own->mapped())};
      }
      std::memcpy(upload.map, reinterpret_cast<const void*>(at), n);
      list->CopyBuffer(dst, dst_off + at - base, upload.buffer, upload.offset,
                       n);
    }
    at = end;
  }
  if (g_cs.batch_open && !g_cs.batches[g_cs.batch_cur].in_frame)
    g_cs.chunk_needs_batch = true;
  return true;
}


DELTA_OPTION(bool, kMirrorOwn, "DELTA_GPU_MIRROR_OWN", true);
DELTA_OPTION(bool, kMirrorOwnSplit, "DELTA_GPU_MIRROR_OWN_SPLIT", true);
constexpr u32 kOwnMaxFaults = 4;
u64 g_capped_block[512];
int g_capped_frame[512];
base::Atomic<u32> g_capped_n{0};

OwnBlock& OwnAt(u64 block) {
  // Fault handlers on any thread can get here first; exactly one table wins.
  OwnBlock* own = __atomic_load_n(&g_cs.own, __ATOMIC_ACQUIRE);
  if (!own) {
    void* p =
        mmap(nullptr, kOwnBlocks * sizeof(OwnBlock), PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (__atomic_compare_exchange_n(&g_cs.own, &own, static_cast<OwnBlock*>(p),
                                    false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      own = static_cast<OwnBlock*>(p);
    else
      munmap(p, kOwnBlocks * sizeof(OwnBlock));
  }
  return own[MirrorOffset(block) / kMirrorBlock];
}

void MirrorReportCapped() {
  static u32 printed = 0;
  const u32 n =
      base::Min<u32>(g_capped_n.load(base::memory_order_relaxed), 512);
  for (; printed < n; printed++)
    BASE_LOGI("csmirror", "block {:#x} unownable after {} faults at frame {}",
              (unsigned long)g_capped_block[printed], kOwnMaxFaults,
              g_capped_frame[printed]);
}

bool MirrorOwned(u64 block) {
  return g_cs.own && InMirror(block) &&
         OwnAt(block).state.load(base::memory_order_acquire) != 0;
}

bool InMirror(u64 address) {
  return MirrorOffset(address) != kNotMirrored;
}

namespace {
void NoteProt(OwnBlock& o, u8 op) {
  o.last_prot.store(op, base::memory_order_relaxed);
  o.last_prot_frame.store(g_frame.num, base::memory_order_relaxed);
}

// Signal-safe: the kernel reports EFAULT for a page this process can't read.
bool PageReadable(u64 address) {
  u8 byte;
  iovec local{&byte, 1};
  iovec remote{reinterpret_cast<void*>(address), 1};
  return ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) == 1;
}

// Two threads fault on one protected block: the second can reach its handler
// after the first reopened it, find nothing to resolve, and must just retry.
// Bounded, so a fault the mirror can't explain still dies.
bool MirrorFaultRaced(u64 page) {
  static thread_local u64 last = 0;
  static thread_local u32 repeats = 0;
  repeats = page == last ? repeats + 1 : 0;
  last = page;
  if (!InMirror(page))
    return repeats < 4096 && PageReadable(page);
  const u64 block = page & ~(kMirrorBlock - 1);
  // Reopened, or already owned again by the next protect: either way the
  // retried access reaches a resolvable state. A protect takes Busy before
  // its mprotect, so the state is read after the page: an unreadable page
  // with the block neither Busy nor Protected is reopened by then.
  const bool readable = PageReadable(page);
  const u64 s = OwnAt(block).state.load(base::memory_order_acquire);
  if (repeats < 4096 &&
      (readable || (s & (kOwnBusy | kOwnProtected)) || PageReadable(page)))
    return true;
  // About to become a crash: say what the mirror believed about the block.
  bool queued = false;
  for (u64 b : g_cs.own_unsubmitted)
    queued = queued || b == block;
  BASE_LOGW("csmirror",
            "unresolved fault in block {:#x}: state={:#x} (protected={} "
            "pending={} busy={} submission={}) queued={} protecting={}/{} "
            "resolves={} retries={} last-prot={} at frame {} (now {})",
            (unsigned long)block, (unsigned long)s, (s & kOwnProtected) != 0,
            (s & kOwnPending) != 0, (s & kOwnBusy) != 0,
            (unsigned long)(s & kOwnSubmission), queued,
            g_cs.own_protecting.size(), g_cs.own_unsubmitted.size(),
            g_mirror_resolves.load(base::memory_order_relaxed), repeats,
            OwnAt(block).last_prot.load() == 1   ? "protect"
            : OwnAt(block).last_prot.load() == 2 ? "resolve"
            : OwnAt(block).last_prot.load() == 3 ? "forget"
                                                 : "none",
            OwnAt(block).last_prot_frame.load(), g_frame.num);
  return false;
}
}  // namespace

bool MirrorResolve(u64 first, u64 end) {
  if (!g_cs.own)
    return false;
  bool resolved = false;
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    if (!InMirror(block))
      continue;
    OwnBlock& o = OwnAt(block);
    for (;;) {
      u64 s = o.state.load(base::memory_order_acquire);
      // MirrorProtectSubmitted holds Busy across its mprotect and sets
      // Protected only after, so a fault in that window must wait for it.
      if (s & kOwnBusy)
        continue;
      if (!(s & kOwnProtected))
        break;
      if (!o.state.compare_exchange_weak(s, s | kOwnBusy,
                                         base::memory_order_acq_rel))
        continue;
      for (u32 i = 0; i < o.spans; i++)
        Device().CopyAfter(g_cs.mirror, MirrorOffset(block) + o.lo[i], o.import,
                           o.import_offset + o.lo[i], o.hi[i] - o.lo[i],
                           s & kOwnSubmission);
      ::mprotect(reinterpret_cast<void*>(block), kMirrorBlock,
                 PROT_READ | PROT_WRITE | PROT_EXEC);
      NoteProt(o, 2);
      WriteTracker::OpenedByOwner(block, block + kMirrorBlock);
      if (!(o.state.load(base::memory_order_acquire) & kOwnPending)) {
        DirectRestore(block);
        o.spans = 0;
      }
      o.cpu_dirty.store(1, base::memory_order_relaxed);
      if (o.faults.fetch_add(1, base::memory_order_relaxed) + 1 ==
          kOwnMaxFaults) {
        const u32 i = g_capped_n.fetch_add(1, base::memory_order_relaxed);
        if (i < 512) {
          g_capped_block[i] = block;
          g_capped_frame[i] = g_frame.num;
        }
      }
      g_mirror_resolves.fetch_add(1, base::memory_order_relaxed);
      // MirrorOwn may have queued the block since `s` was read; dropping its
      // pending bit would queue it twice and spin MirrorProtectSubmitted.
      if (o.state.fetch_and(kOwnPending, base::memory_order_acq_rel) &
          kOwnPending)
        o.reopened.store(1, base::memory_order_release);
      resolved = true;
      break;
    }
  }
  return resolved;
}

void MirrorForget(u64 base, u64 bytes) {
  if (!g_cs.own || !bytes)
    return;
  const u64 end = base + bytes;
  for (u64 block = base & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    if (!InMirror(block))
      continue;
    OwnBlock& o = OwnAt(block);
    u64 s;
    for (;;) {
      s = o.state.load(base::memory_order_acquire);
      if (!s)
        break;
      if (!(s & kOwnBusy) && o.state.compare_exchange_weak(
                                 s, s | kOwnBusy, base::memory_order_acq_rel))
        break;
    }
    if (!s)
      continue;
    // Offsets of the remapped part within the block; the rest is untouched.
    const u32 cut_lo = static_cast<u32>(base::Max(block, base) - block);
    const u32 cut_hi =
        static_cast<u32>(base::Min(block + kMirrorBlock, end) - block);
    if (s & kOwnProtected) {
      const auto copy_back = [&](u32 lo, u32 hi) {
        if (lo < hi)
          Device().CopyAfter(g_cs.mirror, MirrorOffset(block) + lo, o.import,
                             o.import_offset + lo, hi - lo, s & kOwnSubmission);
      };
      for (u32 i = 0; i < o.spans; i++) {
        copy_back(o.lo[i], base::Min(o.hi[i], cut_lo));
        copy_back(base::Max(o.lo[i], cut_hi), o.hi[i]);
      }
      if (cut_lo)
        ::mprotect(reinterpret_cast<void*>(block), cut_lo,
                   PROT_READ | PROT_WRITE | PROT_EXEC);
      if (cut_hi < kMirrorBlock)
        ::mprotect(reinterpret_cast<void*>(block + cut_hi),
                   kMirrorBlock - cut_hi, PROT_READ | PROT_WRITE | PROT_EXEC);
    }
    NoteProt(o, 3);
    DirectRestore(block);
    o.state.store(0, base::memory_order_release);
  }
}

bool MirrorOwnsAny(u64 first, u64 end) {
  if (!g_cs.own || first >= end ||
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

namespace {
base::Pair<u32, u32> SpanIn(u64 block, u64 base, u64 end) {
  return {static_cast<u32>(base::Max(block, base) - block),
          static_cast<u32>(base::Min(end, block + kMirrorBlock) - block)};
}

bool SpanFits(const OwnBlock& o, u32 lo, u32 hi) {
  if (!o.state.load(base::memory_order_acquire) || o.spans < kOwnSpans)
    return true;
  for (u32 i = 0; i < o.spans; i++)
    if (lo <= o.hi[i] && o.lo[i] <= hi)
      return true;
  return false;
}

void AddSpan(OwnBlock& o, u32 lo, u32 hi) {
  u32 n = 0;
  for (u32 i = 0; i < o.spans; i++) {
    if (lo <= o.hi[i] && o.lo[i] <= hi) {
      lo = base::Min(lo, o.lo[i]);
      hi = base::Max(hi, o.hi[i]);
      continue;
    }
    o.lo[n] = o.lo[i];
    o.hi[n++] = o.hi[i];
  }
  o.lo[n] = lo;
  o.hi[n++] = hi;
  o.spans = n;
}
}  // namespace

// IsReadableRangeCached over a whole block walks its 16 pages; the answer
// only changes with the memory generation, so it is kept per block.
bool BlockReadable(u64 block) {
  base::Atomic<u64>& memo = OwnAt(block).readable;
  const u64 gen = gpu::ReadableGeneration().load(base::memory_order_acquire);
  const u64 seen = memo.load(base::memory_order_relaxed);
  if (seen >> 1 == gen)
    return seen & 1;
  const bool readable = gpu::IsReadableRangeCached(block, kMirrorBlock);
  memo.store(gen << 1 | readable, base::memory_order_relaxed);
  return readable;
}

bool MirrorOwnable(u64 base, u64 bytes) {
  if (!kMirrorOwn || !gcn::g_direct_table || !Device().caps().copy_after ||
      !g_cs.mirror || !g_cs.mirror->address())
    return false;
  for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
       block += kMirrorBlock) {
    const OwnBlock& o = OwnAt(block);
    const auto [lo, hi] = SpanIn(block, base, base + bytes);
    if (!BlockReadable(block) ||
        o.faults.load(base::memory_order_relaxed) >= kOwnMaxFaults ||
        !SpanFits(o, lo, hi))
      return false;
  }
  return true;
}

rhi::Buffer* MirrorImport(u64 address, u64* import_base);

void MirrorOwn(u64 base, u64 bytes) {
  static bool hooked = false;
  if (!hooked) {
    hooked = true;
    WriteTracker::SetOwner(&MirrorResolve, &MirrorOwnsAny, &MirrorFaultRaced);
  }
  const u64 end = base + bytes;
  for (u64 block = base & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    OwnBlock& o = OwnAt(block);
    const auto [lo, hi] = SpanIn(block, base, end);
    if (!o.state.load(base::memory_order_acquire)) {
      u64 import_base = 0;
      o.import = MirrorImport(block, &import_base);
      o.import_offset = block - import_base;
      o.spans = 0;
      DirectRedirect(block, g_cs.mirror->address() + MirrorOffset(block));
      g_cs.mirror_was_owned.insert(block);
    }
    AddSpan(o, lo, hi);
    // Publish it with the latest list that writes it.
    o.owner_list = g_cs.list;
    if (!(o.state.fetch_or(kOwnPending, base::memory_order_acq_rel) &
          kOwnPending))
      g_cs.own_unsubmitted.push_back(block);
  }
}

void MirrorProtectBegin(rhi::CommandList* list) {
  base::Vector<u64>& blocks = g_cs.own_protecting;
  if (!blocks.empty())
    return;
  // A block MirrorForget dropped is no longer pending; one re-owned since
  // may be queued twice.
  for (u64 block : g_cs.own_unsubmitted) {
    const OwnBlock& o = OwnAt(block);
    if (o.owner_list == list &&
        (o.state.load(base::memory_order_acquire) & kOwnPending))
      blocks.push_back(block);
  }
  if (blocks.empty())
    return;
  g_cs.own_protect_list = list;
  DELTA_ZONE("mirror.protect");
  base::Sort(blocks.begin(), blocks.end());
  blocks.erase(base::Unique(blocks.begin(), blocks.end()), blocks.end());
  base::Vector<u64> opened;
  for (u64 block : blocks) {
    OwnBlock& o = OwnAt(block);
    u64 s = o.state.load(base::memory_order_acquire);
    while ((s & kOwnBusy) || !o.state.compare_exchange_weak(
                                 s, kOwnBusy, base::memory_order_acq_rel))
      s = o.state.load(base::memory_order_acquire);
    if (!(s & kOwnProtected))
      opened.push_back(block);
  }
  for (size_t i = 0; i < blocks.size();) {
    size_t j = i + 1;
    while (j < blocks.size() && blocks[j] == blocks[j - 1] + kMirrorBlock)
      j++;
    ::mprotect(reinterpret_cast<void*>(blocks[i]), (j - i) * kMirrorBlock,
               PROT_NONE);
    for (size_t k = i; k < j; k++)
      NoteProt(OwnAt(blocks[k]), 1);
    i = j;
  }
  // Shut again, so guest memory is stable: give the mirror the CPU writes made
  // while it was open, ahead of the queued dispatches in submission order. A
  // block that was open until now counts too: its copy-in can have been
  // skipped as fresh before CPU writes since were collected, and a resolve
  // would copy those stale words back over the guest's (SotC's heap).
  rhi::CommandList* refresh = nullptr;
  for (u64 block : blocks) {
    OwnBlock& o = OwnAt(block);
    const bool reopened = o.reopened.exchange(0, base::memory_order_acq_rel);
    if ((!reopened && !opened.Contains(block)) || !o.import)
      continue;
    if (!refresh && !(refresh = BeginImmediate()))
      break;
    refresh->CopyBuffer(g_cs.mirror, MirrorOffset(block), o.import,
                        o.import_offset, kMirrorBlock);
  }
  if (refresh)
    EndImmediate(refresh, "mirror.refresh");
}

void MirrorProtectSubmitted(u64 submission) {
  if (g_cs.own_protecting.empty())
    return;
  for (u64 block : g_cs.own_protecting)
    OwnAt(block).state.store(submission | kOwnProtected,
                             base::memory_order_release);
  auto& queued = g_cs.own_unsubmitted;
  size_t kept = 0;
  for (u64 block : queued)
    if (OwnAt(block).owner_list != g_cs.own_protect_list)
      queued[kept++] = block;
  queued.resize(kept);
  g_cs.own_protecting.clear();
  g_cs.own_protect_list = nullptr;
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
  if (!g_cs.batch_open)
    return false;
  g_cs.list->Barrier(rhi::kAccessVertexRead | rhi::kAccessIndexRead |
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

DELTA_OPTION(bool, kMirrorBypass, "DELTA_GPU_MIRROR_BYPASS", true);

rhi::Buffer* MirrorBypass(Renderer& renderer,
                          u64 base,
                          u64 bytes,
                          u64* offset) {
  if (!kMirrorBypass || !Device().caps().unified_memory)
    return nullptr;
  if (CsRangeDirtyOverlapping(base, bytes) &&
      !FlushCsWritesForGpu(renderer, base, bytes, "mirror"))
    return nullptr;
  for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
       block += kMirrorBlock)
    if (MirrorOwned(block))
      return nullptr;
  if (MirrorHoldsNewest(base, bytes))
    return nullptr;
  return DirectBindAt(base, bytes, offset);
}

bool MirrorCopyInGuest(u64 base, u64 bytes) {
  if (!OrderMirrorAfterFrameRead())
    return false;
  CsBatchBeginImpl();
  if (!g_cs.batch_open)
    return false;
  const u64 page = base::Max(Device().caps().sparse_page, kMirrorBlock);
  const u64 end = base + bytes;
  u64 run = 0, run_end = 0;
  bool in_run = false;
  const auto copy_run = [&] {
    in_run = false;
    const u64 lo = MirrorOffset(run) & ~(page - 1);
    const u64 hi = (MirrorOffset(run_end - 1) + page) & ~(page - 1);
    g_cs_stats.mirror_in_bytes += run_end - run;
    g_cs_stats.mirror_in_n++;
    if (CsSyncReporting()) {
      u64& e = g_cs_stats.mirror_in_union[run];
      e = base::Max(e, run_end);
      auto& t = g_cs_stats.mirror_in_top[run];
      t.first += run_end - run;
      t.second++;
    }
    return Device().CommitSparse(g_cs.mirror, lo, hi - lo) &&
           MirrorCopy(run, run_end - run, false);
  };
  for (u64 block = base & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    bool fresh = MirrorOwned(block);
    if (!fresh) {
      const bool cpu_dirty = OwnAt(block).cpu_dirty.exchange(0);
      fresh = g_cs.mirror_clean.count(block) && !cpu_dirty;
    }
    if (fresh) {
      if (in_run && !copy_run())
        return false;
      continue;
    }
    const bool whole = BlockReadable(block);
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
      g_cs.mirror_clean.insert(block);
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

namespace {
struct WordMergePipe {
  rhi::BindGroupLayout* set = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
  bool tried = false;
};

const WordMergePipe* WordMerge() {
  static WordMergePipe p;
  if (!p.tried) {
    p.tried = true;
#ifdef DELTA_HAVE_SPIRV_BACKEND
    const auto code = gcn::BuildWordMergeShader();
#else
    const base::Vector<u32> code;
#endif
    rhi::BindGroupLayoutDesc set;
    for (u32 i = 0; i < 3; i++)
      set.bindings.push_back(
          {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
    set.push = true;
    p.set = code.empty() ? nullptr : Device().CreateBindGroupLayout(set);
    rhi::PipelineLayoutDesc layout;
    layout.groups = {p.set};
    layout.push_constant_bytes = 16;
    layout.push_constant_stages = rhi::kStageCompute;
    p.layout = p.set ? Device().CreatePipelineLayout(layout) : nullptr;
    rhi::ComputePipelineDesc pipe;
    pipe.layout = p.layout;
    pipe.code = rhi::Code(code);
    p.pipe = p.layout ? Device().CreateComputePipeline(pipe) : nullptr;
  }
  return p.pipe ? &p : nullptr;
}

struct MergeOperand {
  rhi::Buffer* buffer;
  u64 offset;
};

// Records the merge of `bytes` of `result` against `shadow` into `dst`.
void RecordMergePass(rhi::CommandList* c,
                     const WordMergePipe& p,
                     MergeOperand result,
                     MergeOperand shadow,
                     MergeOperand dst,
                     u64 bytes) {
  const u64 align = Device().caps().storage_offset_alignment;
  const u64 max_words =
      base::Min<u64>(u64(65535) * gcn::kWordMergeGroup,
                     (Device().caps().max_storage_buffer_range - align) / 4);
  c->SetPipeline(p.pipe);
  for (u64 done = 0; done < bytes;) {
    const u64 words = base::Min((bytes - done) / 4, max_words);
    const MergeOperand ops[3] = {result, shadow, dst};
    rhi::BindingWrite writes[3];
    u32 params[4];
    for (u32 i = 0; i < 3; i++) {
      const u64 offset = ops[i].offset + done;
      const u64 window = offset & ~(align - 1);
      writes[i].binding = i;
      writes[i].buffer = ops[i].buffer;
      writes[i].offset = window;
      writes[i].range = offset + words * 4 - window;
      params[i] = static_cast<u32>((offset - window) / 4);
    }
    params[3] = static_cast<u32>(words);
    c->PushBindGroup(0, p.set, writes, 3);
    c->SetPushConstants(0, sizeof(params), params);
    c->Dispatch(static_cast<u32>((words + gcn::kWordMergeGroup - 1) /
                                 gcn::kWordMergeGroup),
                1, 1);
    rhi::write_journal::Record(dst.buffer, dst.offset + done, words * 4, 'm');
    done += words * 4;
  }
}

// Merges mirror [run, to) into guest memory against `shadow`.
bool RecordWordMerge(u64 run, u64 to, const MirrorShadow& shadow) {
  const WordMergePipe* p = WordMerge();
  if (!p || ((run | to) & 3))
    return false;
  base::Vector<base::Pair<rhi::Buffer*, u64>> imports;
  for (u64 at = run; at < to;) {
    u64 import_base = 0;
    rhi::Buffer* import = MirrorImport(at, &import_base);
    if (!import)
      return false;
    imports.push_back({import, import_base});
    at = base::Min(to, import_base + import->desc().size);
  }
  rhi::CommandList* const c = g_cs.list;
  c->ProfileBegin(3);
  c->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessComputeRW);
  u64 at = run;
  for (const auto& [import, import_base] : imports) {
    const u64 end = base::Min(to, import_base + import->desc().size);
    RecordMergePass(c, *p, {g_cs.mirror, MirrorOffset(at)},
                    {shadow.buffer, shadow.offset + (at - shadow.base)},
                    {import, at - import_base}, end - at);
    at = end;
  }
  c->Barrier(rhi::kAccessComputeWrite, kAccessAll | rhi::kAccessHostRead);
  c->ProfileEnd();
  return true;
}
}  // namespace

// Inlined so the mirror records each caller as the writer.
[[gnu::always_inline]] static inline void RecordGuestWrite(
    rhi::CommandList* c,
                             bool mirrored,
                             u64 base,
                             u64 n,
                             rhi::Buffer* src,
                             const base::Vector<base::Pair<u64, u64>>& skip,
                             u64 src_off = 0,
                             const MergeOperand* shadow = nullptr) {
  const WordMergePipe* merge = shadow ? WordMerge() : nullptr;
  c->Barrier(kAccessComputeRW | kAccessCopyRW,
             kAccessComputeRW | kAccessCopyRW);
  const auto copy = [&](u64 lo, u64 hi) {
    if (merge && !((lo | hi) & 3)) {
      const MergeOperand result{src, src_off + lo - base};
      const MergeOperand before{shadow->buffer, shadow->offset + lo - base};
      if (mirrored) {
        RecordMergePass(c, *merge, result, before,
                        {g_cs.mirror, MirrorOffset(lo)}, hi - lo);
        return;
      }
      for (u64 at = lo; at < hi;) {
        u64 import_base = 0;
        rhi::Buffer* import = MirrorImport(at, &import_base);
        const u64 to = base::Min(hi, import_base + import->desc().size);
        RecordMergePass(c, *merge, {src, result.offset + at - lo},
                        {before.buffer, before.offset + at - lo},
                        {import, at - import_base}, to - at);
        at = to;
      }
      return;
    }
    if (mirrored) {
      c->CopyBuffer(g_cs.mirror, MirrorOffset(lo), src, src_off + lo - base,
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
  c->Barrier(rhi::kAccessCopyWrite | rhi::kAccessComputeWrite,
             kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostRead);
  if (mirrored) {
    // Publish from the mirror against the same staged shadow, or every word
    // the dispatch left alone goes out as the mirror's own copy of the guest,
    // which can predate the CPU's stores (SotC's freed heap chunks).
    const MirrorShadow staged{merge ? shadow->buffer : nullptr, base,
                              merge ? shadow->offset : 0, n};
    MirrorWriteBack(base, n, true, merge ? &staged : nullptr);
  } else {
    NotePendingGuestWrite(base, base + n);
    NoteGuestWrite(base, n);
  }
}

DELTA_OPTION(bool, kStageShadow, "DELTA_GPU_STAGE_SHADOW", true);

void CaptureStageShadow(CsRange& e, u64 bytes) {
  bytes &= ~u64(3);
  if (!kStageShadow || !bytes || !e.buf || e.imported || !WordMerge())
    return;
  if (!e.shadow || e.shadow->desc().size < bytes) {
    if (e.shadow)
      g_cs.frame_retired.push_back(e.shadow);
    e.shadow = CreateCsBuffer((bytes + 0xFFFF) & ~u64(0xFFFF),
                              rhi::kBufferStorage | rhi::kBufferCopyDst,
                              rhi::MemoryKind::kDevice);
    if (!e.shadow)
      return;
  }
  rhi::CommandList* const c = g_cs.list;
  c->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
  c->CopyBuffer(e.shadow, 0, e.buf, 0, bytes);
  c->Barrier(kAccessCopyRW, kAccessComputeRW | kAccessCopyRW);
  e.shadow_bytes = bytes;
  e.shadow_stamp = e.staged_stamp;
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
        tiled.elem_bytes < 4 && linear.elem_bytes == tiled.elem_bytes, false,
        e.res.dfmt == 7);
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
  if (!g_cs.batch_open)
    return false;
  rhi::CommandList* const c = g_cs.list;
  rhi::Buffer* src = e.buf;
  const u64 staged_bytes = retile ? linear.size & ~u64(3) : n & ~u64(3);
  const bool shadowed = e.shadow && e.staged_stamp &&
                        e.shadow_stamp == e.staged_stamp &&
                        e.shadow_bytes >= staged_bytes;
  MergeOperand shadow{e.shadow, 0};
  if (retile) {
    // Both sides are tiled over the same guest read, so a texel the dispatches
    // left alone, and every byte between texels, compares equal and keeps the
    // CPU's word: the title reuses an image's memory for its own heap.
    const u64 padded = (n + 3) & ~u64(3);
    const u64 linear_bytes = (linear.size + 3) & ~u64(3);
    src = AcquireScratch(padded);
    if (!src || !RecordGuestRead(c, base, n, src))
      return false;
    RecordImageTiling(c, *table, src, 0, padded, e.buf, linear_bytes, false);
    shadow.buffer = nullptr;
    if (shadowed) {
      rhi::Buffer* before = AcquireScratch(padded);
      if (before && RecordGuestRead(c, base, n, before)) {
        RecordImageTiling(c, *table, before, 0, padded, e.shadow, linear_bytes,
                          false);
        shadow.buffer = before;
      }
    }
  }
  base::Vector<base::Pair<u64, u64>> skip;
  for (const auto* spans : {&e.cpu_writes, &e.fill_kept})
    for (const auto& [lo, hi] : *spans)
      if (base::Max(lo, base) < base::Min(hi, base + n))
        skip.emplace_back(base::Max(lo, base), base::Min(hi, base + n));
  base::Sort(skip.begin(), skip.end());
  *kept = !skip.empty();
  RecordGuestWrite(c, mirrored, base, n, src, skip, 0,
                   shadowed && shadow.buffer ? &shadow : nullptr);
  e.fill_kept.clear();
  MarkPending(e);
  return true;
}

bool WriteTextureToGuest(rhi::Texture* texture,
                         rhi::TextureState& state,
                         u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear,
                         u8 aspect,
                         bool repack111110) {
  const TileTable* table = GetTileTable(tiled, linear, tiled.elem_bytes < 4,
                                        false, repack111110);
  if (!table || !g_frame.recording)
    return false;
  CsBatchBeginImpl();
  if (g_cs.batch_open && !g_cs.batches[g_cs.batch_cur].in_frame) {
    if (!CsBatchSubmit())
      return false;
    CsBatchBeginImpl();
  }
  if (!g_cs.batch_open || !g_cs.batches[g_cs.batch_cur].in_frame)
    return false;
  g_cs.scratch_owner++;
  const u64 tiled_bytes = (tiled.size + 3) & ~u64(3);
  const u64 linear_bytes = (linear.size + 3) & ~u64(3);
  rhi::Buffer* raw = AcquireScratch(tiled_bytes);
  rhi::Buffer* pixels = raw ? AcquireScratch(linear_bytes) : nullptr;
  bool mirrored = false;
  if (!pixels || !PrepareGuestWrite(base, tiled.size, &mirrored))
    return false;
  rhi::CommandList* const c = g_cs.list;
  if (!RecordGuestRead(c, base, tiled.size, raw))
    return false;
  const auto& level = linear.mips[0];
  rhi::BufferTextureCopy copy;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  copy.region.width = level.width;
  copy.region.height = level.height;
  copy.region.aspect = aspect;
  const rhi::TextureState old_state = state;
  TransitionImage(c, texture, state, rhi::TextureState::kCopySrc, aspect);
  c->CopyTextureToBuffer(pixels, texture, &copy, 1);
  TransitionImage(c, texture, state, old_state, aspect);
  RecordImageTiling(c, *table, raw, 0, tiled_bytes, pixels, linear_bytes,
                    false);
  RecordGuestWrite(c, mirrored, base, tiled.size, raw, {});
  return true;
}

const MirrorWrite* MirrorWritten(u64 base, u64 bytes) {
  auto it = g_cs.mirror_writes.upper_bound(base);
  if (it == g_cs.mirror_writes.begin())
    return nullptr;
  --it;
  return base + bytes <= it->second.end ? &it->second : nullptr;
}

namespace {
void RecordMirrorWrite(u64 first, u64 bytes, const void* site) {
  u64 end = first + bytes;
  auto it = g_cs.mirror_writes.upper_bound(first);
  if (it != g_cs.mirror_writes.begin()) {
    --it;
    if (end <= it->second.end) {
      it->second.seq = ++g_cs.mirror_seq;
      it->second.site = site;
      return;
    }
    if (it->second.end < first)
      ++it;
  }
  while (it != g_cs.mirror_writes.end() && it->first <= end) {
    first = base::Min(first, it->first);
    end = base::Max(end, it->second.end);
    it = g_cs.mirror_writes.erase(it);
  }
  g_cs.mirror_writes[first] = {end, ++g_cs.mirror_seq, site};
}
}  // namespace

bool CaptureMirrorShadows(const u64* bases,
                          const u64* sizes,
                          u32 count,
                          MirrorShadow* out) {
  static rhi::Buffer* pool = nullptr;
  static const guest::SessionReset reset_pool([] {
    if (pool)
      Device().Destroy(pool);
    pool = nullptr;
  });
  if (!WordMerge())
    return false;
  const u64 align = Device().caps().storage_offset_alignment;
  u64 total = 0;
  for (u32 i = 0; i < count; i++)
    total += (sizes[i] + align - 1) & ~(align - 1);
  if (!total)
    return true;
  if (!pool || pool->desc().size < total) {
    if (pool)
      g_cs.frame_retired.push_back(pool);
    pool = CreateCsBuffer((total + 0xFFFFF) & ~u64(0xFFFFF),
                          rhi::kBufferStorage | rhi::kBufferCopyDst,
                          rhi::MemoryKind::kDevice);
    if (!pool)
      return false;
  }
  rhi::CommandList* const c = g_cs.list;
  c->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
  u64 offset = 0;
  for (u32 i = 0; i < count; i++) {
    out[i] = {pool, bases[i], offset, sizes[i]};
    const u64 end = bases[i] + sizes[i];
    for (u64 at = bases[i]; at < end;) {
      const bool owned = MirrorOwned(at & ~(kMirrorBlock - 1));
      u64 to = at;
      while (to < end && MirrorOwned(to & ~(kMirrorBlock - 1)) == owned)
        to = base::Min(end, (to & ~(kMirrorBlock - 1)) + kMirrorBlock);
      if (!owned)
        c->CopyBuffer(pool, offset + (at - bases[i]), g_cs.mirror,
                      MirrorOffset(at), to - at);
      at = to;
    }
    offset += (sizes[i] + align - 1) & ~(align - 1);
  }
  c->Barrier(kAccessCopyRW, kAccessComputeRW);
  return true;
}

void MirrorWriteBack(u64 base,
                     u64 bytes,
                     bool may_own,
                     const MirrorShadow* shadow) {
  const void* const site = __builtin_return_address(0);
  RecordMirrorWrite(base, bytes, site);
  g_cs.mirror_self.push_back({base, base + bytes});
  const u64 end = base + bytes;
  may_own = may_own && (kMirrorOwnSplit || MirrorOwnable(base, bytes));
  const auto publish = [&](u64 run, u64 to) {
    if (!shadow || !shadow->buffer)
      return MirrorCopy(run, to - run, true);
    for (u64 at = run; at < to;) {
      const bool owned = MirrorOwned(at & ~(kMirrorBlock - 1));
      u64 end = at;
      while (end < to && MirrorOwned(end & ~(kMirrorBlock - 1)) == owned)
        end = base::Min(to, (end & ~(kMirrorBlock - 1)) + kMirrorBlock);
      if (owned ? !MirrorCopy(at, end - at, true)
                : !RecordWordMerge(at, end, *shadow) &&
                      !MirrorCopy(at, end - at, true))
        return false;
      at = end;
    }
    return true;
  };
  const auto finish = [&](u64 run, u64 to, bool owned) {
    if (owned) {
      for (u64 at = run; at < to;) {
        const u64 end =
            base::Min(to, (at & ~(kMirrorBlock - 1)) + kMirrorBlock);
        OwnBlock& o = OwnAt(at & ~(kMirrorBlock - 1));
        const bool merged = shadow && shadow->buffer &&
                            !MirrorOwned(at & ~(kMirrorBlock - 1)) &&
                            RecordWordMerge(at, end, *shadow);
        o.owned_by = site;
        o.owned_merged = merged;
        at = end;
      }
      MirrorOwn(run, to - run);
      g_cs_stats.mirror_own_bytes += to - run;
      return;
    }
    g_cs_stats.mirror_wb_bytes += to - run;
    g_cs_stats.mirror_wb_n++;
    if (CsSyncReporting()) {
      u64& e = g_cs_stats.mirror_wb_union[run];
      e = base::Max(e, to);
      auto& t = g_cs_stats.mirror_wb_top[run];
      t.first += to - run;
      t.second++;
    }
    if (publish(run, to))
      NotePendingGuestWrite(run, to);
  };
  u64 run = base;
  bool run_owned = false;
  for (u64 at = base; at < end;) {
    const u64 to = base::Min(end, (at & ~(kMirrorBlock - 1)) + kMirrorBlock);
    // Only a block the write covers whole is the GPU's alone. A partial one
    // shares its pages with CPU data (SotC keeps its heap tree next to its
    // compute buffers), and a later resolve would copy the span's stale words
    // back over the CPU's: publish it now, merged, instead.
    const bool whole = to - at == kMirrorBlock ||
                       MirrorOwned(at & ~(kMirrorBlock - 1));
    const bool owned = may_own && whole && MirrorOwnable(at, to - at);
    if (at != base && owned != run_owned) {
      finish(run, at, run_owned);
      run = at;
    }
    run_owned = owned;
    at = to;
  }
  finish(run, end, run_owned);
  NoteGuestWrite(base, bytes);
  InvalidateTexRange(base, bytes);
}

void MirrorBypassWritten(u64 base, u64 bytes) {
  // A new revision readers must take from guest memory: the mirror's copy of
  // these blocks is stale until the next copy-in.
  RecordMirrorWrite(base, bytes, __builtin_return_address(0));
  MirrorNoteGuestWrites(base, base + bytes, false);
  NotePendingGuestWrite(base, base + bytes);
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
  const bool own = MirrorOwnable(base, bytes);
  if (own) {
    if (!MirrorCopyInGuest(base, bytes))
      return false;
  } else {
    const u64 page = base::Max(Device().caps().sparse_page, kMirrorBlock);
    const u64 lo = MirrorOffset(base) & ~(page - 1);
    const u64 hi = (MirrorOffset(base) + bytes + page - 1) & ~(page - 1);
    if (!Device().CommitSparse(g_cs.mirror, lo, hi - lo))
      return false;
  }
  CsBatchBeginImpl();
  if (!g_cs.batch_open)
    return false;
  g_cs.list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  g_cs.list->FillBuffer(g_cs.mirror, MirrorOffset(base), bytes, value);
  g_cs.list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW | kAccessCopyRW);
  MirrorWriteBack(base, bytes, own);
  return true;
}

bool GpuFill(u64 base, u64 bytes, u32 lo, u32 hi) {
  if (!bytes || ((base | bytes) & (lo == hi ? 3 : 7)))
    return false;
  if (lo == hi && MirrorFill(base, bytes, lo))
    return true;
  CsBatchBeginImpl();
  if (!g_cs.batch_open)
    return false;
  g_cs.scratch_owner++;
  rhi::Buffer* src = AcquireScratch(bytes);
  if (!src)
    return false;
  rhi::CommandList* const c = g_cs.list;
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
  if (!bytes || !g_cs.batch_open)
    return false;
  g_cs.scratch_owner++;
  rhi::Buffer* tmp = AcquireScratch(bytes);
  return tmp && RecordGuestRead(g_cs.list, src, bytes, tmp) &&
         GpuWriteGuest(dst, bytes, tmp, 0);
}

bool GpuWriteGuest(u64 dst, u64 bytes, rhi::Buffer* src, u64 src_off) {
  CsBatchBeginImpl();
  bool mirrored = false;
  if (!g_cs.batch_open || !PrepareGuestWrite(dst, bytes, &mirrored))
    return false;
  RecordGuestWrite(g_cs.list, mirrored, dst, bytes, src, {}, src_off);
  return true;
}

static bool MirrorWatch(u64 first, u64 end) {
  if (GuestWriteTracker().Arm(first, end - first, true))
    return true;
  for (u64 page = first; page < end; page += kMirrorHotPage)
    if (!GuestWriteTracker().Arm(page, kMirrorHotPage, true))
      g_cs.mirror_hot.insert(page);
  return false;
}

u64 MirrorCacheBlock(u64 block) {
  if (!kMirror || !Mirror() || !g_cs.mirror->address() || !InMirror(block) ||
      MirrorOwned(block) || !BlockReadable(block))
    return 0;
  const u64 address = g_cs.mirror->address() + MirrorOffset(block);
  if (g_cs.mirror_cached.count(block))
    return address;
  if (!Device().CommitSparse(g_cs.mirror, MirrorOffset(block), kMirrorBlock))
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
    if (!g_cs.batch_open)
      return 0;
  }
  if (!MirrorCopy(block, kMirrorBlock, false, list))
    return 0;
  if (watched)
    g_cs.mirror_clean.insert(block);
  g_cs.mirror_cached.insert(block);
  return address;
}

bool GpuIndirectArgs(u64 args, u64 bytes) {
  if (!kMirror || !gcn::g_direct_table || !Mirror() ||
      !g_cs.mirror->address() || (args & 3) || !InMirror(args) ||
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

bool DrawParametersSupported() {
  return Device().caps().draw_parameters;
}

bool CanDrawMeshIndirect() {
  return Device().caps().mesh_shader;
}

rhi::Buffer* IndirectArgsBuffer(u64 args, u64* offset) {
  MirrorFrameRead();
  *offset = MirrorOffset(args);
  return g_cs.mirror;
}

constexpr u64 kIndirectRing = 4ull << 20;
constexpr u64 kIndirectEntry = 32;

// A slot in the indirect ring, or false when the ring is full this frame.
static bool IndirectRingSlot(u64* offset) {
  if (!g_cs.indirect_ring) {
    rhi::BufferDesc desc;
    desc.size = kIndirectRing;
    desc.usage = rhi::kBufferIndirect | rhi::kBufferCopySrc |
                 rhi::kBufferCopyDst | rhi::kBufferStorage;
    if (Device().caps().conditional)
      desc.usage |= rhi::kBufferPredicate;
    desc.name = "indirect args";
    g_cs.indirect_ring = Device().CreateBuffer(desc);
    if (!g_cs.indirect_ring)
      return false;
  }
  if (g_cs.indirect_frame != g_frame.num) {
    g_cs.indirect_frame = g_frame.num;
    g_cs.indirect_frame_n = 0;
  }
  if (++g_cs.indirect_frame_n > kIndirectRing / kIndirectEntry / 4)
    return false;
  *offset = g_cs.indirect_head;
  g_cs.indirect_head = (g_cs.indirect_head + kIndirectEntry) % kIndirectRing;
  return true;
}

static rhi::Buffer* StageArgs(rhi::CommandList* list,
                              u64 args,
                              u64 bytes,
                              u64* offset,
                              u32 dst_access) {
  if (bytes > kIndirectEntry || !gpu::IsReadableRangeCached(args, bytes) ||
      !IndirectRingSlot(offset))
    return nullptr;
  if (!RecordGuestRead(list, args, bytes, g_cs.indirect_ring, *offset))
    return nullptr;
  list->Barrier(rhi::kAccessCopyWrite, dst_access);
  g_cs_stats.indirect_staged_n++;
  return g_cs.indirect_ring;
}

rhi::Buffer* StageIndirectArgs(rhi::CommandList* list,
                               u64 args,
                               u64 bytes,
                               u64* offset) {
  return StageArgs(list, args, bytes, offset, rhi::kAccessIndirectRead);
}

DELTA_OPTION(bool, kIndirectZeroBase, "DELTA_GPU_INDIRECT_ZERO_BASE", true);

// GCN never adds an indirect draw's start vertex or start instance to the
// vertex and instance ids a shader sees: the CP writes them into user SGPRs
// and the shader adds them if it wants them. Vulkan folds firstVertex,
// vertexOffset and firstInstance into VertexIndex and InstanceIndex, so a
// shader indexing by instance id would read another element. Draw from a
// copy of the arguments with those fields zeroed.
rhi::Buffer* ZeroIndirectBases(rhi::CommandList* list,
                               rhi::Buffer* args,
                               u64 args_offset,
                               bool indexed,
                               u64* offset) {
  if (!kIndirectZeroBase || gcn::g_draw_parameters)
    return args;
  u64 slot;
  if (!IndirectRingSlot(&slot))
    return args;
  const u64 bytes = indexed ? 20 : 16;
  list->Barrier(rhi::kAccessAllWrite, rhi::kAccessCopyRead);
  list->CopyBuffer(g_cs.indirect_ring, slot, args, args_offset, bytes);
  list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyWrite);
  list->FillBuffer(g_cs.indirect_ring, slot + (indexed ? 12 : 8), 8, 0);
  list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessIndirectRead);
  *offset = slot;
  return g_cs.indirect_ring;
}

bool BeginGuestPredicate(u64 flag) {
  if (!Device().caps().conditional || !g_frame.recording || g_cs.predicate ||
      !FlushCsWritesForGpu(DefaultRenderer(), flag, 4, "cond"))
    return false;
  EndRegion();
  g_cs.predicate = StageArgs(g_frame.list, flag, 4, &g_predicate_offset,
                             rhi::kAccessPredicateRead);
  return g_cs.predicate != nullptr;
}

void EndGuestPredicate() {
  g_cs.predicate = nullptr;
}

rhi::Buffer* GuestPredicate(u64* offset) {
  *offset = g_predicate_offset;
  return g_cs.predicate;
}

namespace {
bool MirrorLive(u64 at) {
  const u64 block = at & ~(kMirrorBlock - 1);
  return g_cs.mirror_cached.count(block) && !MirrorOwned(block);
}

rhi::CommandList* MirrorRefreshList() {
  if (g_frame.recording) {
    EndRegion();
    return g_frame.list;
  }
  CsBatchBeginImpl();
  return g_cs.batch_open ? g_cs.list : nullptr;
}
}  // namespace

void MirrorRefreshWritten() {
  if (g_cs.mirror_stale.empty())
    return;
  rhi::CommandList* list = MirrorRefreshList();
  if (!list)
    return;
  list->Barrier(kAccessAll, rhi::kAccessCopyWrite);
  for (const auto& [first, end] : g_cs.mirror_stale) {
    if (!MirrorLive(first))
      continue;
    if (MirrorWatch(first, end))
      g_cs.mirror_clean.insert(first & ~(kMirrorBlock - 1));
    MirrorCopy(first, end - first, false, list);
  }
  g_cs.mirror_stale.clear();
}

void MirrorRefreshCached() {
  CollectGuestWrites();
  static int scanned_frame = -1;
  base::Vector<u64> released;
  if (scanned_frame != g_frame.num) {
    scanned_frame = g_frame.num;
    for (u64 block : g_cs.mirror_was_owned)
      if (!MirrorOwned(block))
        released.push_back(block);
  }
  for (u64 block : released) {
    g_cs.mirror_was_owned.erase(block);
    g_cs.mirror_cached.erase(block);
    if (const u64 address = MirrorCacheBlock(block))
      DirectRedirect(block, address);
  }
  if (g_cs.mirror_hot.empty())
    return;
  rhi::CommandList* list = MirrorRefreshList();
  if (!list)
    return;
  list->Barrier(kAccessAll, rhi::kAccessCopyWrite);
  base::Vector<u64> pages(g_cs.mirror_hot.begin(), g_cs.mirror_hot.end());
  base::Sort(pages.begin(), pages.end());
  for (size_t i = 0; i < pages.size();) {
    const u64 first = pages[i];
    u64 end = first + kMirrorHotPage;
    for (++i; i < pages.size() && pages[i] == end; ++i)
      end += kMirrorHotPage;
    const bool copy = MirrorLive(first);
    for (u64 page = first; page < end; page += kMirrorHotPage)
      if (!copy || GuestWriteTracker().Arm(page, kMirrorHotPage, true))
        g_cs.mirror_hot.erase(page);
    if (copy)
      MirrorCopy(first, end - first, false, list);
  }
}

namespace {
void AdvanceMirrorWindows(u64 first, u64 end) {
  auto it = g_cs.mirror_writes.upper_bound(first);
  if (it != g_cs.mirror_writes.begin())
    --it;
  u64 revision = 0;
  for (; it != g_cs.mirror_writes.end() && it->first < end; ++it)
    if (first < it->second.end) {
      if (!revision)
        revision = ++g_cs.mirror_seq;
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
  g_cs.scratch_owner++;
  rhi::Buffer* scratch = AcquireScratch(linear.size);
  const bool mirrored = MirrorHoldsNewest(base, tiled.size);
  rhi::Buffer* source = mirrored ? g_cs.mirror : AcquireScratch(tiled.size);
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
  if (g_cs.batch_open && !g_cs.batches[g_cs.batch_cur].in_frame)
    g_cs.chunk_needs_batch = true;
  g_cs_stats.tex_bridge_n++;
  g_cs_stats.tex_bridge_bytes += tiled.size;
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
    list->CopyBuffer(dst, dst_off, g_cs.mirror, MirrorOffset(base), bytes);
  else if (!RecordGuestRead(list, base, bytes, dst, dst_off))
    return false;
  MirrorFrameRead();
  list->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  if (!mirrored && !WatchMirrorHostReads(base, bytes))
    AdvanceMirrorWindows(base, base + bytes);
  if (g_cs.batch_open && !g_cs.batches[g_cs.batch_cur].in_frame)
    g_cs.chunk_needs_batch = true;
  g_cs_stats.buf_bridge_n++;
  g_cs_stats.buf_bridge_bytes += bytes;
  return true;
}

void MirrorNoteGuestWrites(u64 first, u64 end, bool announced) {
  if (g_cs.mirror_clean.empty() && g_cs.mirror_writes.empty())
    return;
  if (first >= end ||
      (first >= kMirrorLowWindow &&
       (end <= kMirrorPs5Base || first >= kMirrorPs5Base + kMirrorWindow)))
    return;
  if (announced)
    for (const auto& [lo, hi] : g_cs.mirror_self)
      if ((lo & ~u64(0xFFF)) <= first && end <= ((hi + 0xFFF) & ~u64(0xFFF)))
        return;
  AdvanceMirrorWindows(first, end);
  for (u64 block = first & ~(kMirrorBlock - 1); block < end;
       block += kMirrorBlock) {
    if (g_cs.own && InMirror(block) && !MirrorOwned(block))
      OwnAt(block).cpu_dirty.store(1, base::memory_order_release);
    g_cs.mirror_clean.erase(block);
    if (g_cs.mirror_cached.count(block))
      g_cs.mirror_stale.push_back(
          {base::Max(first, block), base::Min(end, block + kMirrorBlock)});
  }
}

void MirrorWritesCollected() {
  g_cs.mirror_self.clear();
}

}  // namespace gpu::render
