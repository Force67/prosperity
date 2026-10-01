/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#include "gpu/write_tracker.h"

#include "gpu/guest_page_table.h"
#include "host_memory/host_memory.h"
#include "options/options.h"
#include "profile/profile.h"

#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/userfaultfd.h>
#include <immintrin.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/time/time.h"
#endif

#if defined(__linux__) && defined(PAGEMAP_SCAN) && \
    defined(UFFD_FEATURE_WP_ASYNC)
#define DELTA_HAVE_WRITE_TRACKER 1
#endif

namespace gpu {
namespace {

DELTA_OPTION(bool, kScanMode, "DELTA_GPU_WT_SCAN", false);

constexpr u64 kPage = 4096;
constexpr u64 kRegisterAlign = 4096;
constexpr u64 kScanGap = 1024 * 1024;
constexpr u64 kCoalesceGap = 64 * 1024;
// A page reported this many times in one frame is written continuously.
constexpr u32 kHotReports = 3;
constexpr u32 kVolatileFrames = 600;

u64 PageDown(u64 v) {
  return v & ~(kPage - 1);
}
u64 PageUp(u64 v) {
  return (v + kPage - 1) & ~(kPage - 1);
}

// Adds [first, end) to a coalesced run map; returns the bytes newly covered.
u64 InsertRun(base::Map<u64, u64>& runs, u64 first, u64 end) {
  u64 added = end - first;
  auto it = runs.upper_bound(first);
  if (it != runs.begin() && base::Prev(it)->second >= first)
    --it;
  while (it != runs.end() && it->first <= end) {
    const u64 lo = base::Max(first, it->first), hi = base::Min(end, it->second);
    if (hi > lo)
      added -= hi - lo;
    first = base::Min(first, it->first);
    end = base::Max(end, it->second);
    it = runs.erase(it);
  }
  runs.emplace(first, end);
  return added;
}

// Removes [first, end) from a run map; returns the bytes it covered.
u64 EraseRun(base::Map<u64, u64>& runs, u64 first, u64 end) {
  u64 removed = 0;
  auto it = runs.upper_bound(first);
  if (it != runs.begin())
    --it;
  while (it != runs.end() && it->first < end) {
    const u64 a = it->first, b = it->second;
    if (b <= first) {
      ++it;
      continue;
    }
    removed += base::Min(b, end) - base::Max(a, first);
    it = runs.erase(it);
    if (a < first)
      runs.emplace(a, first);
    if (b > end)
      it = runs.emplace(end, b).first;
  }
  return removed;
}

// The parts of [first, end) no run covers.
base::Vector<base::Pair<u64, u64>> Gaps(const base::Map<u64, u64>& runs,
                                        u64 first,
                                        u64 end) {
  base::Vector<base::Pair<u64, u64>> gaps;
  auto it = runs.upper_bound(first);
  if (it != runs.begin() && base::Prev(it)->second > first)
    --it;
  u64 cursor = first;
  for (; it != runs.end() && it->first < end && cursor < end; ++it) {
    if (it->first > cursor)
      gaps.emplace_back(cursor, it->first);
    cursor = base::Max(cursor, it->second);
  }
  if (cursor < end)
    gaps.emplace_back(cursor, end);
  return gaps;
}

#if defined(__linux__)
// Fault mode. A page's `armed` state: open, protected, or written and waiting
// in the queue for Collect. Only the owner arms (open -> protected); a writer
// takes protected -> queued, so a page is never re-armed while the write that
// opened it is still in flight, and never queued twice.
constexpr u8 kOpen = 0, kProtected = 1, kQueued = 2;
// Pages protected at once are at most half of this, so the queue of written
// pages can never fill.
constexpr u64 kQueueSize = 1u << 20;
u64* g_queue = nullptr;  // written pages; 0 = slot not filled yet
base::Atomic<u64> g_queue_head{0};
u64 g_queue_tail = 0;  // owner only

// Readable, executable too: a protected page may hold code, and it faults on
// execution then as well; opening it must not leave it unexecutable.
constexpr int kOpenProt = PROT_READ | PROT_WRITE | PROT_EXEC;

void Enqueue(u64 page) {
  const u64 at = g_queue_head.fetch_add(1, base::memory_order_relaxed);
  __atomic_store_n(&g_queue[at % kQueueSize], page, __ATOMIC_RELEASE);
}

// Async-signal-safe: atomics, the page table and one mprotect.
bool OnWriteFault(uintptr_t addr) {
  const u64 page = addr & ~(kPage - 1);
  auto* p = const_cast<GuestPageTable::Page*>(GuestPages().Find(page));
  if (!p || !p->ever_armed.load(base::memory_order_relaxed))
    return false;
  u8 armed = kProtected;
  if (p->armed.compare_exchange_strong(armed, kQueued,
                                       base::memory_order_acq_rel)) {
    ::mprotect(reinterpret_cast<void*>(page), kPage, kOpenProt);
    Enqueue(page);
  }
  // Otherwise another thread is opening it right now: retrying the write is
  // all that is left to do.
  return true;
}

// Any thread: the host kernel is about to write [addr, addr+len).
void OnHostWrite(void* addr, size_t len) {
  const u64 first = reinterpret_cast<u64>(addr) & ~(kPage - 1);
  const u64 end =
      (reinterpret_cast<u64>(addr) + len + kPage - 1) & ~(kPage - 1);
  const GuestPageTable& table = GuestPages();
  u64 run = 0;
  const auto flush = [&](u64 at) {
    if (run)
      ::mprotect(reinterpret_cast<void*>(run), at - run, kOpenProt);
    run = 0;
  };
  for (u64 page = first; page < end; page += kPage) {
    auto* p = const_cast<GuestPageTable::Page*>(table.Find(page));
    u8 armed = kProtected;
    if (p && p->armed.compare_exchange_strong(armed, kQueued,
                                              base::memory_order_acq_rel)) {
      if (!run)
        run = page;
      Enqueue(page);
    } else {
      flush(page);
    }
  }
  flush(end);
}
#endif

}  // namespace

WriteTracker& GuestWriteTracker() {
  static WriteTracker tracker;
  return tracker;
}

bool WriteTracker::Enable() {
  if (enabled())
    return true;
  return (!kScanMode && EnableFault()) || EnableScan();
}

#if defined(__linux__)
bool WriteTracker::EnableFault() {
  void* queue = mmap(nullptr, kQueueSize * sizeof(u64), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (queue == MAP_FAILED)
    return false;
  g_queue = static_cast<u64*>(queue);
  host_memory::SetWriteFaultHandler(&OnWriteFault);
  host_memory::SetHostWriteHook(&OnHostWrite);
  mode_ = Mode::kFault;
  return true;
}

bool WriteTracker::ArmFault(u64 first, u64 end) {
  DELTA_ZONE("wt.arm");
  if ((armed_bytes_ + (end - first)) / kPage >= kQueueSize / 2)
    return false;
  GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage) {
    const GuestPageTable::Page* p = table.Find(page);
    if (p && (p->volatile_until > frame_ ||
              p->armed.load(base::memory_order_acquire) == kQueued))
      return false;  // hot, or its last write is not collected yet
  }
  // Protect each run of open pages; mark before protecting, so a write in
  // between is one the fault handler can take.
  u64 run = 0;
  const auto protect = [&](u64 at) {
    if (!run)
      return true;
    const bool ok = ::mprotect(reinterpret_cast<void*>(run), at - run,
                               PROT_READ) == 0;
    if (!ok)
      for (u64 page = run; page < at; page += kPage)
        table.At(page)->armed.store(kOpen, base::memory_order_release);
    run = 0;
    return ok;
  };
  for (u64 page = first; page < end; page += kPage) {
    GuestPageTable::Page* p = table.At(page);
    if (!p)
      return false;
    if (p->armed.load(base::memory_order_acquire) != kOpen) {
      if (!protect(page))
        return false;
      continue;
    }
    p->ever_armed.store(1, base::memory_order_relaxed);
    p->armed.store(kProtected, base::memory_order_release);
    if (!run)
      run = page;
  }
  if (!protect(end))
    return false;
  armed_bytes_ += InsertRun(armed_, first, end);
  return true;
}

void WriteTracker::DisarmFault(u64 first, u64 end) {
  DELTA_ZONE("wt.disarm");
  GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage)
    if (GuestPageTable::Page* p = table.At(page)) {
      u8 armed = kProtected;
      p->armed.compare_exchange_strong(armed, kOpen,
                                       base::memory_order_acq_rel);
    }
  ::mprotect(reinterpret_cast<void*>(first), end - first, kOpenProt);
  armed_bytes_ -= EraseRun(armed_, first, end);
}

void WriteTracker::TakeFaulted(base::Vector<Range>& out) {
  const u64 head = g_queue_head.load(base::memory_order_acquire);
  if (head == g_queue_tail)
    return;
  base::Vector<u64> pages;
  pages.reserve(head - g_queue_tail);
  for (u64 at = g_queue_tail; at < head; at++) {
    u64* slot = &g_queue[at % kQueueSize];
    u64 page;
    // A writer between claiming the slot and filling it: a few instructions.
    while (!(page = __atomic_exchange_n(slot, 0, __ATOMIC_ACQUIRE)))
      _mm_pause();
    pages.push_back(page);
  }
  g_queue_tail = head;
  base::Sort(pages.begin(), pages.end());
  GuestPageTable& table = GuestPages();
  for (size_t i = 0; i < pages.size();) {
    const u64 first = pages[i];
    u64 end = first + kPage;
    for (++i; i < pages.size() && pages[i] <= end; ++i)
      end = base::Max(end, pages[i] + kPage);
    for (u64 page = first; page < end; page += kPage)
      if (GuestPageTable::Page* p = table.At(page)) {
        u8 armed = kQueued;
        p->armed.compare_exchange_strong(armed, kOpen,
                                         base::memory_order_acq_rel);
      }
    armed_bytes_ -= EraseRun(armed_, first, end);
    out.emplace_back(first, end);
  }
}
#else
bool WriteTracker::EnableFault() {
  return false;
}
bool WriteTracker::ArmFault(u64, u64) {
  return false;
}
void WriteTracker::DisarmFault(u64, u64) {}
void WriteTracker::TakeFaulted(base::Vector<Range>&) {}
#endif

#if defined(DELTA_HAVE_WRITE_TRACKER)

long WriteTracker::MinorFaults() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_minflt;
}

bool WriteTracker::EnableScan() {
  const int uffd = static_cast<int>(
      syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
  if (uffd < 0)
    return false;
  uffdio_api api{};
  api.api = UFFD_API;
  api.features = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED;
  const int pagemap = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (ioctl(uffd, UFFDIO_API, &api) ||
      !(api.features & UFFD_FEATURE_WP_ASYNC) || pagemap < 0) {
    close(uffd);
    if (pagemap >= 0)
      close(pagemap);
    return false;
  }
  uffd_ = uffd;
  pagemap_ = pagemap;
  mode_ = Mode::kScan;
  return true;
}

bool WriteTracker::Register(u64 first, u64 end) {
  DELTA_ZONE("wt.register");
  for (const auto& [lo, hi] : Gaps(registered_, first, end)) {
    // Coarser than the pages asked for, so neighbouring buffers do not each
    // split the mapping again; a range that runs into a hole falls back to the
    // exact pages.
    const u64 wide_lo = lo & ~(kRegisterAlign - 1);
    const u64 wide_hi = (hi + kRegisterAlign - 1) & ~(kRegisterAlign - 1);
    bool registered = false;
    for (const auto& [a, b] :
         {base::Pair{wide_lo, wide_hi}, base::Pair{lo, hi}}) {
      uffdio_register reg{};
      reg.range.start = a;
      reg.range.len = b - a;
      reg.mode = UFFDIO_REGISTER_MODE_WP;
      if (!ioctl(uffd_, UFFDIO_REGISTER, &reg)) {
        InsertRun(registered_, a, b);
        registered = true;
        break;
      }
    }
    if (!registered)
      return false;
  }
  return true;
}

bool WriteTracker::ArmRange(u64 first, u64 end) {
  if (mode_ == Mode::kFault)
    return ArmFault(first, end);
  DELTA_ZONE("wt.arm");
  const GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage)
    if (const GuestPageTable::Page* p = table.Find(page);
        p && p->volatile_until > frame_)
      return false;
  if (!Register(first, end))
    return false;
  for (const auto& [lo, hi] : Gaps(armed_, first, end)) {
    uffdio_writeprotect wp{};
    wp.range.start = lo;
    wp.range.len = hi - lo;
    wp.mode = UFFDIO_WRITEPROTECT_MODE_WP;
    if (ioctl(uffd_, UFFDIO_WRITEPROTECT, &wp))
      return false;
    armed_bytes_ += InsertRun(armed_, lo, hi);
    MarkArmed(lo, hi, true);
  }
  return true;
}

bool WriteTracker::Arm(u64 base, u64 bytes) {
  if (!enabled() || !bytes)
    return false;
  const u64 first = PageDown(base), end = PageUp(base + bytes);
  if (policy_.arm_filter && !policy_.arm_filter(first, end))
    return false;
  // Most calls ask again about pages armed long ago.
  if (AllArmed(first, end))
    return true;
  // Fault mode protects exactly what it is asked to: a gap page may be code,
  // a guard page or another mapping, and costs nothing to leave alone.
  if (mode_ == Mode::kFault)
    return ArmRange(first, end);
  // Close a small gap to a neighbouring armed run: every separate run is a
  // mapping of its own after registration, and the scan pays per mapping
  // (GTA:SA held ~2700 runs, ~10 ms a frame of scans).
  u64 wide_first = first, wide_end = end;
  auto next = armed_.lower_bound(first);
  if (next != armed_.begin()) {
    const auto prev = base::Prev(next);
    if (prev->second < first && first - prev->second <= kCoalesceGap)
      wide_first = prev->second;
  }
  if (next != armed_.end() && next->first > end &&
      next->first - end <= kCoalesceGap)
    wide_end = next->first;
  if ((wide_first != first || wide_end != end) &&
      ArmRange(wide_first, wide_end))
    return true;
  return ArmRange(first, end);
}

// Unregistered too: the span scan arms every registered page it walks, and a
// page left registered would be armed again at the next one.
void WriteTracker::Disarm(u64 first, u64 end) {
  if (mode_ == Mode::kFault) {
    DisarmFault(first, end);
    return;
  }
  DELTA_ZONE("wt.disarm");
  uffdio_writeprotect wp{};
  wp.range.start = first;
  wp.range.len = end - first;
  wp.mode = 0;
  ioctl(uffd_, UFFDIO_WRITEPROTECT, &wp);
  uffdio_range range{first, end - first};
  ioctl(uffd_, UFFDIO_UNREGISTER, &range);
  EraseRun(registered_, first, end);
  armed_bytes_ -= EraseArmed(first, end);
}

// The walk costs ~0.3 us a mapping in [first, end), registered or not, and a
// few ns a page.
bool WriteTracker::Scan(u64 first, u64 end, base::Vector<Range>& out) {
  DELTA_ZONE("wt.scan");
  page_region regions[256];
  pm_scan_arg arg{};
  arg.size = sizeof(arg);
  arg.flags = PM_SCAN_WP_MATCHING;
  arg.start = first;
  arg.end = end;
  arg.vec = reinterpret_cast<u64>(regions);
  arg.vec_len = base::ArraySize(regions);
  arg.category_mask = PAGE_IS_WRITTEN | PAGE_IS_WPALLOWED;
  arg.return_mask = PAGE_IS_WRITTEN;
  for (;;) {
    const long n = ioctl(pagemap_, PAGEMAP_SCAN, &arg);
    if (n < 0)
      return false;
    for (long i = 0; i < n; i++)
      out.emplace_back(regions[i].start, regions[i].end);
    if (arg.walk_end >= end || n < static_cast<long>(base::ArraySize(regions)))
      return true;
    arg.start = arg.walk_end;
  }
}

#else

bool WriteTracker::EnableScan() {
  return false;
}
bool WriteTracker::Register(u64, u64) {
  return false;
}
bool WriteTracker::Arm(u64 base, u64 bytes) {
  if (mode_ != Mode::kFault || !bytes)
    return false;
  const u64 first = PageDown(base), end = PageUp(base + bytes);
  if (policy_.arm_filter && !policy_.arm_filter(first, end))
    return false;
  return AllArmed(first, end) || ArmFault(first, end);
}
bool WriteTracker::ArmRange(u64 first, u64 end) {
  return mode_ == Mode::kFault && ArmFault(first, end);
}
void WriteTracker::Disarm(u64 first, u64 end) {
  DisarmFault(first, end);
}
long WriteTracker::MinorFaults() {
  return 0;
}
bool WriteTracker::Scan(u64, u64, base::Vector<Range>&) {
  return false;
}

#endif

void WriteTracker::Drain(base::Vector<Range>& out) {
  base::Vector<Range> remapped;
  {
    base::LockGuard lock(noted_lock_);
    out.insert(out.end(), noted_.begin(), noted_.end());
    noted_.clear();
    remapped.swap(remapped_);
  }
  // A new mapping is not registered, whatever the old one was.
  for (const auto& [first, end] : remapped) {
    EraseRun(registered_, first, end);
    armed_bytes_ -= EraseArmed(first, end);
    out.emplace_back(first, end);
  }
}

bool WriteTracker::AllArmed(u64 first, u64 end) const {
  const GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage) {
    const GuestPageTable::Page* p = table.Find(page);
    if (!p || p->armed.load(base::memory_order_acquire) != 1)
      return false;
  }
  return true;
}

void WriteTracker::MarkArmed(u64 first, u64 end, bool armed) {
  GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage)
    if (GuestPageTable::Page* p = table.At(page))
      p->armed.store(armed ? 1 : 0, base::memory_order_release);
}

// EraseRun on armed_, clearing only the pages that were armed: a remap can
// span gigabytes of which a few pages ever were.
u64 WriteTracker::EraseArmed(u64 first, u64 end) {
  auto it = armed_.upper_bound(first);
  if (it != armed_.begin())
    --it;
  for (; it != armed_.end() && it->first < end; ++it)
    if (it->second > first)
      MarkArmed(base::Max(first, it->first), base::Min(end, it->second), false);
  return EraseRun(armed_, first, end);
}

void WriteTracker::Collect(base::Vector<Range>& out) {
  if (enabled() && policy_.before_collect)
    policy_.before_collect();
  DELTA_ZONE("wt.collect");
  if (!enabled())
    return;
  collects_++;
  const auto t0 = base::TimeTicks::Now();
  Drain(out);
  const size_t first_new = out.size();
  if (mode_ == Mode::kFault)
    TakeFaulted(out);
  // A write to an armed page is a page fault, counted in the faulting task's
  // minor faults whether the task was in user or kernel mode. No fault
  // anywhere in the process since the last scan: nothing armed was written,
  // and the scan (~36 ns an armed page, ~60 times a frame) can be skipped.
  // Four in five collects in GTA:SA are that quiet.
  const long faults = mode_ == Mode::kScan ? MinorFaults() : faults_at_scan_;
  if (faults != faults_at_scan_) {
    // Runs close together are scanned as one span; a wide gap is not, since
    // the walk pays for every mapping in it (GTA:SA keeps ~37k 64 KiB guest
    // mappings between its buffers).
    for (auto it = armed_.begin(); it != armed_.end();) {
      const u64 first = it->first;
      u64 end = it->second;
      for (++it; it != armed_.end() && it->first - end < kScanGap; ++it)
        end = it->second;
      if (!Scan(first, end, out))
        out.emplace_back(first, end);  // not expected: trust nothing in it
    }
  }
  faults_at_scan_ = faults;
  GuestPageTable& table = GuestPages();
  for (size_t i = first_new; i < out.size(); i++)
    for (u64 page = PageDown(out[i].first); page < out[i].second;
         page += kPage) {
      written_pages_++;
      GuestPageTable::Page* p = table.At(page);
      if (!p)
        continue;
      if (p->report_frame != frame_) {
        p->previous_report = p->report_frame;
        p->report_frame = frame_;
        p->reports = 0;
        reported_.push_back(page);
      }
      p->reports++;
    }
  collect_ns_ += ((base::TimeTicks::Now() - t0).InMicroseconds() * 1000);
}

void WriteTracker::NoteWrite(u64 base, u64 bytes) {
  if (!enabled() || !bytes)
    return;
  base::LockGuard lock(noted_lock_);
  noted_.emplace_back(PageDown(base), PageUp(base + bytes));
}

void WriteTracker::Release(u64 base, u64 bytes) {
  if (!enabled() || !bytes)
    return;
  const u64 first = PageDown(base), end = PageUp(base + bytes);
  if (mode_ == Mode::kFault)
    DisarmFault(first, end);
  else
    Disarm(first, end);
  NoteWrite(base, bytes);
}

void WriteTracker::NoteRemap(u64 base, u64 bytes) {
  if (!enabled() || !bytes)
    return;
  if (policy_.on_remap)
    policy_.on_remap(base, bytes);
  base::LockGuard lock(noted_lock_);
  remapped_.emplace_back(PageDown(base), PageUp(base + bytes));
}

void WriteTracker::EndFrame() {
  if (!enabled())
    return;
  // Written this frame and the one before, or several times in this one: data
  // the title rewrites every frame. A copy of it is good for one submission at
  // most, and keeping it armed costs a fault and a scan every frame.
  GuestPageTable& table = GuestPages();
  base::Vector<u64> demote;
  for (u64 page : reported_) {
    const GuestPageTable::Page* p = table.Find(page);
    if (p && (p->reports >= kHotReports || p->previous_report + 1 == frame_))
      demote.push_back(page);
  }
  reported_.clear();
  frame_++;
  base::Sort(demote.begin(), demote.end());
  for (size_t i = 0; i < demote.size();) {
    const u64 first = demote[i];
    u64 end = first + kPage;
    for (++i; i < demote.size() && demote[i] == end; ++i)
      end += kPage;
    for (u64 page = first; page < end; page += kPage)
      if (GuestPageTable::Page* p = table.At(page))
        p->volatile_until = frame_ + kVolatileFrames;
    Disarm(first, end);
    // Unarmed, its writes go unreported: whoever holds a copy must drop it.
    NoteWrite(first, end - first);
  }
}

}  // namespace gpu
