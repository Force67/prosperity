/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#include "gpu/write_tracker.h"
#include "guest/session.h"

#include "gpu/guest_page_table.h"
#include "host_memory/host_memory.h"
#include "options/options.h"
#include "profile/profile.h"

#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/userfaultfd.h>
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
constexpr u32 kHotReports = 3;
constexpr u32 kVolatileFrames = 600;

u64 PageDown(u64 v) {
  return v & ~(kPage - 1);
}
u64 PageUp(u64 v) {
  return (v + kPage - 1) & ~(kPage - 1);
}

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
constexpr u8 kOpen = 0, kProtected = 1, kQueued = 2;
constexpr u64 kQueueSize = 1u << 20;
u64* g_queue = nullptr;
base::Atomic<u64> g_queue_head{0};
u64 g_queue_tail = 0;

constexpr int kOpenProt = PROT_READ | PROT_WRITE | PROT_EXEC;

void Enqueue(u64 page) {
  const u64 at = g_queue_head.fetch_add(1, base::memory_order_relaxed);
  __atomic_store_n(&g_queue[at % kQueueSize], page, __ATOMIC_RELEASE);
}

base::Atomic<bool (*)(u64, u64)> g_owner_resolve{nullptr};
base::Atomic<bool (*)(u64, u64)> g_owner_owns{nullptr};
base::Atomic<bool (*)(u64)> g_owner_raced{nullptr};

// A fault no protection of ours explains: the owner says whether the page was
// reopened (or reclaimed) since, so the access can retry.
bool RetryUnclaimed(u64 page) {
  const auto raced = g_owner_raced.load(base::memory_order_acquire);
  return raced && raced(page);
}

// Armed but not ours to open: a disarm or another fault's open is mid-flight,
// so retry; bounded so a page someone else shut for good still dies.
bool RetryArmed(u64 page) {
  static thread_local u64 last = 0;
  static thread_local u32 repeats = 0;
  repeats = page == last ? repeats + 1 : 0;
  last = page;
  return repeats < 100000 || RetryUnclaimed(page);
}

bool OnWriteFault(uintptr_t addr) {
  const u64 page = addr & ~(kPage - 1);
  if (const auto resolve = g_owner_resolve.load(base::memory_order_acquire);
      resolve && resolve(page, page + kPage))
    return true;
  auto* p = const_cast<GuestPageTable::Page*>(GuestPages().Find(page));
  if (!p || !p->ever_armed.load(base::memory_order_relaxed))
    return RetryUnclaimed(page);
  u8 armed = kProtected;
  if (p->armed.compare_exchange_strong(armed, kQueued,
                                       base::memory_order_acq_rel)) {
    ::mprotect(reinterpret_cast<void*>(page), kPage, kOpenProt);
    Enqueue(page);
    return true;
  }
  return RetryUnclaimed(page) || RetryArmed(page);
}

void OnHostWrite(void* addr, size_t len) {
  const u64 first = reinterpret_cast<u64>(addr) & ~(kPage - 1);
  const u64 end =
      (reinterpret_cast<u64>(addr) + len + kPage - 1) & ~(kPage - 1);
  if (const auto resolve = g_owner_resolve.load(base::memory_order_acquire))
    resolve(first, end);
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

void WriteTracker::Reset() {
  host_memory::SetWriteFaultHandler(nullptr);
  host_memory::SetHostWriteHook(nullptr);
  SetOwner(nullptr, nullptr);
#if defined(__linux__)
  if (g_queue)
    munmap(g_queue, kQueueSize * sizeof(u64));
  g_queue = nullptr;
  g_queue_head.store(0);
  g_queue_tail = 0;
  if (uffd_ >= 0)
    close(uffd_);
  if (pagemap_ >= 0)
    close(pagemap_);
#endif
  guest::ResetResource(*this);
}

namespace {
const guest::SessionReset g_session_reset([] { GuestWriteTracker().Reset(); });
}  // namespace

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

void WriteTracker::SetOwner(bool (*resolve)(u64, u64),
                            bool (*owns)(u64, u64),
                            bool (*raced)(u64)) {
  g_owner_raced.store(raced, base::memory_order_release);
  g_owner_owns.store(owns, base::memory_order_release);
  g_owner_resolve.store(resolve, base::memory_order_release);
  SetGuestOwnerHolds(owns);
}

void WriteTracker::OpenedByOwner(u64 first, u64 end) {
  const GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage) {
    auto* p = const_cast<GuestPageTable::Page*>(table.Find(page));
    u8 armed = kProtected;
    if (p && p->armed.compare_exchange_strong(armed, kQueued,
                                              base::memory_order_acq_rel))
      Enqueue(page);
  }
}

bool WriteTracker::ArmFault(u64 first, u64 end, bool hot) {
  DELTA_ZONE("wt.arm");
  if (const auto owns = g_owner_owns.load(base::memory_order_acquire);
      owns && owns(first, end))
    return false;
  if ((armed_bytes_ + (end - first)) / kPage >= kQueueSize / 2)
    return false;
  GuestPageTable& table = GuestPages();
  for (u64 page = first; page < end; page += kPage) {
    const GuestPageTable::Page* p = table.Find(page);
    if (p && ((!hot && p->volatile_until > frame_) ||
              p->armed.load(base::memory_order_acquire) == kQueued))
      return false;
  }
  u64 run = 0;
  const auto protect = [&](u64 at) {
    if (!run)
      return true;
    const bool ok =
        ::mprotect(reinterpret_cast<void*>(run), at - run, PROT_READ) == 0;
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
    p->kept |= hot;
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
    while (!(page = __atomic_exchange_n(slot, 0, __ATOMIC_ACQUIRE))) {
#if defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#endif
    }
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
void WriteTracker::SetOwner(bool (*)(u64, u64),
                            bool (*)(u64, u64),
                            bool (*)(u64)) {}
void WriteTracker::OpenedByOwner(u64, u64) {}
bool WriteTracker::ArmFault(u64, u64, bool) {
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

bool WriteTracker::ArmRange(u64 first, u64 end, bool hot) {
  if (mode_ == Mode::kFault)
    return ArmFault(first, end, hot);
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

bool WriteTracker::Arm(u64 base, u64 bytes, bool hot) {
  if (!enabled() || !bytes)
    return false;
  const u64 first = PageDown(base), end = PageUp(base + bytes);
  if (policy_.arm_filter && !policy_.arm_filter(first, end))
    return false;
  if (AllArmed(first, end))
    return true;
  if (mode_ == Mode::kFault)
    return ArmRange(first, end, hot);
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
bool WriteTracker::Arm(u64 base, u64 bytes, bool hot) {
  if (mode_ != Mode::kFault || !bytes)
    return false;
  const u64 first = PageDown(base), end = PageUp(base + bytes);
  if (policy_.arm_filter && !policy_.arm_filter(first, end))
    return false;
  return AllArmed(first, end) || ArmFault(first, end, hot);
}
bool WriteTracker::ArmRange(u64 first, u64 end, bool hot) {
  return mode_ == Mode::kFault && ArmFault(first, end, hot);
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

size_t WriteTracker::Drain(base::Vector<Range>& out) {
  base::Vector<Range> remapped;
  size_t announced = 0;
  {
    base::LockGuard lock(noted_lock_);
    announced = noted_.size();
    out.insert(out.end(), noted_.begin(), noted_.end());
    noted_.clear();
    remapped.swap(remapped_);
  }
  for (const auto& [first, end] : remapped) {
    EraseRun(registered_, first, end);
    armed_bytes_ -= EraseArmed(first, end);
    out.emplace_back(first, end);
  }
  return announced;
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

u64 WriteTracker::EraseArmed(u64 first, u64 end) {
  auto it = armed_.upper_bound(first);
  if (it != armed_.begin())
    --it;
  for (; it != armed_.end() && it->first < end; ++it)
    if (it->second > first)
      MarkArmed(base::Max(first, it->first), base::Min(end, it->second), false);
  return EraseRun(armed_, first, end);
}

size_t WriteTracker::Collect(base::Vector<Range>& out) {
  if (enabled() && policy_.before_collect)
    policy_.before_collect();
  DELTA_ZONE("wt.collect");
  if (!enabled())
    return 0;
  collects_++;
  const auto t0 = base::TimeTicks::Now();
  const size_t announced = Drain(out);
  const size_t first_new = out.size();
  if (mode_ == Mode::kFault)
    TakeFaulted(out);
  const long faults = mode_ == Mode::kScan ? MinorFaults() : faults_at_scan_;
  if (faults != faults_at_scan_) {
    for (auto it = armed_.begin(); it != armed_.end();) {
      const u64 first = it->first;
      u64 end = it->second;
      for (++it; it != armed_.end() && it->first - end < kScanGap; ++it)
        end = it->second;
      if (!Scan(first, end, out))
        out.emplace_back(first, end);
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
  return announced;
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
  GuestPageTable& table = GuestPages();
  base::Vector<u64> demote;
  for (u64 page : reported_) {
    const GuestPageTable::Page* p = table.Find(page);
    if (p && !p->kept &&
        (p->reports >= kHotReports || p->previous_report + 1 == frame_))
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
    NoteWrite(first, end - first);
  }
}

}  // namespace gpu
