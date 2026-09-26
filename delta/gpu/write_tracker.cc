/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#include "gpu/write_tracker.h"

#include <algorithm>
#include <chrono>
#include <iterator>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)
#define DELTA_HAVE_WRITE_TRACKER 1
#endif

namespace gpu {
namespace {

constexpr u64 kPage = 4096;
constexpr u64 kRegisterAlign = 4096;
constexpr u64 kScanGap = 1024 * 1024;
constexpr u64 kCoalesceGap = 64 * 1024;
// A page reported this many times in one frame is written continuously.
constexpr u32 kHotReports = 3;
constexpr int kVolatileFrames = 600;

u64 PageDown(u64 v) {
  return v & ~(kPage - 1);
}
u64 PageUp(u64 v) {
  return (v + kPage - 1) & ~(kPage - 1);
}

// Adds [first, end) to a coalesced run map; returns the bytes newly covered.
u64 InsertRun(std::map<u64, u64>& runs, u64 first, u64 end) {
  u64 added = end - first;
  auto it = runs.upper_bound(first);
  if (it != runs.begin() && std::prev(it)->second >= first)
    --it;
  while (it != runs.end() && it->first <= end) {
    const u64 lo = std::max(first, it->first), hi = std::min(end, it->second);
    if (hi > lo)
      added -= hi - lo;
    first = std::min(first, it->first);
    end = std::max(end, it->second);
    it = runs.erase(it);
  }
  runs.emplace(first, end);
  return added;
}

// Removes [first, end) from a run map; returns the bytes it covered.
u64 EraseRun(std::map<u64, u64>& runs, u64 first, u64 end) {
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
    removed += std::min(b, end) - std::max(a, first);
    it = runs.erase(it);
    if (a < first)
      runs.emplace(a, first);
    if (b > end)
      it = runs.emplace(end, b).first;
  }
  return removed;
}

// The parts of [first, end) no run covers.
std::vector<std::pair<u64, u64>> Gaps(const std::map<u64, u64>& runs,
                                      u64 first,
                                      u64 end) {
  std::vector<std::pair<u64, u64>> gaps;
  auto it = runs.upper_bound(first);
  if (it != runs.begin() && std::prev(it)->second > first)
    --it;
  u64 cursor = first;
  for (; it != runs.end() && it->first < end && cursor < end; ++it) {
    if (it->first > cursor)
      gaps.emplace_back(cursor, it->first);
    cursor = std::max(cursor, it->second);
  }
  if (cursor < end)
    gaps.emplace_back(cursor, end);
  return gaps;
}

}  // namespace

WriteTracker& GuestWriteTracker() {
  static WriteTracker tracker;
  return tracker;
}

#if defined(DELTA_HAVE_WRITE_TRACKER)

long WriteTracker::MinorFaults() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_minflt;
}

bool WriteTracker::Enable() {
  if (uffd_ >= 0)
    return true;
  const int uffd = static_cast<int>(
      syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
  if (uffd < 0)
    return false;
  uffdio_api api{};
  api.api = UFFD_API;
  api.features = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED;
  const int pagemap = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (ioctl(uffd, UFFDIO_API, &api) || !(api.features & UFFD_FEATURE_WP_ASYNC) ||
      pagemap < 0) {
    close(uffd);
    if (pagemap >= 0)
      close(pagemap);
    return false;
  }
  uffd_ = uffd;
  pagemap_ = pagemap;
  return true;
}

bool WriteTracker::Register(u64 first, u64 end) {
  for (const auto& [lo, hi] : Gaps(registered_, first, end)) {
    // Coarser than the pages asked for, so neighbouring buffers do not each
    // split the mapping again; a range that runs into a hole falls back to the
    // exact pages.
    const u64 wide_lo = lo & ~(kRegisterAlign - 1);
    const u64 wide_hi = (hi + kRegisterAlign - 1) & ~(kRegisterAlign - 1);
    bool registered = false;
    for (const auto& [a, b] : {std::pair{wide_lo, wide_hi}, std::pair{lo, hi}}) {
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
  if (!volatile_.empty())
    for (u64 page = first; page < end; page += kPage)
      if (volatile_.count(page))
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
  }
  return true;
}

bool WriteTracker::Arm(u64 base, u64 bytes) {
  if (uffd_ < 0 || !bytes)
    return false;
  const u64 first = PageDown(base), end = PageUp(base + bytes);
  // Close a small gap to a neighbouring armed run: every separate run is a
  // mapping of its own after registration, and the scan pays per mapping
  // (GTA:SA held ~2700 runs, ~10 ms a frame of scans).
  u64 wide_first = first, wide_end = end;
  auto next = armed_.lower_bound(first);
  if (next != armed_.begin()) {
    const auto prev = std::prev(next);
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
  uffdio_writeprotect wp{};
  wp.range.start = first;
  wp.range.len = end - first;
  wp.mode = 0;
  ioctl(uffd_, UFFDIO_WRITEPROTECT, &wp);
  uffdio_range range{first, end - first};
  ioctl(uffd_, UFFDIO_UNREGISTER, &range);
  EraseRun(registered_, first, end);
  armed_bytes_ -= EraseRun(armed_, first, end);
}

// The walk costs ~0.3 us a mapping in [first, end), registered or not, and a
// few ns a page.
bool WriteTracker::Scan(u64 first, u64 end, std::vector<Range>& out) {
  page_region regions[256];
  pm_scan_arg arg{};
  arg.size = sizeof(arg);
  arg.flags = PM_SCAN_WP_MATCHING;
  arg.start = first;
  arg.end = end;
  arg.vec = reinterpret_cast<u64>(regions);
  arg.vec_len = std::size(regions);
  arg.category_mask = PAGE_IS_WRITTEN | PAGE_IS_WPALLOWED;
  arg.return_mask = PAGE_IS_WRITTEN;
  for (;;) {
    const long n = ioctl(pagemap_, PAGEMAP_SCAN, &arg);
    if (n < 0)
      return false;
    for (long i = 0; i < n; i++)
      out.emplace_back(regions[i].start, regions[i].end);
    if (arg.walk_end >= end || n < static_cast<long>(std::size(regions)))
      return true;
    arg.start = arg.walk_end;
  }
}

#else

bool WriteTracker::Enable() {
  return false;
}
bool WriteTracker::Register(u64, u64) {
  return false;
}
bool WriteTracker::Arm(u64, u64) {
  return false;
}
bool WriteTracker::ArmRange(u64, u64) {
  return false;
}
void WriteTracker::Disarm(u64, u64) {}
long WriteTracker::MinorFaults() {
  return 0;
}
bool WriteTracker::Scan(u64, u64, std::vector<Range>&) {
  return false;
}

#endif

void WriteTracker::Drain(std::vector<Range>& out) {
  std::vector<Range> remapped;
  {
    std::lock_guard lock(noted_lock_);
    out.insert(out.end(), noted_.begin(), noted_.end());
    noted_.clear();
    remapped.swap(remapped_);
  }
  // A new mapping is not registered, whatever the old one was.
  for (const auto& [first, end] : remapped) {
    EraseRun(registered_, first, end);
    armed_bytes_ -= EraseRun(armed_, first, end);
    out.emplace_back(first, end);
  }
}

void WriteTracker::Collect(std::vector<Range>& out) {
  if (uffd_ < 0)
    return;
  collects_++;
  const auto t0 = std::chrono::steady_clock::now();
  Drain(out);
  const size_t first_new = out.size();
  // A write to an armed page is a page fault, counted in the faulting task's
  // minor faults whether the task was in user or kernel mode. No fault
  // anywhere in the process since the last scan: nothing armed was written,
  // and the scan (~36 ns an armed page, ~60 times a frame) can be skipped.
  // Four in five collects in GTA:SA are that quiet.
  const long faults = MinorFaults();
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
  for (size_t i = first_new; i < out.size(); i++)
    for (u64 page = PageDown(out[i].first); page < out[i].second; page += kPage) {
      written_pages_++;
      hot_[page]++;
    }
  collect_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
}

void WriteTracker::NoteWrite(u64 base, u64 bytes) {
  if (uffd_ < 0 || !bytes)
    return;
  std::lock_guard lock(noted_lock_);
  noted_.emplace_back(PageDown(base), PageUp(base + bytes));
}

void WriteTracker::NoteRemap(u64 base, u64 bytes) {
  if (uffd_ < 0 || !bytes)
    return;
  std::lock_guard lock(noted_lock_);
  remapped_.emplace_back(PageDown(base), PageUp(base + bytes));
}

void WriteTracker::EndFrame() {
  if (uffd_ < 0)
    return;
  frame_++;
  for (auto it = volatile_.begin(); it != volatile_.end();)
    it = --it->second <= 0 ? volatile_.erase(it) : std::next(it);
  // Written this frame and the one before, or several times in this one: data
  // the title rewrites every frame. A copy of it is good for one submission at
  // most, and keeping it armed costs a fault and a scan every frame.
  std::vector<u64> demote;
  for (const auto& [page, reports] : hot_) {
    int& last = last_written_[page];
    if (reports >= kHotReports || last == frame_ - 1)
      demote.push_back(page);
    last = frame_;
  }
  hot_.clear();
  if (frame_ % 600 == 0)
    std::erase_if(last_written_,
                  [&](const auto& kv) { return kv.second < frame_ - 1; });
  std::sort(demote.begin(), demote.end());
  for (size_t i = 0; i < demote.size();) {
    const u64 first = demote[i];
    u64 end = first + kPage;
    for (++i; i < demote.size() && demote[i] == end; ++i)
      end += kPage;
    for (u64 page = first; page < end; page += kPage)
      volatile_[page] = kVolatileFrames;
    Disarm(first, end);
    // Unarmed, its writes go unreported: whoever holds a copy must drop it.
    NoteWrite(first, end - first);
  }
}

}  // namespace gpu
