/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/threading/mutex.h"

namespace gpu {

// Which guest pages were written since a cache copied them, for caches that
// keep a copy of guest memory longer than one submission.
//
// Default (fault mode): an armed page is made read-only. The first write to it
// faults; the kernel's SIGSEGV handler hands it here (host_memory::
// HandleWriteFault), the page is made writable, queued, and the write resumes.
// Collect drains the queue: no syscall and no scan, so the cost is paid per
// written page rather than per question. A written page stays open until a
// cache arms it again. The host kernel does not fault on a protected page but
// fails the call, so code that lets it write guest memory (read(), recv())
// announces the range first (host_memory::BeforeHostWrite).
//
// DELTA_GPU_WT_SCAN=1 (scan mode): userfaultfd async write protection plus
// the PAGEMAP_SCAN ioctl (Linux 6.7). Faults resolve in the kernel and a scan
// finds the written pages; each scan costs per mapping walked.
//
// Only CPU writes through the armed mapping are seen. A write through another
// mapping of the same memory, or by a device into imported pages, must be
// announced with NoteWrite.
class WriteTracker {
 public:
  using Range = base::Pair<u64, u64>;  // [first, end), page aligned

  // False when the host cannot track writes; every other call is then a no-op.
  bool Enable();
  bool enabled() const { return mode_ != Mode::kOff; }

  // Arms the pages of [base, base+bytes). False when they cannot be tracked,
  // or are written too often to be worth it.
  bool Arm(u64 base, u64 bytes);

  // Appends every armed range written (or remapped) since the last call, and
  // arms it again. Announced writes are included.
  void Collect(base::Vector<Range>& out);

  // Thread safe. A write the tracker cannot see (another mapping, a device).
  void NoteWrite(u64 base, u64 bytes);
  // Thread safe. The guest mapped or unmapped [base, base+bytes): the pages
  // there are no longer the ones that were armed, and count as written.
  void NoteRemap(u64 base, u64 bytes);

  // Once a frame: pages reported more than a few times in the frame are hot
  // CPU data, and a fault on every write to them costs more than copying.
  void EndFrame();

  u64 armed_bytes() const { return armed_bytes_; }
  u64 collects() const { return collects_; }
  u64 written_pages() const { return written_pages_; }
  u64 runs() const { return armed_.size(); }
  u64 collect_ns() const { return collect_ns_; }

 private:
  enum class Mode { kOff, kScan, kFault };

  bool EnableScan();
  bool EnableFault();
  bool ArmFault(u64 first, u64 end);
  void DisarmFault(u64 first, u64 end);
  void TakeFaulted(base::Vector<Range>& out);
  bool Register(u64 first, u64 end);
  bool AllArmed(u64 first, u64 end) const;
  // Mirrors armed_ into the guest page table, one run at a time.
  void MarkArmed(u64 first, u64 end, bool armed);
  u64 EraseArmed(u64 first, u64 end);
  bool ArmRange(u64 first, u64 end);
  void Disarm(u64 first, u64 end);
  bool Scan(u64 first, u64 end, base::Vector<Range>& out);
  void Drain(base::Vector<Range>& out);
  static long MinorFaults();

  Mode mode_ = Mode::kOff;
  int uffd_ = -1;
  int pagemap_ = -1;
  base::Map<u64, u64> armed_;       // first -> end, coalesced
  base::Map<u64, u64> registered_;  // first -> end, coalesced
  u64 armed_bytes_ = 0;
  base::Mutex noted_lock_;
  base::Vector<Range> noted_, remapped_;
  // Pages reported this frame; their counts live in the guest page table.
  base::Vector<u64> reported_;
  u32 frame_ = 1;
  u64 collects_ = 0, written_pages_ = 0, collect_ns_ = 0;
  long faults_at_scan_ = -1;
};

WriteTracker& GuestWriteTracker();

inline void NoteGuestWrite(u64 base, u64 bytes) {
  GuestWriteTracker().NoteWrite(base, bytes);
}

inline void NoteGuestRemap(u64 base, u64 bytes) {
  GuestWriteTracker().NoteRemap(base, bytes);
}

}  // namespace gpu
