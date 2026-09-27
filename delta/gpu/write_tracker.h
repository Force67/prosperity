/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace gpu {

// Which guest pages were written since a cache copied them, for caches that
// keep a copy of guest memory longer than one submission.
//
// Linux userfaultfd write protection in async mode plus the PAGEMAP_SCAN
// ioctl (Linux 6.7): an armed page takes one kernel-resolved fault on its
// first write, writes by the kernel itself (read() into a buffer) included,
// and a scan reports it and arms it again. No signal handler is involved.
//
// Only CPU writes through the armed mapping are seen. A write through another
// mapping of the same memory, or by a device into imported pages, must be
// announced with NoteWrite.
class WriteTracker {
 public:
  using Range = std::pair<u64, u64>;  // [first, end), page aligned

  // False when the host cannot track writes; every other call is then a no-op.
  bool Enable();
  bool enabled() const { return uffd_ >= 0; }

  // Arms the pages of [base, base+bytes). False when they cannot be tracked,
  // or are written too often to be worth it.
  bool Arm(u64 base, u64 bytes);

  // Appends every armed range written (or remapped) since the last call, and
  // arms it again. Announced writes are included.
  void Collect(std::vector<Range>& out);

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
  bool Register(u64 first, u64 end);
  bool AllArmed(u64 first, u64 end) const;
  // Mirrors armed_ into the guest page table, one run at a time.
  void MarkArmed(u64 first, u64 end, bool armed);
  u64 EraseArmed(u64 first, u64 end);
  bool ArmRange(u64 first, u64 end);
  void Disarm(u64 first, u64 end);
  bool Scan(u64 first, u64 end, std::vector<Range>& out);
  void Drain(std::vector<Range>& out);
  static long MinorFaults();

  int uffd_ = -1;
  int pagemap_ = -1;
  std::map<u64, u64> armed_;       // first -> end, coalesced
  std::map<u64, u64> registered_;  // first -> end, coalesced
  u64 armed_bytes_ = 0;
  std::mutex noted_lock_;
  std::vector<Range> noted_, remapped_;
  // Pages reported this frame; their counts live in the guest page table.
  std::vector<u64> reported_;
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
