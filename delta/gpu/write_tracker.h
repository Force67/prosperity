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

class WriteTracker {
 public:
  using Range = base::Pair<u64, u64>;

  bool Enable();
  void Reset();
  bool enabled() const { return mode_ != Mode::kOff; }

  bool Arm(u64 base, u64 bytes, bool hot = false);

  size_t Collect(base::Vector<Range>& out);

  void NoteWrite(u64 base, u64 bytes);
  void Release(u64 base, u64 bytes);
  void NoteRemap(u64 base, u64 bytes);

  void EndFrame();
  struct Policy {
    bool (*arm_filter)(u64 first, u64 end) = nullptr;
    void (*on_remap)(u64 base, u64 bytes) = nullptr;
    void (*before_collect)() = nullptr;
  };
  void SetPolicy(const Policy& policy) { policy_ = policy; }
  static void SetOwner(bool (*resolve)(u64 first, u64 end),
                       bool (*owns)(u64 first, u64 end));
  static void OpenedByOwner(u64 first, u64 end);

  u64 armed_bytes() const { return armed_bytes_; }
  u64 collects() const { return collects_; }
  u64 written_pages() const { return written_pages_; }
  u64 runs() const { return armed_.size(); }
  u64 collect_ns() const { return collect_ns_; }

 private:
  enum class Mode { kOff, kScan, kFault };

  bool EnableScan();
  bool EnableFault();
  bool ArmFault(u64 first, u64 end, bool hot);
  void DisarmFault(u64 first, u64 end);
  void TakeFaulted(base::Vector<Range>& out);
  bool Register(u64 first, u64 end);
  bool AllArmed(u64 first, u64 end) const;
  void MarkArmed(u64 first, u64 end, bool armed);
  u64 EraseArmed(u64 first, u64 end);
  bool ArmRange(u64 first, u64 end, bool hot = false);
  void Disarm(u64 first, u64 end);
  bool Scan(u64 first, u64 end, base::Vector<Range>& out);
  size_t Drain(base::Vector<Range>& out);
  static long MinorFaults();

  Mode mode_ = Mode::kOff;
  Policy policy_;
  int uffd_ = -1;
  int pagemap_ = -1;
  base::Map<u64, u64> armed_;
  base::Map<u64, u64> registered_;
  u64 armed_bytes_ = 0;
  base::Mutex noted_lock_;
  base::Vector<Range> noted_, remapped_;
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
