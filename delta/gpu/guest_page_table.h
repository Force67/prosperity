/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/atomic.h"

namespace gpu {

class GuestPageTable {
 public:
  struct Page {
    base::Atomic<u32> readable{0};
    base::Atomic<u8> armed{0};
    base::Atomic<u8> ever_armed{0};
    u16 reports = 0;
    u32 report_frame = 0;
    u32 previous_report = 0;
    u32 volatile_until = 0;
    u8 kept = 0;
  };
  static constexpr u32 kUnreadable = 1u << 31;
  static constexpr u32 kPageShift = 12;
  static constexpr u32 kBlockShift = 16;

  void Reset();

  Page* At(u64 address);
  const Page* Find(u64 address) const;

  void AddCsDirty(u64 first, u64 end, int delta) {
    AddCount(kCsDirty, first, end, delta);
  }
  bool CsClear(u64 first, u64 end) const {
    return CountClear(kCsDirty, first, end);
  }
  void AddPendingWrite(u64 first, u64 end, int delta) {
    AddCount(kPendingWrite, first, end, delta);
  }
  bool PendingWriteClear(u64 first, u64 end) const {
    return CountClear(kPendingWrite, first, end);
  }

 private:
  static constexpr u32 kChunkShift = 30;
  static constexpr u64 kChunks = 1ull << (47 - kChunkShift);
  static constexpr u64 kPagesPerChunk = 1ull << (kChunkShift - kPageShift);
  static constexpr u64 kBlocksPerChunk = 1ull << (kChunkShift - kBlockShift);
  enum Count { kCsDirty, kPendingWrite, kCounts };
  struct Chunk {
    Page pages[kPagesPerChunk];
    base::Atomic<u16> counts[kCounts][kBlocksPerChunk];
  };
  void AddCount(Count count, u64 first, u64 end, int delta);
  bool CountClear(Count count, u64 first, u64 end) const;

  Chunk* ChunkAt(u64 address);
  const Chunk* FindChunk(u64 address) const {
    const u64 index = address >> kChunkShift;
    return index < kChunks ? chunks_[index].load(base::memory_order_acquire)
                           : nullptr;
  }

  base::Atomic<Chunk*> chunks_[kChunks] = {};
};

GuestPageTable& GuestPages();

bool GuestOwnerHolds(u64 first, u64 end);
void SetGuestOwnerHolds(bool (*holds)(u64 first, u64 end));

inline const GuestPageTable::Page* GuestPageTable::Find(u64 address) const {
  const Chunk* chunk = FindChunk(address);
  return chunk ? &chunk->pages[(address >> kPageShift) & (kPagesPerChunk - 1)]
               : nullptr;
}

inline GuestPageTable::Page* GuestPageTable::At(u64 address) {
  Chunk* chunk = ChunkAt(address);
  return chunk ? &chunk->pages[(address >> kPageShift) & (kPagesPerChunk - 1)]
               : nullptr;
}

}  // namespace gpu
