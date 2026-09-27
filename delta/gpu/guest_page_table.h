/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include <base/atomic.h>


namespace gpu {

// What the renderer keeps per 4 KiB page of guest memory, in one directly
// indexed table: the per-draw paths ask about the same pages thousands of
// times a frame, and a lookup here costs no hashing and never collides.
//
// Two levels over the 47-bit user address space. The first time any page of a
// 1 GiB chunk is asked about, the chunk's records are reserved; the host then
// commits them a table page at a time as they are written.
class GuestPageTable {
 public:
  struct Page {
    // The MemoryGeneration() the page was proven readable at, or with
    // kUnreadable set, proven unreadable at. Any thread; a stale answer only
    // costs a probe.
    base::Atomic<u32> readable{0};
    // Write tracker state, owned by the thread that owns the tracker.
    bool armed = false;
    u16 reports = 0;          // write reports in report_frame
    u32 report_frame = 0;     // frame of the latest write report
    u32 previous_report = 0;  // frame of the report before that frame
    u32 volatile_until = 0;   // frame before which the page is not armed
  };
  static constexpr u32 kUnreadable = 1u << 31;
  static constexpr u32 kPageShift = 12;
  static constexpr u32 kBlockShift = 16;

  // The record for the page holding `address`, created on first use. Null
  // above user space.
  Page* At(u64 address);
  // As At, but null where nothing was ever created: a read of a page nobody
  // wrote has its answer in the default record.
  const Page* Find(u64 address) const;

  // GPU-dirty compute ranges touching each 64 KiB block. Written by the
  // renderer's owner, read from any thread.
  void AddCsDirty(u64 first, u64 end, int delta);
  // No GPU-dirty compute range touches [first, end).
  bool CsClear(u64 first, u64 end) const;

 private:
  static constexpr u32 kChunkShift = 30;
  static constexpr u64 kChunks = 1ull << (47 - kChunkShift);
  static constexpr u64 kPagesPerChunk = 1ull << (kChunkShift - kPageShift);
  static constexpr u64 kBlocksPerChunk = 1ull << (kChunkShift - kBlockShift);
  struct Chunk {
    Page pages[kPagesPerChunk];
    base::Atomic<u16> cs_dirty[kBlocksPerChunk];
  };

  Chunk* ChunkAt(u64 address);
  const Chunk* FindChunk(u64 address) const {
    const u64 index = address >> kChunkShift;
    return index < kChunks ? chunks_[index].load(base::memory_order_acquire)
                           : nullptr;
  }

  base::Atomic<Chunk*> chunks_[kChunks] = {};
};

GuestPageTable& GuestPages();

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
