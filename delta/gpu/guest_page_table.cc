/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#include "gpu/guest_page_table.h"
#include "guest/session.h"

#include <sys/mman.h>
#include "base/math/value_bounds.h"

namespace gpu {

GuestPageTable& GuestPages() {
  static GuestPageTable* table = new GuestPageTable;
  return *table;
}

namespace {
base::Atomic<bool (*)(u64, u64)> g_owner_holds{nullptr};
}  // namespace

bool GuestOwnerHolds(u64 first, u64 end) {
  const auto holds = g_owner_holds.load(base::memory_order_acquire);
  return holds && holds(first, end);
}

void SetGuestOwnerHolds(bool (*holds)(u64, u64)) {
  g_owner_holds.store(holds, base::memory_order_release);
}

void GuestPageTable::Reset() {
  for (auto& entry : chunks_)
    if (auto* chunk = entry.exchange(nullptr))
      munmap(chunk, sizeof(Chunk));
}

namespace {
const guest::SessionReset g_session_reset([] { GuestPages().Reset(); });
}  // namespace

GuestPageTable::Chunk* GuestPageTable::ChunkAt(u64 address) {
  const u64 index = address >> kChunkShift;
  if (index >= kChunks)
    return nullptr;
  Chunk* chunk = chunks_[index].load(base::memory_order_acquire);
  if (chunk)
    return chunk;
  void* p = mmap(nullptr, sizeof(Chunk), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED)
    return nullptr;
  Chunk* fresh = static_cast<Chunk*>(p);
  if (chunks_[index].compare_exchange_strong(chunk, fresh,
                                             base::memory_order_acq_rel))
    return fresh;
  munmap(p, sizeof(Chunk));
  return chunk;
}

void GuestPageTable::AddCount(Count count, u64 first, u64 end, int delta) {
  end = base::Min<u64>(end, kChunks << kChunkShift);
  if (first >= end)
    return;
  const u32 shift = UnitShift(count);
  for (u64 unit = first >> shift; unit <= (end - 1) >> shift; unit++) {
    Chunk* chunk = ChunkAt(unit << shift);
    if (!chunk)
      return;
    Slot(*chunk, count, unit)
        .fetch_add(static_cast<u16>(delta), base::memory_order_relaxed);
  }
}

bool GuestPageTable::CountClear(Count count, u64 first, u64 end) const {
  if (first >= end)
    return true;
  if ((end - 1) >> kChunkShift >= kChunks)
    return false;
  const u32 shift = UnitShift(count);
  const u64 per_chunk = 1ull << (kChunkShift - shift);
  const u64 last = (end - 1) >> shift;
  for (u64 unit = first >> shift; unit <= last;) {
    const Chunk* chunk = FindChunk(unit << shift);
    if (!chunk) {
      unit = (unit | (per_chunk - 1)) + 1;
      continue;
    }
    if (Slot(const_cast<Chunk&>(*chunk), count, unit)
            .load(base::memory_order_relaxed))
      return false;
    unit++;
  }
  return true;
}

}  // namespace gpu
