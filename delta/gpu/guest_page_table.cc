/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#include "gpu/guest_page_table.h"
#include "guest/session.h"

#include <sys/mman.h>
#include "base/math/value_bounds.h"

namespace gpu {

GuestPageTable& GuestPages() {
  // Never destroyed: the renderer thread may still ask at exit.
  static GuestPageTable* table = new GuestPageTable;
  return *table;
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
  // Zero pages are default records. Reserved only: a chunk costs what its
  // written records occupy.
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

void GuestPageTable::AddCsDirty(u64 first, u64 end, int delta) {
  end = base::Min<u64>(end, kChunks << kChunkShift);
  if (first >= end)
    return;
  for (u64 block = first >> kBlockShift; block <= (end - 1) >> kBlockShift;
       block++) {
    Chunk* chunk = ChunkAt(block << kBlockShift);
    if (!chunk)
      return;
    chunk->cs_dirty[block & (kBlocksPerChunk - 1)].fetch_add(
        static_cast<u16>(delta), base::memory_order_relaxed);
  }
}

bool GuestPageTable::CsClear(u64 first, u64 end) const {
  if (first >= end)
    return true;
  if ((end - 1) >> kChunkShift >= kChunks)
    return false;
  const u64 last = (end - 1) >> kBlockShift;
  for (u64 block = first >> kBlockShift; block <= last;) {
    const Chunk* chunk = FindChunk(block << kBlockShift);
    if (!chunk) {  // nothing was ever dirty in this chunk
      block = (block | (kBlocksPerChunk - 1)) + 1;
      continue;
    }
    if (chunk->cs_dirty[block & (kBlocksPerChunk - 1)].load(
            base::memory_order_relaxed))
      return false;
    block++;
  }
  return true;
}

}  // namespace gpu
