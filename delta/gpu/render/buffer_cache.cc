/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#include "gpu/render/buffer_cache.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>
#include <unordered_map>
#include <vector>

#include <utl/options.h>

#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/renderer.h"

namespace gpu::render {
namespace {

DELTA_OPTION(u32, kBufferCacheMb, "DELTA_GPU_BUFFER_CACHE_MB", 512);

constexpr u64 kBlockBytes = 64ull << 20;
constexpr u64 kAlign = 256;
// Frames a released copy may still be read by (the ring's frame slots plus
// the one being recorded).
constexpr int kRetireFrames = 3;
// A copy no draw asked for in this many frames gives its memory back.
constexpr int kIdleFrames = 120;
constexpr u32 kIndexShift = 16;  // 64 KiB blocks for invalidation lookups

struct Block {
  rhi::Buffer* buffer = nullptr;
  u8* map = nullptr;
  std::map<u64, u64> free;  // offset -> bytes, coalesced
};

struct Entry {
  u32 block = 0;
  u64 offset = 0;
  u64 bytes = 0;
  int last_used = 0;
};

struct Retired {
  u32 block;
  u64 offset;
  u64 bytes;
  int frame;
};

std::vector<Block> g_blocks;
std::unordered_map<u64, Entry> g_entries;  // by guest base
std::unordered_map<u64, std::vector<u64>> g_index;  // 64 KiB block -> bases
std::vector<Retired> g_retired;
u64 g_live_bytes = 0;

void Free(Block& block, u64 offset, u64 bytes) {
  auto next = block.free.lower_bound(offset);
  if (next != block.free.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      offset = prev->first;
      bytes += prev->second;
      block.free.erase(prev);
    }
  }
  if (next != block.free.end() && offset + bytes == next->first) {
    bytes += next->second;
    block.free.erase(next);
  }
  block.free.emplace(offset, bytes);
}

bool Allocate(u64 bytes, u32& block_index, u64& offset) {
  bytes = (bytes + kAlign - 1) & ~(kAlign - 1);
  if (bytes > kBlockBytes)
    return false;
  for (u32 i = 0; i < g_blocks.size(); i++) {
    for (auto it = g_blocks[i].free.begin(); it != g_blocks[i].free.end();
         ++it) {
      if (it->second < bytes)
        continue;
      const u64 at = it->first, left = it->second - bytes;
      g_blocks[i].free.erase(it);
      if (left)
        g_blocks[i].free.emplace(at + bytes, left);
      block_index = i;
      offset = at;
      return true;
    }
  }
  if ((g_blocks.size() + 1) * kBlockBytes > u64(kBufferCacheMb) << 20)
    return false;
  rhi::BufferDesc desc;
  desc.size = kBlockBytes;
  desc.usage = rhi::kBufferVertex;
  desc.memory = rhi::MemoryKind::kUpload;
  desc.name = "guest buffer cache";
  Block block;
  block.buffer = Device().CreateBuffer(desc);
  block.map = block.buffer ? static_cast<u8*>(block.buffer->mapped()) : nullptr;
  if (!block.map) {
    if (block.buffer)
      Device().Destroy(block.buffer);
    return false;
  }
  block.free.emplace(bytes, kBlockBytes - bytes);
  g_blocks.push_back(std::move(block));
  block_index = static_cast<u32>(g_blocks.size() - 1);
  offset = 0;
  return true;
}

void Retire(const Entry& e) {
  g_retired.push_back({e.block, e.offset,
                       (e.bytes + kAlign - 1) & ~(kAlign - 1), g_frame.num});
  g_live_bytes -= e.bytes;
}

void Index(u64 base, u64 bytes) {
  for (u64 b = base >> kIndexShift; b <= (base + bytes - 1) >> kIndexShift; b++)
    g_index[b].push_back(base);
}

}  // namespace

bool FindCachedBuffer(u64 base, u64 bytes, CachedBuffer& out) {
  auto it = g_entries.find(base);
  if (it == g_entries.end() || it->second.bytes < bytes ||
      CsRangeDirtyOverlapping(base, bytes))
    return false;
  it->second.last_used = g_frame.num;
  out = {g_blocks[it->second.block].buffer, it->second.offset};
  return true;
}

bool CacheGuestBuffer(u64 base, u64 bytes, CachedBuffer& out) {
  if (!bytes)
    return false;
  u32 block;
  u64 offset;
  if (!Allocate(bytes, block, offset))
    return false;
  if (auto old = g_entries.find(base); old != g_entries.end()) {
    Retire(old->second);
    g_entries.erase(old);
  }
  std::memcpy(g_blocks[block].map + offset, reinterpret_cast<const void*>(base),
              bytes);
  g_entries[base] = {block, offset, bytes, g_frame.num};
  g_live_bytes += bytes;
  Index(base, bytes);
  out = {g_blocks[block].buffer, offset};
  return true;
}

void InvalidateCachedBuffers(u64 first, u64 end) {
  if (g_entries.empty() || end <= first)
    return;
  const auto drop = [&](const std::vector<u64>& bases) {
    for (u64 base : bases) {
      auto it = g_entries.find(base);
      if (it == g_entries.end() || base >= end ||
          first >= base + it->second.bytes)
        continue;
      Retire(it->second);
      g_entries.erase(it);
    }
  };
  // A remapped reservation can span gigabytes: walk the smaller side.
  const u64 first_block = first >> kIndexShift;
  const u64 last_block = (end - 1) >> kIndexShift;
  if (last_block - first_block + 1 > g_index.size()) {
    for (const auto& [block, bases] : g_index)
      if (block >= first_block && block <= last_block)
        drop(bases);
    return;
  }
  for (u64 b = first_block; b <= last_block; b++)
    if (auto found = g_index.find(b); found != g_index.end())
      drop(found->second);
}

void BufferCacheEndFrame() {
  for (auto it = g_entries.begin(); it != g_entries.end();) {
    if (it->second.last_used + kIdleFrames < g_frame.num) {
      Retire(it->second);
      it = g_entries.erase(it);
    } else {
      ++it;
    }
  }
  std::erase_if(g_retired, [](const Retired& r) {
    if (r.frame + kRetireFrames > g_frame.num)
      return false;
    Free(g_blocks[r.block], r.offset, r.bytes);
    return true;
  });
  // The index only grows between rebuilds; rebuild it from the live copies
  // now and then instead of unindexing each drop.
  if (g_frame.num % 64 == 0) {
    g_index.clear();
    for (const auto& [base, e] : g_entries)
      Index(base, e.bytes);
  }
}

u64 CachedBufferBytes() {
  return g_live_bytes;
}

}  // namespace gpu::render
