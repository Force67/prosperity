#include "gpu/render/memory/guest_direct.h"
#include "guest/session.h"

#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/compute/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/renderer.h"
#include "gpu/write_tracker.h"
#include "host_memory/host_memory.h"
#include "options/options.h"

namespace gpu::render {
namespace {
DELTA_OPTION(bool, kDirect, "DELTA_GPU_DIRECT", true);

constexpr u64 kBlocks = kDirectBlocks;
constexpr u64 kFlatBlocks = gcn::kDirectFlatBlocks;
constexpr u64 kDirectoryBlocks = gcn::kDirectDirectoryBlocks;
constexpr u64 kDirectoryEntries = gcn::kDirectDirectoryEntries;
constexpr u64 kTableBytes = gcn::kDirectTableEntries * sizeof(u64);
constexpr u64 kChunk = 2ull << 20;

struct Mapping {
  u64 start, end, offset;
  bool readable = true, writable = true;
};
constexpr u64 kPage = 1ull << gcn::kDirectPageShift;
constexpr u32 kLeafPages = 1u << (kDirectBlockShift - gcn::kDirectPageShift);
struct PageEntry {
  u64 address = 0, end = 0;
};
constexpr u32 kLeafBytes = kLeafPages * sizeof(PageEntry);
constexpr u32 kLeavesPerSlab = 256;
struct Leaf {
  u32 slot;
  PageEntry pages[kLeafPages];
};
struct HostImport {
  u64 end, identity, snapshot;
  rhi::Buffer* buffer;
  bool readonly;
};
struct Retired {
  rhi::Buffer* buffer;
  u32 slot;
  int frame;
  u64 submission;
};
struct Remap {
  u64 base, end;
};
base::Mutex g_remap_lock;

struct BindImport {
  rhi::Buffer* buffer;
  u64 bytes;
};
struct HighTable {
  u64* entries;
  u64* saved;
  base::Atomic<u8>* redirected;
};
struct Block {
  u64* entry = nullptr;
  u64* saved = nullptr;
  base::Atomic<u8>* redirected = nullptr;
};

// Direct guest memory state of one guest session; the session reset clears
// all of it.
struct GuestDirectState {
  base::Vector<Mapping> host_maps;
  base::Map<u64, HostImport> host_imports;
  base::HashMap<u64, Leaf> leaves;
  base::Vector<rhi::Buffer*> leaf_slabs;
  base::Vector<u32> free_leaves;
  base::Vector<Retired> retired;
  u32 next_leaf = 0;
  u64 map_snapshot = 0;
  base::Vector<Remap> remaps;
  base::Vector<Remap> maps_dirty;
  u8* alias = nullptr;
  u64 alias_size = 0;
  rhi::Buffer* table = nullptr;
  rhi::Buffer* misses = nullptr;
  base::HashMap<u64, rhi::Buffer*> chunks;
  base::HashMap<u64, BindImport> binds;
  base::Vector<Mapping> maps;
  base::HashSet<u64> filled;
  u64* saved = nullptr;
  base::Atomic<u8>* redirected = nullptr;
  base::Atomic<HighTable*> high[kDirectoryEntries]{};
  u64 periodic_at = 0;
  int init_state = -1;
  u64 maps_read_at = ~0ull;
  bool maps_changed = false;
  u64 alias_probe_at = ~0ull;
};
GuestDirectState g_direct;

int FindDmemFd() {
  DIR* dir = opendir("/proc/self/fd");
  if (!dir)
    return -1;
  int found = -1;
  while (dirent* e = readdir(dir)) {
    char path[64], target[128];
    std::snprintf(path, sizeof(path), "/proc/self/fd/%s", e->d_name);
    const ssize_t n = readlink(path, target, sizeof(target) - 1);
    if (n <= 0)
      continue;
    target[n] = 0;
    if (std::strstr(target, "memfd:delta_dmem")) {
      found = std::atoi(e->d_name);
      break;
    }
  }
  closedir(dir);
  return found;
}

void AttachDirectAlias(bool force = false) {
  if (g_direct.alias)
    return;
  const u64 generation =
      gpu::GuestMappingGeneration().load(base::memory_order_acquire);
  if (!force && g_direct.alias_probe_at == generation)
    return;
  g_direct.alias_probe_at = generation;
  const int fd = FindDmemFd();
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0)
    return;
  const u64 size = static_cast<u64>(st.st_size);
  void* alias = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_NORESERVE, fd, 0);
  if (alias == MAP_FAILED)
    return;
  g_direct.alias = static_cast<u8*>(alias);
  g_direct.alias_size = size;
  g_direct.maps_read_at = ~0ull;
}

bool MapsDirtyIn(u64 lo, u64 hi) {
  base::LockGuard lock(g_remap_lock);
  if (g_direct.maps_dirty.size() > 4096)
    return true;
  for (const Remap& r : g_direct.maps_dirty)
    if (r.base < hi && lo < r.end)
      return true;
  return false;
}

void ReadMaps(bool force = false, u64 lo = 0, u64 hi = ~0ull) {
  const u64 generation =
      gpu::GuestMappingGeneration().load(base::memory_order_acquire);
  if (!force && g_direct.maps_read_at == generation)
    return;
  if (!force && g_direct.maps_read_at != ~0ull && hi != ~0ull &&
      !MapsDirtyIn(lo, hi))
    return;
  {
    base::LockGuard lock(g_remap_lock);
    g_direct.maps_dirty.clear();
  }
  FILE* f = std::fopen("/proc/self/maps", "r");
  if (!f)
    return;
  g_direct.maps_read_at = generation;
  g_direct.maps_changed = true;
  ++g_direct.map_snapshot;
  g_direct.host_maps.clear();
  g_direct.maps.clear();
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long start, end, offset;
    char perms[8];
    if (std::sscanf(line, "%llx-%llx %7s %llx", &start, &end, perms, &offset) !=
        4)
      continue;
    if (!std::strstr(line, "memfd:delta_dmem")) {
      if (GuestDirectIndex(start) != ~0ull)
        g_direct.host_maps.push_back(
            {start, end, offset, perms[0] == 'r', perms[1] == 'w'});
      continue;
    }
    if (!g_direct.maps.empty() && g_direct.maps.back().end == start &&
        g_direct.maps.back().offset + (start - g_direct.maps.back().start) ==
            offset) {
      g_direct.maps.back().end = end;
      continue;
    }
    g_direct.maps.push_back({start, end, offset});
  }
  std::fclose(f);
}

bool DirtyTouchesGpu() {
  base::Vector<Remap> dirty;
  {
    base::LockGuard lock(g_remap_lock);
    if (g_direct.maps_dirty.size() > 4096)
      return true;
    dirty = g_direct.maps_dirty;
  }
  for (const Remap& r : dirty) {
    const u64 first = r.base & ~((1ull << kDirectBlockShift) - 1);
    if (((r.end - first) >> kDirectBlockShift) < g_direct.filled.size()) {
      for (u64 b = first; b < r.end; b += 1ull << kDirectBlockShift)
        if (g_direct.filled.count(b))
          return true;
    } else {
      for (u64 b : g_direct.filled)
        if (b < r.end && r.base < b + (1ull << kDirectBlockShift))
          return true;
    }
    auto it = g_direct.host_imports.upper_bound(r.base);
    if (it != g_direct.host_imports.begin())
      --it;
    if (it != g_direct.host_imports.end() && it->first < r.end &&
        r.base < it->second.end)
      return true;
    if (++it != g_direct.host_imports.end() && it->first < r.end)
      return true;
  }
  return false;
}

const Mapping* FindMappingIn(const base::Vector<Mapping>& maps, u64 address) {
  u32 lo = 0, hi = maps.size();
  while (lo < hi) {
    const u32 mid = (lo + hi) / 2;
    if (maps[mid].end <= address)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo < maps.size() && maps[lo].start <= address ? &maps[lo] : nullptr;
}

const Mapping* FindMapping(u64 address) {
  return FindMappingIn(g_direct.maps, address);
}

rhi::Buffer* ChunkAt(u64 offset) {
  const u64 chunk = offset & ~(kChunk - 1);
  auto it = g_direct.chunks.find(chunk);
  if (it != g_direct.chunks.end())
    return it->second;
  rhi::BufferDesc desc;
  desc.size = base::Min(kChunk + (1ull << kDirectBlockShift),
                        g_direct.alias_size - chunk);
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress | rhi::kBufferCopySrc |
               rhi::kBufferCopyDst;
  desc.host_pointer = g_direct.alias + chunk;
  rhi::Buffer* buf = Device().CreateBuffer(desc);
  g_direct.chunks.insert_or_assign(chunk, buf);
  return buf;
}

u64* Table() {
  return reinterpret_cast<u64*>(g_direct.table->mapped());
}

Block BlockAt(u64 index, bool create = false) {
  if (index < kFlatBlocks)
    return {Table() + index, g_direct.saved + index,
            g_direct.redirected + index};
  const u64 directory = index / kDirectoryBlocks;
  auto* high = g_direct.high[directory].load(base::memory_order_acquire);
  if (!high && create) {
    rhi::BufferDesc desc;
    desc.size = kDirectoryBlocks * sizeof(u64);
    desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
    desc.memory = rhi::MemoryKind::kUploadDevice;
    desc.name = "guest high address table";
    auto* buffer = Device().CreateBuffer(desc);
    if (!buffer || !buffer->mapped() || !buffer->address()) {
      if (buffer)
        Device().Destroy(buffer);
      return {};
    }
    void* saved = mmap(nullptr, kDirectoryBlocks * 9, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (saved == MAP_FAILED) {
      Device().Destroy(buffer);
      return {};
    }
    std::memset(buffer->mapped(), 0, desc.size);
    high = new HighTable{
        reinterpret_cast<u64*>(buffer->mapped()), static_cast<u64*>(saved),
        reinterpret_cast<base::Atomic<u8>*>(static_cast<u8*>(saved) +
                                            kDirectoryBlocks * sizeof(u64))};
    g_direct.high[directory].store(high, base::memory_order_release);
    __atomic_store_n(&Table()[kFlatBlocks + directory],
                     buffer->address() | gcn::kDirectDirectory,
                     __ATOMIC_RELEASE);
  }
  if (!high)
    return {};
  const u64 inner = index % kDirectoryBlocks;
  return {high->entries + inner, high->saved + inner, high->redirected + inner};
}

void Retire(rhi::Buffer* buffer, u32 slot = ~0u) {
  g_direct.retired.push_back(
      {buffer, slot, g_frame.num, Device().LastSubmission()});
}

void ReapImports() {
  for (size_t i = 0; i < g_direct.retired.size();) {
    const auto& retired = g_direct.retired[i];
    if (retired.frame + 2 > g_frame.num ||
        !Device().IsComplete(retired.submission) ||
        (retired.buffer && MirrorUsesImport(retired.buffer))) {
      ++i;
      continue;
    }
    if (retired.buffer)
      Device().Destroy(retired.buffer);
    else
      g_direct.free_leaves.push_back(retired.slot);
    g_direct.retired.erase(g_direct.retired.begin() + i);
  }
}

void InvalidateCopiedImports() {
  base::Vector<Remap> remaps;
  {
    base::LockGuard lock(g_remap_lock);
    remaps.swap(g_direct.remaps);
  }
  if (remaps.empty())
    return;
  for (const auto& remap : remaps) {
    auto it = g_direct.host_imports.upper_bound(remap.base);
    if (it != g_direct.host_imports.begin())
      --it;
    while (it != g_direct.host_imports.end() && it->first < remap.end) {
      if (it->second.readonly && remap.base < it->second.end) {
        Retire(it->second.buffer);
        it = g_direct.host_imports.erase(it);
        g_direct.maps_changed = true;
      } else {
        ++it;
      }
    }
  }
}

bool WritableMapping(const Mapping& map, u64 start, u64 end) {
  if (map.writable)
    return true;
  for (u64 page = start; page < end; page += kPage) {
    const auto* state = gpu::GuestPages().Find(page);
    if (!state || !state->armed.load(base::memory_order_acquire))
      return false;
  }
  return true;
}

bool ImportValid(u64 start, const HostImport& imported) {
  const u64 identity = host_memory::MemoryMappingIdentity(
      reinterpret_cast<void*>(start), imported.end - start);
  if (!identity || identity != imported.identity)
    return false;
  for (u64 at = start; at < imported.end;) {
    const Mapping* map = FindMappingIn(g_direct.host_maps, at);
    if (!map)
      return false;
    const u64 end = base::Min(imported.end, map->end);
    if (imported.readonly && WritableMapping(*map, at, end))
      return false;
    at = end;
  }
  return true;
}

void PruneImports() {
  for (auto it = g_direct.host_imports.begin();
       it != g_direct.host_imports.end();) {
    if (!ImportValid(it->first, it->second)) {
      Retire(it->second.buffer);
      it = g_direct.host_imports.erase(it);
    } else {
      it->second.snapshot = g_direct.map_snapshot;
      ++it;
    }
  }
}

HostImport* HostImportAt(u64 address, u64* base, bool retry = true) {
  const Mapping* map = FindMappingIn(g_direct.host_maps, address);
  if (!map)
    return nullptr;
  auto prior = g_direct.host_imports.upper_bound(address);
  if (prior != g_direct.host_imports.begin()) {
    --prior;
    auto& imported = prior->second;
    if (address < imported.end) {
      if (imported.snapshot == g_direct.map_snapshot) {
        *base = prior->first;
        return &imported;
      }
      if (ImportValid(prior->first, imported)) {
        imported.snapshot = g_direct.map_snapshot;
        *base = prior->first;
        return &imported;
      }
      Retire(imported.buffer);
      g_direct.maps_changed = true;
      g_direct.host_imports.erase(prior);
    }
  }
  if (!map->readable)
    return nullptr;
  const u64 chunk = address & ~(kChunk - 1);
  const u64 start = base::Max(chunk, map->start);
  const u64 end =
      base::Min(chunk + kChunk + (1ull << kDirectBlockShift), map->end);
  const u64 identity = host_memory::MemoryMappingIdentity(
      reinterpret_cast<void*>(start), end - start);
  if (!identity)
    return nullptr;
  const bool writable = WritableMapping(*map, start, end);
  rhi::BufferDesc desc;
  desc.size = end - start;
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress | rhi::kBufferCopySrc |
               rhi::kBufferCopyDst;
  desc.name = "guest host read";
  if (writable) {
    const u64 alignment = Device().caps().host_import_alignment;
    if (!alignment || (start % alignment) || (desc.size % alignment))
      return nullptr;
    gpu::GuestWriteTracker().Release(start, desc.size);
    desc.host_pointer = reinterpret_cast<void*>(start);
  } else {
    desc.memory = rhi::MemoryKind::kUploadDevice;
  }
  auto* buffer = Device().CreateBuffer(desc);
  if (!buffer || !buffer->address() || (!writable && !buffer->mapped())) {
    if (buffer)
      Device().Destroy(buffer);
    if (writable && retry) {
      ReadMaps(true);
      return HostImportAt(address, base, false);
    }
    return nullptr;
  }
  if (!writable)
    std::memcpy(buffer->mapped(), reinterpret_cast<const void*>(start),
                desc.size);
  *base = start;
  if (auto previous = g_direct.host_imports.find(start);
      previous != g_direct.host_imports.end()) {
    Retire(previous->second.buffer);
    g_direct.maps_changed = true;
  }
  auto& imported = g_direct.host_imports[start];
  imported = {end, identity, g_direct.map_snapshot, buffer, !writable};
  return &imported;
}

u64 HostEntryFor(u64 block) {
  PageEntry pages[kLeafPages]{};
  bool held = false, failed = false;
  for (u32 page = 0; page < kLeafPages; ++page) {
    const u64 address = block + page * kPage;
    if (const Mapping* direct = FindMapping(address)) {
      const u64 offset = direct->offset + address - direct->start;
      if (offset < g_direct.alias_size) {
        if (auto* chunk = ChunkAt(offset)) {
          const u64 inner = offset & (kChunk - 1);
          pages[page] = {
              chunk->address() + inner,
              base::Min(direct->end, address + chunk->desc().size - inner)};
          held |= chunk->address() != 0;
          failed |= chunk->address() == 0;
        } else {
          failed = true;
        }
      }
      continue;
    }
    u64 base = 0;
    if (auto* imported = HostImportAt(address, &base)) {
      pages[page] = {imported->buffer->address() + address - base,
                     imported->end};
      held = true;
    } else if (const auto* map = FindMappingIn(g_direct.host_maps, address);
               map && map->readable) {
      failed = true;
    }
  }
  auto previous = g_direct.leaves.find(block);
  if (!failed && previous != g_direct.leaves.end() &&
      std::memcmp(previous->second.pages, pages, sizeof(pages)) == 0) {
    const u32 slot = previous->second.slot;
    return (g_direct.leaf_slabs[slot / kLeavesPerSlab]->address() +
            (slot % kLeavesPerSlab) * kLeafBytes) |
           gcn::kDirectLeaf;
  }
  if (previous != g_direct.leaves.end()) {
    Retire(nullptr, previous->second.slot);
    g_direct.leaves.erase(previous);
  }
  if (failed)
    return 0;
  if (!held)
    return gcn::kDirectUnmapped;
  u32 slot;
  if (!g_direct.free_leaves.empty()) {
    slot = g_direct.free_leaves.back();
    g_direct.free_leaves.pop_back();
  } else {
    slot = g_direct.next_leaf;
    if (slot % kLeavesPerSlab == 0) {
      rhi::BufferDesc desc;
      desc.size = kLeavesPerSlab * kLeafBytes;
      desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
      desc.memory = rhi::MemoryKind::kUploadDevice;
      desc.name = "guest page table";
      auto* slab = Device().CreateBuffer(desc);
      if (!slab || !slab->mapped() || !slab->address()) {
        if (slab)
          Device().Destroy(slab);
        return 0;
      }
      g_direct.leaf_slabs.push_back(slab);
    }
    ++g_direct.next_leaf;
  }
  auto* slab = g_direct.leaf_slabs[slot / kLeavesPerSlab];
  const u64 offset = (slot % kLeavesPerSlab) * kLeafBytes;
  std::memcpy(slab->mapped() + offset, pages, sizeof(pages));
  Leaf leaf{slot, {}};
  std::memcpy(leaf.pages, pages, sizeof(pages));
  g_direct.leaves.insert_or_assign(block, leaf);
  return (slab->address() + offset) | gcn::kDirectLeaf;
}

u64 EntryFor(u64 block) {
  const Mapping* m = FindMapping(block);
  if (!m || block + (1ull << kDirectBlockShift) > m->end)
    return HostEntryFor(block);
  const u64 offset = m->offset + (block - m->start);
  if (offset >= g_direct.alias_size)
    return 0;
  rhi::Buffer* chunk = ChunkAt(offset);
  return chunk && chunk->address() ? chunk->address() + (offset & (kChunk - 1))
                                   : 0;
}

void Fill(u64 address) {
  ReadMaps(false, address & ~(kChunk - 1), (address & ~(kChunk - 1)) + kChunk);
  if (GuestDirectIndex(address) == ~0ull)
    return;
  const u64 first = address & ~(kChunk - 1);
  for (u64 block = first; block < first + kChunk;
       block += 1ull << kDirectBlockShift) {
    const u64 index = GuestDirectIndex(block);
    if (index >= kBlocks)
      break;
    const Block target = BlockAt(index, true);
    if (!target.entry)
      continue;
    const u64 entry = EntryFor(block);
    if (target.redirected->load(base::memory_order_acquire)) {
      *target.saved = entry;
    } else {
      const Mapping* host = FindMappingIn(g_direct.host_maps, block);
      const bool cacheable =
          FindMapping(block) ||
          (host &&
           WritableMapping(*host, block, block + (1ull << kDirectBlockShift)));
      const u64 cached = cacheable ? MirrorCacheBlock(block) : 0;
      if (cached)
        DirectRedirect(block, cached);
      else
        *target.entry = entry;
    }
    g_direct.filled.insert(block);
  }
}
}  // namespace

bool InitGuestDirect() {
  if (g_direct.init_state >= 0) {
    if (g_direct.init_state)
      AttachDirectAlias();
    return g_direct.init_state;
  }
  g_direct.init_state = 0;
  if (!kDirect || !Device().caps().host_import ||
      !Device().caps().buffer_address)
    return false;
  AttachDirectAlias();
  rhi::BufferDesc desc;
  desc.size = kTableBytes;
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
  desc.memory = rhi::MemoryKind::kUploadDevice;
  desc.name = "guest direct table";
  g_direct.table = Device().CreateBuffer(desc);
  desc.size = 8 + kDirectMissSlots * 8;
  desc.memory = rhi::MemoryKind::kReadback;
  desc.name = "guest direct misses";
  g_direct.misses = Device().CreateBuffer(desc);
  if (!g_direct.table || !g_direct.misses || !g_direct.table->mapped() ||
      !g_direct.misses->mapped() || !g_direct.table->address() ||
      !g_direct.misses->address())
    return false;
  std::memset(g_direct.table->mapped(), 0, kTableBytes);
  void* saved = mmap(nullptr, kFlatBlocks * 9, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (saved == MAP_FAILED)
    return false;
  g_direct.saved = static_cast<u64*>(saved);
  g_direct.redirected = reinterpret_cast<base::Atomic<u8>*>(
      static_cast<u8*>(saved) + kFlatBlocks * 8);
  std::memset(g_direct.misses->mapped(), 0, 8 + kDirectMissSlots * 8);
  ReadMaps();
  gcn::g_direct_table = g_direct.table->address();
  gcn::g_direct_misses = g_direct.misses->address();
  BASE_LOGI("direct", "guest memory {:#x}+{:#x} through {} views of {} MiB",
            kDirectBase, kDirectSize, g_direct.maps.size(),
            g_direct.alias_size >> 20);
  g_direct.init_state = 1;
  return true;
}

void NoteGuestDirectRemap(u64 base, u64 bytes) {
  if (!bytes || bytes > ~0ull - base)
    return;
  base::LockGuard lock(g_remap_lock);
  g_direct.remaps.push_back({base, base + bytes});
  // Every draw scans this list, and thread stacks come and go back to back:
  // merge touching ranges so it stays short.
  for (Remap& r : g_direct.maps_dirty)
    if (base <= r.end && r.base <= base + bytes) {
      r.base = base::Min(r.base, base);
      r.end = base::Max(r.end, base + bytes);
      return;
    }
  g_direct.maps_dirty.push_back({base, base + bytes});
}

bool InDirectAlias(u64 va) {
  return g_direct.alias && va >= reinterpret_cast<u64>(g_direct.alias) &&
         va < reinterpret_cast<u64>(g_direct.alias) + g_direct.alias_size;
}

void DirectRedirect(u64 block, u64 device_address) {
  const u64 index = GuestDirectIndex(block);
  if (!gcn::g_direct_table || index == ~0ull)
    return;
  const Block target = BlockAt(index, true);
  if (!target.entry)
    return;
  if (!target.redirected->load(base::memory_order_acquire)) {
    ReadMaps(false, block, block + (1ull << kDirectBlockShift));
    *target.saved = EntryFor(block);
    g_direct.filled.insert(block);
  }
  __atomic_store_n(target.entry, device_address, __ATOMIC_RELEASE);
  target.redirected->store(1, base::memory_order_release);
}

void DirectRestore(u64 block) {
  const u64 index = GuestDirectIndex(block);
  if (!gcn::g_direct_table || index == ~0ull)
    return;
  const Block target = BlockAt(index);
  if (!target.entry ||
      !target.redirected->exchange(0, base::memory_order_acq_rel))
    return;
  __atomic_store_n(target.entry, *target.saved, __ATOMIC_RELEASE);
}

rhi::Buffer* DirectImportAt(u64 address, u64* offset, u64* bytes) {
  if (!g_direct.alias && !InitGuestDirect())
    return nullptr;
  ReadMaps(false, address & ~(kChunk - 1),
           (address & ~(kChunk - 1)) + 2 * kChunk);
  const Mapping* m = FindMapping(address);
  if (!m) {
    u64 base = 0;
    auto* imported = HostImportAt(address, &base);
    if (!imported)
      return nullptr;
    *offset = address - base;
    *bytes = imported->end - address;
    return imported->buffer;
  }
  const u64 at = m->offset + (address - m->start);
  if (at >= g_direct.alias_size)
    return nullptr;
  rhi::Buffer* chunk = ChunkAt(at);
  if (!chunk)
    return nullptr;
  *offset = at & (kChunk - 1);
  *bytes = base::Min(kChunk - *offset, m->end - address);
  return chunk;
}

namespace {
u64 BindClass(u64 chunks) {
  u64 cls = 1;
  while (cls < chunks && cls < 32)
    cls *= 2;
  return cls;
}

BindImport BindSpan(u64 start, u64 cls) {
  const u64 key = start | cls;
  auto it = g_direct.binds.find(key);
  if (it != g_direct.binds.end())
    return it->second;
  const u64 size = base::Min(cls * kChunk, g_direct.alias_size - start);
  rhi::BufferDesc desc;
  desc.size = size;
  desc.usage = rhi::kBufferVertex | rhi::kBufferIndex | rhi::kBufferStorage |
               rhi::kBufferUniform;
  desc.host_pointer = g_direct.alias + start;
  desc.name = "guest direct bind";
  const BindImport imp{Device().CreateBuffer(desc), size};
  g_direct.binds.insert_or_assign(key, imp);
  return imp;
}
}  // namespace

rhi::Buffer* DirectBindAt(u64 address, u64 bytes, u64* offset) {
  if (!bytes || (!g_direct.alias && !InitGuestDirect()))
    return nullptr;
  ReadMaps(false, address, address + bytes);
  const Mapping* m = FindMapping(address);
  if (!m || address + bytes > m->end)
    return nullptr;
  const u64 at = m->offset + (address - m->start);
  if (at >= g_direct.alias_size)
    return nullptr;
  const u64 start = at & ~(kChunk - 1);
  const u64 inner = at - start;
  const u64 cls = BindClass((inner + bytes + kChunk - 1) / kChunk);
  if (start + cls * kChunk > g_direct.alias_size)
    return nullptr;
  const BindImport imp = BindSpan(start, cls);
  if (!imp.buffer || inner + bytes > imp.bytes)
    return nullptr;
  *offset = inner;
  return imp.buffer;
}

void SyncGuestDirect() {
  if (!gcn::g_direct_table && !InitGuestDirect())
    return;
  InvalidateCopiedImports();
  const u64 now = base::TickClock::NowNs();
  const bool periodic = now - g_direct.periodic_at >= 1000000000ull;
  if (periodic)
    g_direct.periodic_at = now;
  AttachDirectAlias(periodic);
  if (periodic || DirtyTouchesGpu())
    ReadMaps(true);
  if (g_direct.maps_changed) {
    PruneImports();
    g_direct.maps_changed = false;
    for (u64 block : g_direct.filled) {
      const u64 index = GuestDirectIndex(block);
      const Block target = BlockAt(index);
      if (!target.entry)
        continue;
      if (target.redirected->load(base::memory_order_acquire))
        *target.saved = EntryFor(block);
      else
        *target.entry = EntryFor(block);
    }
  }
  ReapImports();
  MirrorRefreshCached();
  auto* misses = reinterpret_cast<volatile u32*>(g_direct.misses->mapped());
  const u32 count = base::Min<u32>(misses[0], kDirectMissSlots);
  if (!count)
    return;
  const auto* slots =
      reinterpret_cast<const volatile u64*>(g_direct.misses->mapped() + 8);
  for (u32 i = 0; i < count; i++)
    Fill(slots[i]);
  misses[0] = 0;
}

namespace {
const guest::SessionReset g_session_reset([] {
  if (g_direct.alias)
    munmap(g_direct.alias, g_direct.alias_size);
  if (g_direct.saved)
    munmap(g_direct.saved, kFlatBlocks * 9);
  for (auto& entry : g_direct.high) {
    if (auto* high = entry.exchange(nullptr, base::memory_order_acq_rel)) {
      munmap(high->saved, kDirectoryBlocks * 9);
      delete high;
    }
  }
  guest::ResetResource(g_direct);
  gcn::g_direct_table = gcn::g_direct_misses = 0;
});
}  // namespace

}  // namespace gpu::render
