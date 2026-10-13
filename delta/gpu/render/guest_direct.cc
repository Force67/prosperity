#include "gpu/render/guest_direct.h"
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
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
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
base::Vector<Mapping> g_host_maps;
base::Map<u64, HostImport> g_host_imports;
base::HashMap<u64, Leaf> g_leaves;
base::Vector<rhi::Buffer*> g_leaf_slabs;
base::Vector<u32> g_free_leaves;
base::Vector<Retired> g_retired;
u32 g_next_leaf = 0;
u64 g_map_snapshot = 0;
struct Remap {
  u64 base, end;
};
base::Mutex g_remap_lock;
base::Vector<Remap> g_remaps;

u8* g_alias = nullptr;
u64 g_alias_size = 0;
rhi::Buffer* g_table = nullptr;
rhi::Buffer* g_misses = nullptr;
base::HashMap<u64, rhi::Buffer*> g_chunks;  // memfd offset -> import
// Whole-chunk imports for binds, keyed by chunk start and span size class.
struct BindImport {
  rhi::Buffer* buffer;
  u64 bytes;
};
base::HashMap<u64, BindImport> g_binds;
base::Vector<Mapping> g_maps;               // guest views of the memfd
base::HashSet<u64> g_filled;                // blocks with a table entry
// Redirected blocks: the host entry they go back to, and whether they are.
u64* g_saved = nullptr;
base::Atomic<u8>* g_redirected = nullptr;
u32 g_syncs = 0;
int g_init_state = -1;

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

u64 g_maps_read_at = ~0ull;
bool g_maps_changed = false;
u64 g_alias_probe_at = ~0ull;

void AttachDirectAlias(bool force = false) {
  if (g_alias)
    return;
  const u64 generation =
      gpu::GuestMappingGeneration().load(base::memory_order_acquire);
  if (!force && g_alias_probe_at == generation)
    return;
  g_alias_probe_at = generation;
  const int fd = FindDmemFd();
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0)
    return;
  const u64 size = static_cast<u64>(st.st_size);
  void* alias = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_NORESERVE, fd, 0);
  if (alias == MAP_FAILED)
    return;
  g_alias = static_cast<u8*>(alias);
  g_alias_size = size;
  g_maps_read_at = ~0ull;
}

void ReadMaps(bool force = false) {
  const u64 generation =
      gpu::GuestMappingGeneration().load(base::memory_order_acquire);
  if (!force && g_maps_read_at == generation)
    return;
  FILE* f = std::fopen("/proc/self/maps", "r");
  if (!f)
    return;
  g_maps_read_at = generation;
  g_maps_changed = true;
  ++g_map_snapshot;
  g_host_maps.clear();
  g_maps.clear();
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long start, end, offset;
    char perms[8];
    if (std::sscanf(line, "%llx-%llx %7s %llx", &start, &end, perms,
                    &offset) != 4)
      continue;
    if (!std::strstr(line, "memfd:delta_dmem")) {
      if (GuestDirectIndex(start) != ~0ull)
        g_host_maps.push_back(
            {start, end, offset, perms[0] == 'r', perms[1] == 'w'});
      continue;
    }
    if (!g_maps.empty() && g_maps.back().end == start &&
        g_maps.back().offset + (start - g_maps.back().start) == offset) {
      g_maps.back().end = end;
      continue;
    }
    g_maps.push_back({start, end, offset});
  }
  std::fclose(f);
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
  return FindMappingIn(g_maps, address);
}

rhi::Buffer* ChunkAt(u64 offset) {
  const u64 chunk = offset & ~(kChunk - 1);
  auto it = g_chunks.find(chunk);
  if (it != g_chunks.end())
    return it->second;
  rhi::BufferDesc desc;
  // A block's reads may run past its end: the import overlaps the next
  // chunk by a block, so they stay inside it.
  desc.size = base::Min(kChunk + (1ull << kDirectBlockShift),
                        g_alias_size - chunk);
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress |
               rhi::kBufferCopySrc | rhi::kBufferCopyDst;
  desc.host_pointer = g_alias + chunk;
  rhi::Buffer* buf = Device().CreateBuffer(desc);
  g_chunks.insert_or_assign(chunk, buf);
  return buf;
}

u64* Table() {
  return reinterpret_cast<u64*>(g_table->mapped());
}

void Retire(rhi::Buffer* buffer, u32 slot = ~0u) {
  g_retired.push_back({buffer, slot, g_frame.num, Device().LastSubmission()});
}

void ReapImports() {
  for (size_t i = 0; i < g_retired.size();) {
    const auto& retired = g_retired[i];
    if (retired.frame + 2 > g_frame.num ||
        !Device().IsComplete(retired.submission)) {
      ++i;
      continue;
    }
    if (retired.buffer)
      Device().Destroy(retired.buffer);
    else
      g_free_leaves.push_back(retired.slot);
    g_retired.erase(g_retired.begin() + i);
  }
}

void InvalidateCopiedImports() {
  base::Vector<Remap> remaps;
  {
    base::LockGuard lock(g_remap_lock);
    remaps.swap(g_remaps);
  }
  if (remaps.empty())
    return;
  ReadMaps();
  for (const auto& remap : remaps) {
    auto it = g_host_imports.upper_bound(remap.base);
    if (it != g_host_imports.begin())
      --it;
    while (it != g_host_imports.end() && it->first < remap.end) {
      if (it->second.readonly && remap.base < it->second.end) {
        Retire(it->second.buffer);
        it = g_host_imports.erase(it);
        g_maps_changed = true;
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

HostImport* HostImportAt(u64 address, u64* base, bool retry = true) {
  const Mapping* map = FindMappingIn(g_host_maps, address);
  if (!map)
    return nullptr;
  auto prior = g_host_imports.upper_bound(address);
  if (prior != g_host_imports.begin()) {
    --prior;
    auto& imported = prior->second;
    if (address < imported.end) {
      if (imported.snapshot == g_map_snapshot) {
        *base = prior->first;
        return &imported;
      }
      const u64 identity = host_memory::MemoryMappingIdentity(
          reinterpret_cast<void*>(prior->first), imported.end - prior->first);
      if (identity && identity == imported.identity &&
          (!imported.readonly ||
           !WritableMapping(*map, address & ~(kPage - 1),
                            base::Min(imported.end, map->end)))) {
        imported.snapshot = g_map_snapshot;
        *base = prior->first;
        return &imported;
      }
      Retire(imported.buffer);
      g_maps_changed = true;
      g_host_imports.erase(prior);
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
  if (auto previous = g_host_imports.find(start);
      previous != g_host_imports.end()) {
    Retire(previous->second.buffer);
    g_maps_changed = true;
  }
  auto& imported = g_host_imports[start];
  imported = {end, identity, g_map_snapshot, buffer, !writable};
  return &imported;
}

u64 HostEntryFor(u64 block) {
  PageEntry pages[kLeafPages]{};
  bool held = false, failed = false;
  for (u32 page = 0; page < kLeafPages; ++page) {
    const u64 address = block + page * kPage;
    if (const Mapping* direct = FindMapping(address)) {
      const u64 offset = direct->offset + address - direct->start;
      if (offset < g_alias_size) {
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
    } else if (const auto* map = FindMappingIn(g_host_maps, address);
               map && map->readable) {
      failed = true;
    }
  }
  auto previous = g_leaves.find(block);
  if (!failed && previous != g_leaves.end() &&
      std::memcmp(previous->second.pages, pages, sizeof(pages)) == 0) {
    const u32 slot = previous->second.slot;
    return (g_leaf_slabs[slot / kLeavesPerSlab]->address() +
            (slot % kLeavesPerSlab) * kLeafBytes) |
           gcn::kDirectLeaf;
  }
  if (previous != g_leaves.end()) {
    Retire(nullptr, previous->second.slot);
    g_leaves.erase(previous);
  }
  if (failed)
    return 0;
  if (!held)
    return gcn::kDirectUnmapped;
  u32 slot;
  if (!g_free_leaves.empty()) {
    slot = g_free_leaves.back();
    g_free_leaves.pop_back();
  } else {
    slot = g_next_leaf;
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
      g_leaf_slabs.push_back(slab);
    }
    ++g_next_leaf;
  }
  auto* slab = g_leaf_slabs[slot / kLeavesPerSlab];
  const u64 offset = (slot % kLeavesPerSlab) * kLeafBytes;
  std::memcpy(slab->mapped() + offset, pages, sizeof(pages));
  Leaf leaf{slot, {}};
  std::memcpy(leaf.pages, pages, sizeof(pages));
  g_leaves.insert_or_assign(block, leaf);
  return (slab->address() + offset) | gcn::kDirectLeaf;
}

// Ordinary direct-memory blocks need one lookup. Other mappings carry a
// page table so holes and short host spans never become physical GPU reads.
u64 EntryFor(u64 block) {
  const Mapping* m = FindMapping(block);
  if (!m || block + (1ull << kDirectBlockShift) > m->end)
    return HostEntryFor(block);
  const u64 offset = m->offset + (block - m->start);
  if (offset >= g_alias_size)
    return 0;
  rhi::Buffer* chunk = ChunkAt(offset);
  return chunk && chunk->address()
             ? chunk->address() + (offset & (kChunk - 1))
             : 0;
}

void Fill(u64 address) {
  ReadMaps();
  if (GuestDirectIndex(address) == ~0ull)
    return;
  // The whole 2 MiB around the miss: neighbours are read next.
  const u64 first = address & ~(kChunk - 1);
  for (u64 block = first; block < first + kChunk;
       block += 1ull << kDirectBlockShift) {
    const u64 index = GuestDirectIndex(block);
    if (index >= kBlocks)
      break;
    const u64 entry = EntryFor(block);
    if (g_redirected[index].load(base::memory_order_acquire)) {
      g_saved[index] = entry;
    } else {
      const Mapping* host = FindMappingIn(g_host_maps, block);
      const bool cacheable =
          FindMapping(block) ||
          (host &&
           WritableMapping(*host, block, block + (1ull << kDirectBlockShift)));
      const u64 cached = cacheable ? MirrorCacheBlock(block) : 0;
      if (cached)
        DirectRedirect(block, cached);
      else
        Table()[index] = entry;
    }
    g_filled.insert(block);
  }
}
}  // namespace

bool InitGuestDirect() {
  if (g_init_state >= 0) {
    if (g_init_state)
      AttachDirectAlias();
    return g_init_state;
  }
  g_init_state = 0;
  if (!kDirect || !Device().caps().host_import ||
      !Device().caps().buffer_address)
    return false;
  AttachDirectAlias();
  rhi::BufferDesc desc;
  desc.size = kBlocks * 8;
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
  desc.memory = rhi::MemoryKind::kUploadDevice;
  desc.name = "guest direct table";
  g_table = Device().CreateBuffer(desc);
  desc.size = 8 + kDirectMissSlots * 8;
  desc.memory = rhi::MemoryKind::kReadback;
  desc.name = "guest direct misses";
  g_misses = Device().CreateBuffer(desc);
  if (!g_table || !g_misses || !g_table->mapped() || !g_misses->mapped() ||
      !g_table->address() || !g_misses->address())
    return false;
  std::memset(g_table->mapped(), 0, kBlocks * 8);
  void* saved = mmap(nullptr, kBlocks * 9, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (saved == MAP_FAILED)
    return false;
  g_saved = static_cast<u64*>(saved);
  g_redirected =
      reinterpret_cast<base::Atomic<u8>*>(static_cast<u8*>(saved) + kBlocks * 8);
  std::memset(g_misses->mapped(), 0, 8 + kDirectMissSlots * 8);
  ReadMaps();
  gcn::g_direct_table = g_table->address();
  gcn::g_direct_misses = g_misses->address();
  BASE_LOGI("direct", "guest memory {:#x}+{:#x} through {} views of {} MiB",
            kDirectBase, kDirectSize, g_maps.size(), g_alias_size >> 20);
  g_init_state = 1;
  return true;
}

void NoteGuestDirectRemap(u64 base, u64 bytes) {
  if (!bytes || bytes > ~0ull - base)
    return;
  base::LockGuard lock(g_remap_lock);
  g_remaps.push_back({base, base + bytes});
}

bool InDirectAlias(u64 va) {
  return g_alias && va >= reinterpret_cast<u64>(g_alias) &&
         va < reinterpret_cast<u64>(g_alias) + g_alias_size;
}

void DirectRedirect(u64 block, u64 device_address) {
  const u64 index = GuestDirectIndex(block);
  if (!gcn::g_direct_table || index == ~0ull)
    return;
  if (!g_redirected[index].load(base::memory_order_acquire)) {
    ReadMaps();
    g_saved[index] = EntryFor(block);
    g_filled.insert(block);
  }
  __atomic_store_n(&Table()[index], device_address, __ATOMIC_RELEASE);
  g_redirected[index].store(1, base::memory_order_release);
}

void DirectRestore(u64 block) {
  const u64 index = GuestDirectIndex(block);
  if (!gcn::g_direct_table || index == ~0ull)
    return;
  if (!g_redirected[index].exchange(0, base::memory_order_acq_rel))
    return;
  __atomic_store_n(&Table()[index], g_saved[index], __ATOMIC_RELEASE);
}

rhi::Buffer* DirectImportAt(u64 address, u64* offset, u64* bytes) {
  if (!g_alias && !InitGuestDirect())
    return nullptr;
  ReadMaps();
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
  if (at >= g_alias_size)
    return nullptr;
  rhi::Buffer* chunk = ChunkAt(at);
  if (!chunk)
    return nullptr;
  *offset = at & (kChunk - 1);
  *bytes = base::Min(kChunk - *offset, m->end - address);
  return chunk;
}

namespace {
// Spans round up to a power-of-two chunk count, so neighbouring bindings of
// a title share imports: starts are chunk-aligned, the class fits the key's
// free low bits.
u64 BindClass(u64 chunks) {
  u64 cls = 1;
  while (cls < chunks && cls < 32)
    cls *= 2;
  return cls;
}

BindImport BindSpan(u64 start, u64 cls) {
  const u64 key = start | cls;
  auto it = g_binds.find(key);
  if (it != g_binds.end())
    return it->second;
  const u64 size = base::Min(cls * kChunk, g_alias_size - start);
  rhi::BufferDesc desc;
  desc.size = size;
  desc.usage = rhi::kBufferVertex | rhi::kBufferIndex | rhi::kBufferStorage |
               rhi::kBufferUniform;
  desc.host_pointer = g_alias + start;
  desc.name = "guest direct bind";
  const BindImport imp{Device().CreateBuffer(desc), size};
  g_binds.insert_or_assign(key, imp);
  return imp;
}
}  // namespace

rhi::Buffer* DirectBindAt(u64 address, u64 bytes, u64* offset) {
  if (!bytes || (!g_alias && !InitGuestDirect()))
    return nullptr;
  ReadMaps();
  const Mapping* m = FindMapping(address);
  // The fetch reads one contiguous span: it has to stay in this view.
  if (!m || address + bytes > m->end)
    return nullptr;
  const u64 at = m->offset + (address - m->start);
  if (at >= g_alias_size)
    return nullptr;
  const u64 start = at & ~(kChunk - 1);
  const u64 inner = at - start;
  const u64 cls = BindClass((inner + bytes + kChunk - 1) / kChunk);
  if (start + cls * kChunk > g_alias_size)
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
  // Refresh imported addresses after a remap, with periodic refreshes for
  // host mappings that bypass the guest notification.
  const bool periodic = ++g_syncs % 256 == 0;
  AttachDirectAlias(periodic);
  ReadMaps(periodic);
  if (g_maps_changed) {
    g_maps_changed = false;
    for (u64 block : g_filled) {
      const u64 index = GuestDirectIndex(block);
      if (g_redirected[index].load(base::memory_order_acquire))
        g_saved[index] = EntryFor(block);
      else
        Table()[index] = EntryFor(block);
    }
  }
  ReapImports();
  MirrorRecache();
  auto* misses = reinterpret_cast<volatile u32*>(g_misses->mapped());
  const u32 count = base::Min<u32>(misses[0], kDirectMissSlots);
  if (!count)
    return;
  const auto* slots =
      reinterpret_cast<const volatile u64*>(g_misses->mapped() + 8);
  for (u32 i = 0; i < count; i++)
    Fill(slots[i]);
  misses[0] = 0;
}

namespace {
const guest::SessionReset g_session_reset([] {
  if (g_alias)
    munmap(g_alias, g_alias_size);
  if (g_saved)
    munmap(g_saved, kBlocks * 9);
  g_init_state = -1;
  g_maps_read_at = ~0ull;
  g_maps_changed = false;
  g_alias_probe_at = ~0ull;
  gcn::g_direct_table = gcn::g_direct_misses = 0;
  guest::ResetResource(g_alias);
  guest::ResetResource(g_alias_size);
  guest::ResetResource(g_table);
  guest::ResetResource(g_misses);
  guest::ResetResource(g_chunks);
  guest::ResetResource(g_binds);
  guest::ResetResource(g_maps);
  guest::ResetResource(g_filled);
  guest::ResetResource(g_saved);
  guest::ResetResource(g_redirected);
  guest::ResetResource(g_syncs);
  guest::ResetResource(g_host_maps);
  guest::ResetResource(g_host_imports);
  guest::ResetResource(g_leaves);
  guest::ResetResource(g_leaf_slabs);
  guest::ResetResource(g_free_leaves);
  guest::ResetResource(g_retired);
  guest::ResetResource(g_next_leaf);
  guest::ResetResource(g_map_snapshot);
  guest::ResetResource(g_remaps);
});
}  // namespace

}  // namespace gpu::render
