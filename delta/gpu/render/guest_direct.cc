#include "gpu/render/guest_direct.h"

#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "options/options.h"

namespace gpu::render {
namespace {
DELTA_OPTION(bool, kDirect, "DELTA_GPU_DIRECT", false);

constexpr u64 kBlocks = kDirectSize >> kDirectBlockShift;
constexpr u64 kChunk = 2ull << 20;

struct Mapping {
  u64 start, end, offset;
};

u8* g_alias = nullptr;
u64 g_alias_size = 0;
rhi::Buffer* g_table = nullptr;
rhi::Buffer* g_misses = nullptr;
base::HashMap<u64, rhi::Buffer*> g_chunks;  // memfd offset -> import
base::Vector<Mapping> g_maps;               // guest views of the memfd
base::HashSet<u64> g_filled;                // blocks with a table entry
u32 g_syncs = 0;

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

void ReadMaps() {
  g_maps.clear();
  FILE* f = std::fopen("/proc/self/maps", "r");
  if (!f)
    return;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    if (!std::strstr(line, "memfd:delta_dmem"))
      continue;
    unsigned long long start, end, offset;
    char perms[8];
    if (std::sscanf(line, "%llx-%llx %7s %llx", &start, &end, perms,
                    &offset) != 4)
      continue;
    if (!g_maps.empty() && g_maps.back().end == start &&
        g_maps.back().offset + (start - g_maps.back().start) == offset) {
      g_maps.back().end = end;
      continue;
    }
    g_maps.push_back({start, end, offset});
  }
  std::fclose(f);
}

const Mapping* FindMapping(u64 address) {
  u32 lo = 0, hi = g_maps.size();
  while (lo < hi) {
    const u32 mid = (lo + hi) / 2;
    if (g_maps[mid].end <= address)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo < g_maps.size() && g_maps[lo].start <= address ? &g_maps[lo]
                                                           : nullptr;
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

// The entry for the 64 KiB block at `block` (a guest address): the device
// address of its first byte, or 0 where no memfd view holds it.
u64 EntryFor(u64 block) {
  const Mapping* m = FindMapping(block);
  if (!m)
    return 0;
  const u64 offset = m->offset + (block - m->start);
  if (offset >= g_alias_size)
    return 0;
  rhi::Buffer* chunk = ChunkAt(offset);
  return chunk && chunk->address()
             ? chunk->address() + (offset & (kChunk - 1))
             : 0;
}

void Fill(u64 address) {
  if (address < kDirectBase || address >= kDirectBase + kDirectSize)
    return;
  if (!FindMapping(address))
    ReadMaps();
  // The whole 2 MiB around the miss: neighbours are read next.
  const u64 first = address & ~(kChunk - 1);
  for (u64 block = first; block < first + kChunk;
       block += 1ull << kDirectBlockShift) {
    const u64 index = (block - kDirectBase) >> kDirectBlockShift;
    if (index >= kBlocks)
      break;
    Table()[index] = EntryFor(block);
    g_filled.insert(block);
  }
}
}  // namespace

bool InitGuestDirect() {
  static int state = -1;
  if (state >= 0)
    return state;
  state = 0;
  if (!kDirect || !Device().caps().host_import ||
      !Device().caps().buffer_address)
    return false;
  const int fd = FindDmemFd();
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
    state = -1;  // no direct memory yet
    return false;
  }
  g_alias_size = static_cast<u64>(st.st_size);
  void* alias = mmap(nullptr, g_alias_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_NORESERVE, fd, 0);
  if (alias == MAP_FAILED)
    return false;
  g_alias = static_cast<u8*>(alias);
  rhi::BufferDesc desc;
  desc.size = kBlocks * 8;
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
  desc.memory = rhi::MemoryKind::kUpload;
  desc.name = "guest direct table";
  g_table = Device().CreateBuffer(desc);
  desc.size = 8 + kDirectMissSlots * 8;
  desc.memory = rhi::MemoryKind::kReadback;
  desc.name = "guest direct misses";
  g_misses = Device().CreateBuffer(desc);
  if (!g_table || !g_misses || !g_table->mapped() || !g_misses->mapped())
    return false;
  std::memset(g_table->mapped(), 0, kBlocks * 8);
  std::memset(g_misses->mapped(), 0, 8 + kDirectMissSlots * 8);
  ReadMaps();
  gcn::g_direct_table = g_table->address();
  gcn::g_direct_misses = g_misses->address();
  BASE_LOGI("direct", "guest memory {:#x}+{:#x} through {} views of {} MiB",
            kDirectBase, kDirectSize, g_maps.size(), g_alias_size >> 20);
  state = 1;
  return true;
}

bool InDirectAlias(u64 va) {
  return g_alias && va >= reinterpret_cast<u64>(g_alias) &&
         va < reinterpret_cast<u64>(g_alias) + g_alias_size;
}

void SyncGuestDirect() {
  if (!gcn::g_direct_table && !InitGuestDirect())
    return;
  // A view can move to other bytes: re-read the views now and then and
  // refresh what the table already holds.
  if (++g_syncs % 256 == 0) {
    ReadMaps();
    for (u64 block : g_filled)
      Table()[(block - kDirectBase) >> kDirectBlockShift] = EntryFor(block);
  }
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
}  // namespace gpu::render
