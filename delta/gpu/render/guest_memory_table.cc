/* PS4Delta: guest virtual addresses -> checked device buffer addresses. */
#include "gpu/render/guest_memory_table.h"
#include <cstring>
#include "guest/session.h"

#include "base/containers/array.h"
#include "base/containers/span.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "gpu/render/frame.h"

namespace gpu::render {
namespace {
struct Buffer {
  rhi::Buffer* buffer = nullptr;
  void* mapped = nullptr;
  u64 address = 0;
  u64 size = 0;
};
struct TableSet {
  Buffer table, flags, misses;
  u32 windows = 0;
  bool busy = false;
};
base::Vector<TableSet> g_sets;
constexpr u32 kMissSlots = 63;

bool Ensure(Buffer& out, u64 bytes) {
  if (out.size >= bytes)
    return true;
  Device().Destroy(out.buffer);
  out = {};
  rhi::BufferDesc desc;
  desc.size = (bytes + 255) & ~u64(255);
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
  desc.memory = rhi::MemoryKind::kReadback;
  out.buffer = Device().CreateBuffer(desc);
  if (!out.buffer)
    return false;
  out.mapped = out.buffer->mapped();
  out.address = out.buffer->address();
  out.size = desc.size;
  if (!out.mapped || !out.address) {
    Device().Destroy(out.buffer);
    out = {};
    return false;
  }
  return true;
}
}  // namespace

bool PrepareGuestMemory(base::Span<const GuestMemoryWindow> windows,
                        rhi::Buffer*& table,
                        u64& table_bytes,
                        bool writes,
                        u32& set) {
  if (!Device().caps().buffer_address)
    return false;
  set = 0;
  while (set < g_sets.size() && g_sets[set].busy)
    set++;
  if (set == g_sets.size())
    g_sets.emplace_back();
  TableSet& s = g_sets[set];
  if (!s.misses.buffer) {
    if (!Ensure(s.misses, 8 + kMissSlots * 8))
      return false;
    std::memset(s.misses.mapped, 0, s.misses.size);
  }
  const u64 bytes = (windows.size() + 1) * sizeof(base::Array<u64, 4>);
  if (!Ensure(s.table, bytes) ||
      (writes && !Ensure(s.flags, windows.size() * 4 + 4)))
    return false;
  auto* entries = static_cast<base::Array<u64, 4>*>(s.table.mapped);
  entries[0] = {0, 0, s.misses.address, 0};
  for (size_t i = 0; i < windows.size(); i++) {
    const GuestMemoryWindow& w = windows[i];
    entries[i + 1] = {w.base, w.base + w.size, w.address,
                      writes && w.writable ? s.flags.address + i * 4 : 0};
  }
  if (writes)
    std::memset(s.flags.mapped, 0, windows.size() * 4);
  s.windows = writes ? static_cast<u32>(windows.size()) : 0;
  s.busy = true;
  table = s.table.buffer;
  table_bytes = bytes;
  return true;
}

base::Vector<u32> TakeWrittenWindows(u32 set) {
  base::Vector<u32> written;
  TableSet& s = g_sets[set];
  auto* flags = static_cast<u32*>(s.flags.mapped);
  for (u32 i = 0; i < s.windows; i++)
    if (flags[i]) {
      written.push_back(i);
      flags[i] = 0;
    }
  s.windows = 0;
  return written;
}

base::Vector<u64> TakeGuestMisses(u32 set) {
  base::Vector<u64> misses;
  auto* words = static_cast<u32*>(g_sets[set].misses.mapped);
  if (!words)
    return misses;
  const u32 count = words[0] < kMissSlots ? words[0] : kMissSlots;
  for (u32 i = 0; i < count; i++)
    misses.push_back(words[2 + i * 2] | (u64(words[3 + i * 2]) << 32));
  words[0] = 0;
  return misses;
}

void ReleaseGuestMemory(u32 set) {
  g_sets[set].busy = false;
}

namespace {
const guest::SessionReset g_session_reset([] { guest::ResetResource(g_sets); });
}  // namespace

}  // namespace gpu::render
