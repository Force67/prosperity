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
Buffer g_table, g_flags, g_misses;
u32 g_windows = 0;
// Entry 0 of the table is a header the search can never match (an empty
// window at 0): its third field names the miss buffer, a u32 count then up
// to kMissSlots addresses the shader found in no window.
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
                        bool writes) {
  if (!Device().caps().buffer_address)
    return false;
  if (!g_misses.buffer) {
    if (!Ensure(g_misses, 8 + kMissSlots * 8))
      return false;
    std::memset(g_misses.mapped, 0, g_misses.size);
  }
  // One u32 written flag per window: entry i + 1's dirty field names flag i.
  const u64 bytes = (windows.size() + 1) * sizeof(base::Array<u64, 4>);
  if (!Ensure(g_table, bytes) ||
      (writes && !Ensure(g_flags, windows.size() * 4 + 4)))
    return false;
  auto* entries = static_cast<base::Array<u64, 4>*>(g_table.mapped);
  entries[0] = {0, 0, g_misses.address, 0};
  for (size_t i = 0; i < windows.size(); i++) {
    const GuestMemoryWindow& w = windows[i];
    // No flag: the shader drops stores into a read-only window.
    entries[i + 1] = {w.base, w.base + w.size, w.address,
                      writes && w.writable ? g_flags.address + i * 4 : 0};
  }
  if (writes)
    std::memset(g_flags.mapped, 0, windows.size() * 4);
  g_windows = writes ? static_cast<u32>(windows.size()) : 0;
  table = g_table.buffer;
  table_bytes = bytes;
  return true;
}

base::Vector<u32> TakeWrittenWindows() {
  base::Vector<u32> written;
  auto* flags = static_cast<u32*>(g_flags.mapped);
  for (u32 i = 0; i < g_windows; i++)
    if (flags[i]) {
      written.push_back(i);
      flags[i] = 0;
    }
  g_windows = 0;
  return written;
}
base::Vector<u64> TakeGuestMisses() {
  base::Vector<u64> misses;
  if (!g_misses.mapped)
    return misses;
  auto* words = static_cast<u32*>(g_misses.mapped);
  const u32 count = words[0] < kMissSlots ? words[0] : kMissSlots;
  for (u32 i = 0; i < count; i++)
    misses.push_back(words[2 + i * 2] | (u64(words[3 + i * 2]) << 32));
  words[0] = 0;
  return misses;
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_table);
  guest::ResetResource(g_flags);
  guest::ResetResource(g_misses);
  guest::ResetResource(g_windows);
});
}  // namespace

}  // namespace gpu::render
