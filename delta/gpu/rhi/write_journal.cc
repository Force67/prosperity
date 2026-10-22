#include "gpu/rhi/write_journal.h"

#include "base/atomic.h"
#include "gpu/rhi/device.h"

namespace gpu::rhi::write_journal {
namespace {

constexpr u32 kEntries = 1u << 16;
Entry g_ring[kEntries];
base::Atomic<u64> g_next{0};
u32 g_frame = 0;
u64 g_context = 0;

}  // namespace

void SetContext(u32 frame, u64 context) {
  g_frame = frame;
  g_context = context;
}

void Record(const Buffer* dst, u64 offset, u64 bytes, char kind) {
  if (!dst || !bytes)
    return;
  const auto host = reinterpret_cast<u64>(dst->desc().host_pointer);
  if (!host)
    return;
  Entry& e = g_ring[g_next.fetch_add(1, base::memory_order_relaxed) %
                    kEntries];
  e = {host + offset, host + offset + bytes, g_context, g_frame, kind};
}

u32 Find(u64 address, u64 radius, Entry* out, u32 max) {
  const u64 next = g_next.load(base::memory_order_relaxed);
  const u64 count = next < kEntries ? next : kEntries;
  u32 found = 0;
  for (u64 i = 0; i < count && found < max; i++) {
    const Entry& e = g_ring[(next - 1 - i) % kEntries];
    if (e.first <= address + radius && address < e.end + radius)
      out[found++] = e;
  }
  return found;
}

}  // namespace gpu::rhi::write_journal
