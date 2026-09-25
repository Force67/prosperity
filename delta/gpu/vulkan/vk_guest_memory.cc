/* PS4Delta: guest virtual addresses -> checked device buffer addresses. */
#include "gpu/vulkan/vk_guest_memory.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <unordered_map>
#include <base/logging.h>
#include "gpu/vulkan/vk_frame.h"

namespace gpu::vk {
namespace {
struct Buffer {
  rhi::Buffer* buffer = nullptr;
  void* mapped = nullptr;
  u64 address = 0;
  u64 size = 0, identity = 0;
  bool imported = false;
};
std::unordered_map<u64, Buffer> spans, dirty_spans;
Buffer table_buffer;

void Destroy(Buffer& b) {
  Device().Destroy(b.buffer);
  b = {};
}

bool Create(Buffer& out, u64 bytes, void* host = nullptr) {
  rhi::BufferDesc desc;
  desc.size = bytes;
  desc.usage = rhi::kBufferStorage | rhi::kBufferAddress;
  desc.memory = rhi::MemoryKind::kReadback;
  // These are allocations in this process, not foreign host mappings.
  desc.host_pointer = host;
  out.buffer = Device().CreateBuffer(desc);
  if (!out.buffer) {
    BASE_LOGI("gpuvk", "guest mapping allocation size={} imported={} failed",
              bytes, host != nullptr);
    return false;
  }
  out.imported = host != nullptr;
  out.mapped = out.buffer->mapped();
  out.address = out.buffer->address();
  out.size = bytes;
  if (!out.mapped || !out.address) {
    Destroy(out);
    return false;
  }
  return true;
}
}  // namespace

bool PrepareGuestMemory(std::span<const render::GuestMemoryRange> ranges,
                        rhi::Buffer*& table,
                        u64& table_bytes,
                        bool writes) {
  const rhi::Caps& caps = Device().caps();
  if (!caps.buffer_address || ranges.empty())
    return false;
  for (auto it = spans.begin(); it != spans.end();) {
    const bool current =
        std::any_of(ranges.begin(), ranges.end(), [&](const auto& r) {
          return r.base == it->first && r.size == it->second.size &&
                 r.identity && r.identity == it->second.identity;
        });
    if (!current) {
      Destroy(it->second);
      if (auto d = dirty_spans.find(it->first); d != dirty_spans.end()) {
        Destroy(d->second);
        dirty_spans.erase(d);
      }
      it = spans.erase(it);
    } else {
      ++it;
    }
  }
  std::vector<std::array<u64, 4>> entries;
  for (const auto& range : ranges) {
    Buffer& b = spans[range.base];
    if (!b.buffer) {
      const u64 align = caps.host_import_alignment;
      const bool can_import = range.writable && caps.host_import &&
                              align && !(range.base % align) &&
                              !(range.size % align);
      if ((!can_import ||
           !Create(b, range.size, reinterpret_cast<void*>(range.base))) &&
          !Create(b, range.size)) {
        BASE_LOGI("gpuvk", "cannot map guest GPU range {:#x}+{:#x}", range.base,
                  range.size);
        return false;
      }
      b.identity = range.identity;
    }
    if (!b.imported)
      std::memcpy(b.mapped, reinterpret_cast<const void*>(range.base),
                  range.size);
    u64 dirty_address = 0;
    if (writes && range.writable) {
      // Min/max dword indices followed by a bit for every potentially written
      // dword. Imported spans need only invalidation; copied spans need exact
      // writeback so unrelated guest data is preserved.
      if (range.size > (u64(UINT32_MAX) * 4))
        return false;
      Buffer& dirty = dirty_spans[range.base];
      const u64 bitmap_bytes = 8 + ((range.size + 127) / 128) * 4;
      if (!dirty.buffer) {
        if (!Create(dirty, (bitmap_bytes + 255) & ~u64(255)))
          return false;
        std::memset(dirty.mapped, 0, dirty.size);
        static_cast<u32*>(dirty.mapped)[0] = UINT32_MAX;
      }
      dirty_address = dirty.address;
    }
    entries.push_back(
        {range.base, range.base + range.size, b.address, dirty_address});
  }
  const u64 bytes = entries.size() * sizeof(entries[0]);
  if (table_buffer.size < bytes) {
    Destroy(table_buffer);
    if (!Create(table_buffer, (bytes + 255) & ~u64(255)))
      return false;
  }
  std::memcpy(table_buffer.mapped, entries.data(), bytes);
  table = table_buffer.buffer;
  table_bytes = bytes;
  return true;
}

std::vector<render::GuestMemoryRange> FinishGuestMemoryWrites() {
  std::vector<render::GuestMemoryRange> written;
  for (auto& [base, dirty] : dirty_spans) {
    auto* marks = static_cast<u32*>(dirty.mapped);
    const u32 first = marks[0], end = marks[1];
    if (first >= end)
      continue;
    auto& buffer = spans.at(base);
    // Iterate marked words only, and coalesce adjacent words for cache updates.
    for (u64 word = first / 32; word <= (end - 1) / 32; ++word) {
      u32 bits = marks[2 + word];
      while (bits) {
        const u32 bit = std::countr_zero(bits);
        const u64 offset = (word * 32 + bit) * 4;
        if (offset + 4 <= buffer.size) {
          if (!buffer.imported)
            std::memcpy(reinterpret_cast<void*>(base + offset),
                        static_cast<const u8*>(buffer.mapped) + offset, 4);
          if (!written.empty() &&
              written.back().base + written.back().size == base + offset)
            written.back().size += 4;
          else
            written.push_back({base + offset, 4});
        }
        bits &= bits - 1;
      }
      marks[2 + word] = 0;
    }
    marks[0] = UINT32_MAX;
    marks[1] = 0;
  }
  return written;
}
}  // namespace gpu::vk
