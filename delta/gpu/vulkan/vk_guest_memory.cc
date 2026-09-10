/* PS4Delta: guest virtual addresses -> checked Vulkan buffer addresses. */
#include "gpu/vulkan/vk_guest_memory.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <unordered_map>
#include "gpu/vulkan/vk_device.h"

namespace gpu::vk {
namespace {
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceAddress address = 0;
  u64 size = 0, identity = 0;
  bool imported = false;
};
std::unordered_map<u64, Buffer> spans, dirty_spans;
Buffer table_buffer;

void Destroy(Buffer& b) {
  if (b.mapped && !b.imported)
    vkUnmapMemory(g_dev.device, b.memory);
  if (b.buffer)
    vkDestroyBuffer(g_dev.device, b.buffer, nullptr);
  if (b.memory)
    vkFreeMemory(g_dev.device, b.memory, nullptr);
  b = {};
}

bool Create(Buffer& out, u64 bytes, void* host = nullptr) {
  VkExternalMemoryBufferCreateInfo external{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = bytes;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if (host)
    bi.pNext = &external;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &out.buffer) != VK_SUCCESS)
    return false;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, out.buffer, &mr);
  u32 bits = mr.memoryTypeBits;
  if (host) {
    auto fn = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetDeviceProcAddr(g_dev.device,
                            "vkGetMemoryHostPointerPropertiesEXT"));
    VkMemoryHostPointerPropertiesEXT properties{
        VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!fn || mr.size > bytes ||
        fn(g_dev.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
           host, &properties) != VK_SUCCESS) {
      Destroy(out);
      return false;
    }
    bits &= properties.memoryTypeBits;
  }
  VkPhysicalDeviceMemoryProperties memory;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &memory);
  u32 chosen = UINT32_MAX;
  for (u32 i = 0; i < memory.memoryTypeCount; i++) {
    const auto flags = memory.memoryTypes[i].propertyFlags;
    if ((bits & (1u << i)) &&
        ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
      chosen = i;
      if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        break;
    }
  }
  if (chosen == UINT32_MAX) {
    Destroy(out);
    return false;
  }
  VkImportMemoryHostPointerInfoEXT imported{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
  // These are allocations in this process, not foreign host mappings.
  imported.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  imported.pHostPointer = host;
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  flags.pNext = host ? &imported : nullptr;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &flags;
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = chosen;
  const VkResult allocated =
      vkAllocateMemory(g_dev.device, &ai, nullptr, &out.memory);
  const VkResult bound =
      allocated == VK_SUCCESS
          ? vkBindBufferMemory(g_dev.device, out.buffer, out.memory, 0)
          : allocated;
  if (bound != VK_SUCCESS) {
    BASE_LOGI(
        "gpuvk",
        "guest mapping allocation size={} imported={} allocate={} bind={}",
        bytes, host != nullptr, int(allocated), int(bound));
    Destroy(out);
    return false;
  }
  out.imported = host != nullptr;
  out.mapped = host;
  if (!host && vkMapMemory(g_dev.device, out.memory, 0, VK_WHOLE_SIZE, 0,
                           &out.mapped) != VK_SUCCESS) {
    Destroy(out);
    return false;
  }
  VkBufferDeviceAddressInfo address{
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  address.buffer = out.buffer;
  out.address = vkGetBufferDeviceAddress(g_dev.device, &address);
  out.size = bytes;
  if (!out.address) {
    Destroy(out);
    return false;
  }
  return true;
}
}  // namespace

bool PrepareGuestMemory(std::span<const rhi::GuestMemoryRange> ranges,
                        VkDescriptorBufferInfo& table,
                        bool writes) {
  if (!g_dev.buffer_device_address || ranges.empty())
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
      const u64 align = g_dev.host_import_align;
      const bool can_import = range.writable && g_dev.host_import_available &&
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
    VkDeviceAddress dirty_address = 0;
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
  table = {table_buffer.buffer, 0, bytes};
  return true;
}

std::vector<rhi::GuestMemoryRange> FinishGuestMemoryWrites() {
  std::vector<rhi::GuestMemoryRange> written;
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
