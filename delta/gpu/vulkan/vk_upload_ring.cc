/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_upload_ring.h"

#include "gpu/gpu_check.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace gpu::vk {

VkDeviceSize VbRingBytes() {
  static const VkDeviceSize bytes = [] {
    const char* e = std::getenv("DELTA_GPU_VBRING_MB");
    const long mb = e ? std::atol(e) : 0;
    if (mb <= 0)
      return kVbRing;
    return static_cast<VkDeviceSize>(mb) * 1024 * 1024;
  }();
  return bytes;
}

bool CreateUploadRings(const VkPhysicalDeviceProperties& props) {
  // Vertex ring.
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = VbRingBytes();
  bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VKOK(vkCreateBuffer(g_dev.device, &bi, nullptr, &g_ring.vb));
  VkMemoryRequirements vr;
  vkGetBufferMemoryRequirements(g_dev.device, g_ring.vb, &vr);
  VkMemoryAllocateInfo va{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  va.allocationSize = vr.size;
  va.memoryTypeIndex = FindMemoryType(vr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g_dev.device, &va, nullptr, &g_ring.vb_mem));
  VKOK(vkBindBufferMemory(g_dev.device, g_ring.vb, g_ring.vb_mem, 0));
  VKOK(vkMapMemory(g_dev.device, g_ring.vb_mem, 0, VbRingBytes(), 0,
                   (void**)&g_ring.vb_map));
  NameObject(VK_OBJECT_TYPE_BUFFER, (uint64_t)g_ring.vb, "vertex ring");

  // Index ring (host-visible, 32-bit indices).
  VkBufferCreateInfo ibi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ibi.size = kIbRing;
  ibi.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  VKOK(vkCreateBuffer(g_dev.device, &ibi, nullptr, &g_ring.ib));
  VkMemoryRequirements ir;
  vkGetBufferMemoryRequirements(g_dev.device, g_ring.ib, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex = FindMemoryType(ir.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g_dev.device, &ia, nullptr, &g_ring.ib_mem));
  VKOK(vkBindBufferMemory(g_dev.device, g_ring.ib, g_ring.ib_mem, 0));
  VKOK(vkMapMemory(g_dev.device, g_ring.ib_mem, 0, kIbRing, 0,
                   (void**)&g_ring.ib_map));
  NameObject(VK_OBJECT_TYPE_BUFFER, (uint64_t)g_ring.ib, "index ring");
  // Recomp cbuffer ring + dynamic-UBO descriptors (set 1) + empty set-0 layout.
  g_ring.ubo_align = (uint32_t)props.limits.minUniformBufferOffsetAlignment;
  if (g_ring.ubo_align < 1)
    g_ring.ubo_align = 1;
  if (props.limits.maxDescriptorSetUniformBuffersDynamic < kCbufBindings ||
      props.limits.maxPerStageDescriptorUniformBuffers < kCbufBindings)
    std::fprintf(stderr, "[gpuvk] only %u/%u dynamic UBOs available, need %u\n",
                 props.limits.maxDescriptorSetUniformBuffersDynamic,
                 props.limits.maxPerStageDescriptorUniformBuffers,
                 kCbufBindings);
  {
    VkBufferCreateInfo ub{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ub.size = kUboRing;
    ub.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VKOK(vkCreateBuffer(g_dev.device, &ub, nullptr, &g_ring.ubo_buf));
    VkMemoryRequirements ur;
    vkGetBufferMemoryRequirements(g_dev.device, g_ring.ubo_buf, &ur);
    VkMemoryAllocateInfo um{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    um.allocationSize = ur.size;
    um.memoryTypeIndex = FindMemoryType(
        ur.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKOK(vkAllocateMemory(g_dev.device, &um, nullptr, &g_ring.ubo_mem));
    VKOK(vkBindBufferMemory(g_dev.device, g_ring.ubo_buf, g_ring.ubo_mem, 0));
    VKOK(vkMapMemory(g_dev.device, g_ring.ubo_mem, 0, kUboRing, 0,
                     (void**)&g_ring.ubo_map));
    std::memset(g_ring.ubo_map, 0, kUboRing);
    NameObject(VK_OBJECT_TYPE_BUFFER, (uint64_t)g_ring.ubo_buf, "cbuffer ring");
    g_ring.ubo_stride = (kCbufWindow + g_ring.ubo_align - 1) &
                        ~(VkDeviceSize)(g_ring.ubo_align - 1);
    g_ring.ubo_written.resize(static_cast<size_t>(
        (kUboRing + g_ring.ubo_stride - 1) / g_ring.ubo_stride));

    // kMaxCbufBindings, not 8: a shader pair whose constant buffers exceed the
    // cap is planned only up to it, and every s_buffer_load from a dropped base
    // emits nothing, leaving its destination SGPRs zero. Skyrim's UI shaders
    // sit right at 8 cbufs, so their transform matrix read back as an all-zero
    // matrix and collapsed every vertex position.
    VkDescriptorSetLayoutBinding ubs[kCbufBindings]{};
    for (uint32_t i = 0; i < kCbufBindings; i++) {
      ubs[i].binding = i;
      ubs[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      ubs[i].descriptorCount = 1;
      ubs[i].stageFlags =
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ul{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ul.bindingCount = kCbufBindings;
    ul.pBindings = ubs;
    VKOK(vkCreateDescriptorSetLayout(g_dev.device, &ul, nullptr,
                                     &g_ring.ubo_layout));
    VkDescriptorSetLayoutCreateInfo el{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    el.bindingCount = 0;
    VKOK(vkCreateDescriptorSetLayout(g_dev.device, &el, nullptr,
                                     &g_ring.empty_layout));

    VkDescriptorPoolSize ups{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                             kCbufBindings};
    VkDescriptorPoolCreateInfo upi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    upi.maxSets = 1;
    upi.poolSizeCount = 1;
    upi.pPoolSizes = &ups;
    VKOK(vkCreateDescriptorPool(g_dev.device, &upi, nullptr, &g_ring.ubo_pool));
    VkDescriptorSetAllocateInfo uai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    uai.descriptorPool = g_ring.ubo_pool;
    uai.descriptorSetCount = 1;
    uai.pSetLayouts = &g_ring.ubo_layout;
    VKOK(vkAllocateDescriptorSets(g_dev.device, &uai, &g_ring.ubo_set));
    VkDescriptorBufferInfo ubinfo[kCbufBindings];
    VkWriteDescriptorSet uw[kCbufBindings];
    for (uint32_t i = 0; i < kCbufBindings; i++) {
      ubinfo[i] = {g_ring.ubo_buf, 0, kCbufWindow};
      uw[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      uw[i].dstSet = g_ring.ubo_set;
      uw[i].dstBinding = i;
      uw[i].descriptorCount = 1;
      uw[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      uw[i].pBufferInfo = &ubinfo[i];
    }
    vkUpdateDescriptorSets(g_dev.device, kCbufBindings, uw, 0, nullptr);
  }

  // Raw-buffer set layout (set 2). Every recompiled pipeline layout that has a
  // shader reading buffers by hand names it, so it exists from the start; the
  // ring behind it is allocated only if such a shader actually appears.
  {
    g_ring.sbo_align = (uint32_t)std::max<VkDeviceSize>(
        props.limits.minStorageBufferOffsetAlignment, 4);
    // These are DYNAMIC storage buffers, and maxDescriptorSetStorageBuffersDynamic
    // has a Vulkan floor of 4 while real desktop parts report 16+. Take what the
    // device offers (up to our compile-time ceiling) and tell the recompiler, so
    // a shader referencing more raw buffers than 4 is planned rather than
    // declined wherever the hardware can carry it.
    g_ring.sbo_count =
        std::min<uint32_t>(props.limits.maxDescriptorSetStorageBuffersDynamic,
                           kRawBufBindings);
    if (g_ring.sbo_count < gpu::gcn::kMinGfxBuffers) {
      std::fprintf(stderr,
                   "[gpuvk] only %u dynamic storage buffers available, below "
                   "the %u Vulkan floor: shaders reading raw buffers will "
                   "decline\n",
                   g_ring.sbo_count, gpu::gcn::kMinGfxBuffers);
      g_ring.sbo_count = gpu::gcn::kMinGfxBuffers;
    }
    gpu::gcn::SetMaxGfxBuffers(g_ring.sbo_count);
    g_ring.sbo_stride = (kRawBufWindow + g_ring.sbo_align - 1) &
                        ~(VkDeviceSize)(g_ring.sbo_align - 1);
    VkDescriptorSetLayoutBinding sbs[kRawBufBindings]{};
    for (uint32_t i = 0; i < g_ring.sbo_count; i++) {
      sbs[i].binding = i;
      sbs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      sbs[i].descriptorCount = 1;
      sbs[i].stageFlags =
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sl{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = g_ring.sbo_count;
    sl.pBindings = sbs;
    VKOK(vkCreateDescriptorSetLayout(g_dev.device, &sl, nullptr,
                                     &g_ring.sbo_layout));
  }
  return true;
}

bool EnsureRawBufferRing() {
  if (g_ring.sbo_map)
    return true;
  if (!g_ring.sbo_layout)
    return false;
  VkBufferCreateInfo sb{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  sb.size = kSboRing;
  sb.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VKOK(vkCreateBuffer(g_dev.device, &sb, nullptr, &g_ring.sbo_buf));
  VkMemoryRequirements sr;
  vkGetBufferMemoryRequirements(g_dev.device, g_ring.sbo_buf, &sr);
  VkMemoryAllocateInfo sm{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  sm.allocationSize = sr.size;
  sm.memoryTypeIndex = FindMemoryType(sr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g_dev.device, &sm, nullptr, &g_ring.sbo_mem));
  VKOK(vkBindBufferMemory(g_dev.device, g_ring.sbo_buf, g_ring.sbo_mem, 0));
  VKOK(vkMapMemory(g_dev.device, g_ring.sbo_mem, 0, kSboRing, 0,
                   (void**)&g_ring.sbo_map));
  std::memset(g_ring.sbo_map, 0, kSboRing);
  NameObject(VK_OBJECT_TYPE_BUFFER, (uint64_t)g_ring.sbo_buf,
             "raw buffer ring");
  g_ring.sbo_written.assign(
      static_cast<size_t>((kSboRing + g_ring.sbo_stride - 1) /
                          g_ring.sbo_stride),
      0);

  VkDescriptorPoolSize sps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
                           g_ring.sbo_count};
  VkDescriptorPoolCreateInfo spi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  spi.maxSets = 1;
  spi.poolSizeCount = 1;
  spi.pPoolSizes = &sps;
  VKOK(vkCreateDescriptorPool(g_dev.device, &spi, nullptr, &g_ring.sbo_pool));
  VkDescriptorSetAllocateInfo sai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  sai.descriptorPool = g_ring.sbo_pool;
  sai.descriptorSetCount = 1;
  sai.pSetLayouts = &g_ring.sbo_layout;
  VKOK(vkAllocateDescriptorSets(g_dev.device, &sai, &g_ring.sbo_set));
  VkDescriptorBufferInfo sbinfo[kRawBufBindings];
  VkWriteDescriptorSet sw[kRawBufBindings];
  for (uint32_t i = 0; i < g_ring.sbo_count; i++) {
    sbinfo[i] = {g_ring.sbo_buf, 0, kRawBufWindow};
    sw[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    sw[i].dstSet = g_ring.sbo_set;
    sw[i].dstBinding = i;
    sw[i].descriptorCount = 1;
    sw[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    sw[i].pBufferInfo = &sbinfo[i];
  }
  vkUpdateDescriptorSets(g_dev.device, g_ring.sbo_count, sw, 0, nullptr);
  std::fprintf(stderr,
               "[gpuvk] raw-buffer ring: %llu MB, %u KB windows, %u bindings\n",
               (unsigned long long)(kSboRing >> 20), kRawBufWindow >> 10,
               g_ring.sbo_count);
  return true;
}

bool AllocateTextureUpload(uint32_t slot,
                           VkDeviceSize bytes,
                           VkDeviceSize alignment,
                           TextureUploadSlice& slice) {
  GPU_BUGCHECK(slot < 2, "slot %u is not a frame-ring slot", slot);
  if (!bytes)
    return false;
  alignment = std::max<VkDeviceSize>(alignment, 4);
  GPU_BUGCHECK((alignment & (alignment - 1)) == 0,
               "alignment %llu is not a power of two",
               (unsigned long long)alignment);
  auto& blocks = g_ring.texture_uploads[slot];
  for (auto& block : blocks) {
    const VkDeviceSize aligned =
        (block.offset + alignment - 1) & ~(alignment - 1);
    if (aligned <= block.capacity && bytes <= block.capacity - aligned) {
      slice = {block.buffer, aligned, block.map + aligned};
      block.offset = aligned + bytes;
      return true;
    }
  }

  constexpr VkDeviceSize kInitialCapacity = 16ull * 1024 * 1024;
  VkDeviceSize capacity = kInitialCapacity;
  while (capacity < bytes) {
    if (capacity > std::numeric_limits<VkDeviceSize>::max() / 2)
      return false;
    capacity *= 2;
  }
  TextureUploadBlock block;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = capacity;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &block.buffer) != VK_SUCCESS)
    return false;
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(g_dev.device, block.buffer, &requirements);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &block.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, block.buffer, nullptr);
    return false;
  }
  if (vkBindBufferMemory(g_dev.device, block.buffer, block.memory, 0) !=
          VK_SUCCESS ||
      vkMapMemory(g_dev.device, block.memory, 0, capacity, 0,
                  reinterpret_cast<void**>(&block.map)) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, block.buffer, nullptr);
    vkFreeMemory(g_dev.device, block.memory, nullptr);
    return false;
  }
  block.capacity = capacity;
  block.offset = bytes;
  slice = {block.buffer, 0, block.map};
  NameObject(VK_OBJECT_TYPE_BUFFER, (uint64_t)block.buffer,
             "texup slot %u block %zu (%llu MB)", slot, blocks.size(),
             (unsigned long long)(capacity >> 20));
  blocks.push_back(block);
  return true;
}

void ResetTextureUploads(uint32_t slot) {
  GPU_BUGCHECK(slot < 2, "slot %u is not a frame-ring slot", slot);
  // Reset runs in BeginFrame after this slot's fence wait, so no in-flight
  // transfer references these blocks -- the one point where destroying one is
  // safe. Blocks grow on demand (a loading burst can leave hundreds of idle
  // megabytes behind), so drop any block that has sat unused for ~10s of
  // resets; the next burst simply recreates it.
  constexpr uint64_t kIdleResetsBeforeTrim = 300;
  auto& blocks = g_ring.texture_uploads[slot];
  for (size_t i = 0; i < blocks.size();) {
    TextureUploadBlock& block = blocks[i];
    block.idle_resets = block.offset ? 0 : block.idle_resets + 1;
    block.offset = 0;
    if (block.idle_resets < kIdleResetsBeforeTrim) {
      ++i;
      continue;
    }
    vkUnmapMemory(g_dev.device, block.memory);
    vkDestroyBuffer(g_dev.device, block.buffer, nullptr);
    vkFreeMemory(g_dev.device, block.memory, nullptr);
    blocks.erase(blocks.begin() + i);
  }
}

}  // namespace gpu::vk
