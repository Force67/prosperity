/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_upload_ring.h"

#include "gpu/vulkan/vk_device.h"

#include <cstdio>

namespace gpu::vk {


bool CreateUploadRings(const VkPhysicalDeviceProperties& props) {
  // Vertex ring.
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = kVbRing;
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
  VKOK(vkMapMemory(g_dev.device, g_ring.vb_mem, 0, kVbRing, 0,
                   (void**)&g_ring.vb_map));

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
  return true;
}

}  // namespace gpu::vk
