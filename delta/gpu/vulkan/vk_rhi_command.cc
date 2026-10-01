/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstring>

#include "gpu/vulkan/vk_rhi_internal.h"

namespace gpu::vk::rhi_impl {

namespace {

VulkanBuffer* Buf(rhi::Buffer* b) {
  return static_cast<VulkanBuffer*>(b);
}
VulkanTexture* Tex(rhi::Texture* t) {
  return static_cast<VulkanTexture*>(t);
}

VkAccessFlags AccessFor(u32 a) {
  VkAccessFlags out = 0;
  if (a & rhi::kAccessVertexRead)
    out |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
  if (a & rhi::kAccessIndexRead)
    out |= VK_ACCESS_INDEX_READ_BIT;
  if (a & rhi::kAccessUniformRead)
    out |= VK_ACCESS_UNIFORM_READ_BIT;
  if (a & rhi::kAccessIndirectRead)
    out |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  if (a & (rhi::kAccessShaderRead | rhi::kAccessComputeRead))
    out |= VK_ACCESS_SHADER_READ_BIT;
  if (a & (rhi::kAccessShaderWrite | rhi::kAccessComputeWrite))
    out |= VK_ACCESS_SHADER_WRITE_BIT;
  if (a & rhi::kAccessCopyRead)
    out |= VK_ACCESS_TRANSFER_READ_BIT;
  if (a & rhi::kAccessCopyWrite)
    out |= VK_ACCESS_TRANSFER_WRITE_BIT;
  if (a & rhi::kAccessHostRead)
    out |= VK_ACCESS_HOST_READ_BIT;
  if (a & rhi::kAccessHostWrite)
    out |= VK_ACCESS_HOST_WRITE_BIT;
  if (a & rhi::kAccessColorWrite)
    out |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  if (a & rhi::kAccessDepthWrite)
    out |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  return out;
}

VkPipelineStageFlags StagesFor(u32 a, VkPipelineStageFlags shader) {
  VkPipelineStageFlags out = 0;
  if (a & (rhi::kAccessVertexRead | rhi::kAccessIndexRead))
    out |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
  if (a & rhi::kAccessIndirectRead)
    out |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  if (a & (rhi::kAccessUniformRead | rhi::kAccessShaderRead |
           rhi::kAccessShaderWrite))
    out |= shader;
  if (a & (rhi::kAccessComputeRead | rhi::kAccessComputeWrite))
    out |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (a & (rhi::kAccessCopyRead | rhi::kAccessCopyWrite))
    out |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (a & (rhi::kAccessHostRead | rhi::kAccessHostWrite))
    out |= VK_PIPELINE_STAGE_HOST_BIT;
  if (a & rhi::kAccessColorWrite)
    out |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (a & rhi::kAccessDepthWrite)
    out |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  return out;
}

void StateAccess(rhi::TextureState s,
                 VkPipelineStageFlags shader,
                 VkAccessFlags& access,
                 VkPipelineStageFlags& stages) {
  using rhi::TextureState;
  switch (s) {
    case TextureState::kUndefined:
      access = 0;
      stages = 0;
      return;
    case TextureState::kGeneral:
      access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
               VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      stages = shader | VK_PIPELINE_STAGE_TRANSFER_BIT;
      return;
    case TextureState::kColorTarget:
      access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      return;
    case TextureState::kDepthTarget:
      access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      return;
    case TextureState::kDepthRead:
      access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
               VK_ACCESS_SHADER_READ_BIT;
      stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | shader;
      return;
    case TextureState::kShaderRead:
      access = VK_ACCESS_SHADER_READ_BIT;
      stages = shader;
      return;
    case TextureState::kCopySrc:
      access = VK_ACCESS_TRANSFER_READ_BIT;
      stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
      return;
    case TextureState::kCopyDst:
      access = VK_ACCESS_TRANSFER_WRITE_BIT;
      stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
      return;
  }
}

VkImageSubresourceLayers Layers(const rhi::TextureRegion& r) {
  return {ToVkAspect(r.aspect), r.mip, r.base_layer, r.layers};
}

VkBufferImageCopy ToVkCopy(const rhi::BufferTextureCopy& c) {
  VkBufferImageCopy out{};
  out.bufferOffset = c.buffer_offset;
  out.bufferRowLength = c.row_length;
  out.bufferImageHeight = c.image_height;
  out.imageSubresource = Layers(c.region);
  out.imageOffset = {c.region.x, c.region.y, c.region.z};
  out.imageExtent = {c.region.width, c.region.height, c.region.depth};
  return out;
}

VkAttachmentLoadOp ToVkLoad(rhi::LoadOp op) {
  switch (op) {
    case rhi::LoadOp::kLoad:
      return VK_ATTACHMENT_LOAD_OP_LOAD;
    case rhi::LoadOp::kClear:
      return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case rhi::LoadOp::kDontCare:
      return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
  return VK_ATTACHMENT_LOAD_OP_LOAD;
}

VkClearColorValue ToVkClear(const rhi::ClearColor& c) {
  VkClearColorValue v;
  std::memcpy(&v, &c, sizeof(v));
  return v;
}

}  // namespace

VulkanCommandList::~VulkanCommandList() {
  VkDevice dev = device_.native.device;
  if (cmd)
    vkFreeCommandBuffers(dev, device_.command_pool, 1, &cmd);
  for (VkDescriptorPool pool : transient_pools_)
    vkDestroyDescriptorPool(dev, pool, nullptr);
}

void VulkanCommandList::Begin() {
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  pipeline_ = nullptr;
  for (VkDescriptorPool pool : transient_pools_)
    vkResetDescriptorPool(device_.native.device, pool, 0);
  transient_pool_ = 0;
}

void VulkanCommandList::End() {
  vkEndCommandBuffer(cmd);
}

void VulkanCommandList::BeginRenderPass(const rhi::RenderPassDesc& pass) {
  VkRenderingAttachmentInfo colors[8]{};
  for (u32 i = 0; i < pass.color_count && i < 8; i++) {
    const rhi::ColorAttachment& c = pass.colors[i];
    colors[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colors[i].imageView = Native(c.view);
    colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colors[i].loadOp = ToVkLoad(c.load);
    colors[i].storeOp = c.store == rhi::StoreOp::kStore
                            ? VK_ATTACHMENT_STORE_OP_STORE
                            : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colors[i].clearValue.color = ToVkClear(c.clear);
  }
  const rhi::DepthAttachment& d = pass.depth;
  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  VkRenderingAttachmentInfo stencil{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  if (d.view && d.depth) {
    depth.imageView = Native(d.view);
    depth.imageLayout = d.read_only ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
                                    : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = ToVkLoad(d.depth_load);
    depth.storeOp = d.read_only ? VK_ATTACHMENT_STORE_OP_NONE
                                : VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {d.clear_depth, d.clear_stencil};
  }
  if (d.view && d.stencil) {
    stencil.imageView = Native(d.view);
    stencil.imageLayout = d.stencil_read_only
                              ? VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
                              : VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
    stencil.loadOp = ToVkLoad(d.stencil_load);
    stencil.storeOp = d.stencil_read_only ? VK_ATTACHMENT_STORE_OP_NONE
                                          : VK_ATTACHMENT_STORE_OP_STORE;
    stencil.clearValue.depthStencil = {d.clear_depth, d.clear_stencil};
  }
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{pass.x, pass.y}, {pass.width, pass.height}};
  ri.layerCount = pass.layers;
  ri.colorAttachmentCount = pass.color_count;
  ri.pColorAttachments = colors;
  ri.pDepthAttachment = d.view && d.depth ? &depth : nullptr;
  ri.pStencilAttachment = d.view && d.stencil ? &stencil : nullptr;
  device_.cmd_begin_rendering(cmd, &ri);
}

void VulkanCommandList::EndRenderPass() {
  device_.cmd_end_rendering(cmd);
}

void VulkanCommandList::SetPipeline(rhi::Pipeline* pipeline) {
  pipeline_ = static_cast<VulkanPipeline*>(pipeline);
  vkCmdBindPipeline(cmd, pipeline_->bind_point, pipeline_->pipeline);
}

void VulkanCommandList::SetBindGroup(u32 index,
                                     rhi::BindGroup* group,
                                     const u32* dynamic_offsets,
                                     u32 num_offsets) {
  VkDescriptorSet set = static_cast<VulkanBindGroup*>(group)->set;
  vkCmdBindDescriptorSets(cmd, pipeline_->bind_point, pipeline_->layout->layout,
                          index, 1, &set, num_offsets, dynamic_offsets);
}

void VulkanCommandList::PushBindGroup(u32 index,
                                      rhi::BindGroupLayout* layout,
                                      const rhi::BindingWrite* writes,
                                      u32 num_writes) {
  auto* l = static_cast<VulkanBindGroupLayout*>(layout);
  constexpr u32 kMax = 160;
  VkWriteDescriptorSet vk_writes[kMax];
  VkDescriptorImageInfo images[kMax];
  VkDescriptorBufferInfo buffers[kMax];
  u32 n = 0;
  const auto& bindings = l->desc().bindings;
  for (u32 i = 0; i < num_writes && n < kMax; i++)
    for (size_t b = 0; b < bindings.size(); b++)
      if (bindings[b].binding == writes[i].binding) {
        VkDescriptorType type = l->types[b];
        // Dynamic types cannot be pushed; the offset is in the write.
        if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
          type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
          type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        device_.FillWrite(type, writes[i], vk_writes[n], images[n], buffers[n]);
        n++;
        break;
      }
  if (l->desc().push && device_.cmd_push_descriptor_set) {
    device_.cmd_push_descriptor_set(cmd, pipeline_->bind_point,
                                    pipeline_->layout->layout, index, n,
                                    vk_writes);
    return;
  }
  // No push descriptors: a set from this list's own pools.
  VkDevice dev = device_.native.device;
  VkDescriptorSetAllocateInfo ai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &l->layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  while (true) {
    if (transient_pool_ == transient_pools_.size()) {
      const VkDescriptorPoolSize sizes[] = {
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096},
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
          {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024},
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192},
      };
      VkDescriptorPoolCreateInfo pi{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pi.maxSets = 512;
      pi.poolSizeCount = 4;
      pi.pPoolSizes = sizes;
      VkDescriptorPool pool = VK_NULL_HANDLE;
      if (vkCreateDescriptorPool(dev, &pi, nullptr, &pool) != VK_SUCCESS)
        return;
      transient_pools_.push_back(pool);
    }
    ai.descriptorPool = transient_pools_[transient_pool_];
    if (vkAllocateDescriptorSets(dev, &ai, &set) == VK_SUCCESS)
      break;
    transient_pool_++;
  }
  for (u32 i = 0; i < n; i++)
    vk_writes[i].dstSet = set;
  vkUpdateDescriptorSets(dev, n, vk_writes, 0, nullptr);
  vkCmdBindDescriptorSets(cmd, pipeline_->bind_point, pipeline_->layout->layout,
                          index, 1, &set, 0, nullptr);
}

void VulkanCommandList::SetPushConstants(u32 offset,
                                         u32 bytes,
                                         const void* data) {
  vkCmdPushConstants(cmd, pipeline_->layout->layout,
                     pipeline_->layout->push_stages, offset, bytes, data);
}

void VulkanCommandList::SetVertexBuffers(u32 first,
                                         u32 count,
                                         rhi::Buffer* const* buffers,
                                         const u64* offsets) {
  VkBuffer vk[16];
  VkDeviceSize offs[16];
  count = count > 16 ? 16 : count;
  for (u32 i = 0; i < count; i++) {
    vk[i] = Buf(buffers[i])->buffer;
    offs[i] = offsets[i];
  }
  vkCmdBindVertexBuffers(cmd, first, count, vk, offs);
}

void VulkanCommandList::SetIndexBuffer(rhi::Buffer* buffer,
                                       u64 offset,
                                       rhi::IndexType type) {
  vkCmdBindIndexBuffer(cmd, Buf(buffer)->buffer, offset,
                       type == rhi::IndexType::kUint32 ? VK_INDEX_TYPE_UINT32
                                                       : VK_INDEX_TYPE_UINT16);
}

void VulkanCommandList::SetViewport(float x,
                                    float y,
                                    float width,
                                    float height,
                                    float min_depth,
                                    float max_depth) {
  const VkViewport vp{x, y, width, height, min_depth, max_depth};
  vkCmdSetViewport(cmd, 0, 1, &vp);
}

void VulkanCommandList::SetScissor(i32 x, i32 y, u32 width, u32 height) {
  const VkRect2D r{{x, y}, {width, height}};
  vkCmdSetScissor(cmd, 0, 1, &r);
}

void VulkanCommandList::SetBlendConstants(const float rgba[4]) {
  vkCmdSetBlendConstants(cmd, rgba);
}

void VulkanCommandList::Draw(u32 vertex_count,
                             u32 instance_count,
                             u32 first_vertex,
                             u32 first_instance) {
  vkCmdDraw(cmd, vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanCommandList::DrawIndexed(u32 index_count,
                                    u32 instance_count,
                                    u32 first_index,
                                    i32 vertex_offset,
                                    u32 first_instance) {
  vkCmdDrawIndexed(cmd, index_count, instance_count, first_index, vertex_offset,
                   first_instance);
}

void VulkanCommandList::DrawMeshTasks(u32 x, u32 y, u32 z) {
  device_.cmd_draw_mesh_tasks(cmd, x, y, z);
}

void VulkanCommandList::Dispatch(u32 x, u32 y, u32 z) {
  vkCmdDispatch(cmd, x, y, z);
}

void VulkanCommandList::DispatchIndirect(rhi::Buffer* args, u64 offset) {
  vkCmdDispatchIndirect(cmd, Buf(args)->buffer, offset);
}

void VulkanCommandList::DispatchBase(u32 bx,
                                     u32 by,
                                     u32 bz,
                                     u32 x,
                                     u32 y,
                                     u32 z) {
  vkCmdDispatchBase(cmd, bx, by, bz, x, y, z);
}

void VulkanCommandList::ClearAttachment(u32 attachment,
                                        const rhi::ClearColor& color,
                                        float depth,
                                        u8 stencil,
                                        u8 aspect,
                                        i32 x,
                                        i32 y,
                                        u32 width,
                                        u32 height) {
  VkClearAttachment ca{};
  ca.aspectMask = ToVkAspect(aspect);
  if (attachment != ~0u) {
    ca.colorAttachment = attachment;
    ca.clearValue.color = ToVkClear(color);
  } else {
    ca.clearValue.depthStencil = {depth, stencil};
  }
  VkClearRect rect{{{x, y}, {width, height}}, 0, 1};
  vkCmdClearAttachments(cmd, 1, &ca, 1, &rect);
}

void VulkanCommandList::CopyBuffer(rhi::Buffer* dst,
                                   u64 dst_offset,
                                   rhi::Buffer* src,
                                   u64 src_offset,
                                   u64 bytes) {
  const VkBufferCopy region{src_offset, dst_offset, bytes};
  vkCmdCopyBuffer(cmd, Buf(src)->buffer, Buf(dst)->buffer, 1, &region);
}

void VulkanCommandList::CopyBufferToTexture(
    rhi::Texture* dst,
    rhi::Buffer* src,
    const rhi::BufferTextureCopy* regions,
    u32 count) {
  VkBufferImageCopy vk[16];
  for (u32 done = 0; done < count;) {
    const u32 n = count - done > 16 ? 16 : count - done;
    for (u32 i = 0; i < n; i++)
      vk[i] = ToVkCopy(regions[done + i]);
    vkCmdCopyBufferToImage(cmd, Buf(src)->buffer, Tex(dst)->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, n, vk);
    done += n;
  }
}

void VulkanCommandList::CopyTextureToBuffer(
    rhi::Buffer* dst,
    rhi::Texture* src,
    const rhi::BufferTextureCopy* regions,
    u32 count) {
  VkBufferImageCopy vk[16];
  for (u32 done = 0; done < count;) {
    const u32 n = count - done > 16 ? 16 : count - done;
    for (u32 i = 0; i < n; i++)
      vk[i] = ToVkCopy(regions[done + i]);
    vkCmdCopyImageToBuffer(cmd, Tex(src)->image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           Buf(dst)->buffer, n, vk);
    done += n;
  }
}

void VulkanCommandList::CopyTexture(rhi::Texture* dst,
                                    const rhi::TextureRegion& dst_region,
                                    rhi::Texture* src,
                                    const rhi::TextureRegion& src_region) {
  VkImageCopy c{};
  c.srcSubresource = Layers(src_region);
  c.srcOffset = {src_region.x, src_region.y, src_region.z};
  c.dstSubresource = Layers(dst_region);
  c.dstOffset = {dst_region.x, dst_region.y, dst_region.z};
  c.extent = {src_region.width, src_region.height, src_region.depth};
  vkCmdCopyImage(cmd, Tex(src)->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 Tex(dst)->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
}

void VulkanCommandList::BlitTexture(rhi::Texture* dst,
                                    const rhi::TextureRegion& dst_region,
                                    rhi::Texture* src,
                                    const rhi::TextureRegion& src_region,
                                    rhi::Filter filter) {
  VkImageBlit b{};
  b.srcSubresource = Layers(src_region);
  b.srcOffsets[0] = {src_region.x, src_region.y, src_region.z};
  b.srcOffsets[1] = {src_region.x + static_cast<i32>(src_region.width),
                     src_region.y + static_cast<i32>(src_region.height),
                     src_region.z + static_cast<i32>(src_region.depth)};
  b.dstSubresource = Layers(dst_region);
  b.dstOffsets[0] = {dst_region.x, dst_region.y, dst_region.z};
  b.dstOffsets[1] = {dst_region.x + static_cast<i32>(dst_region.width),
                     dst_region.y + static_cast<i32>(dst_region.height),
                     dst_region.z + static_cast<i32>(dst_region.depth)};
  vkCmdBlitImage(
      cmd, Tex(src)->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      Tex(dst)->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
      filter == rhi::Filter::kLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

void VulkanCommandList::ClearTexture(rhi::Texture* texture,
                                     rhi::TextureState state,
                                     const rhi::TextureRange& range,
                                     const rhi::ClearColor& color) {
  const VkClearColorValue v = ToVkClear(color);
  const VkImageSubresourceRange r{ToVkAspect(range.aspect), range.base_mip,
                                  range.mips, range.base_layer, range.layers};
  vkCmdClearColorImage(cmd, Tex(texture)->image,
                       ToVkLayout(state, range.aspect), &v, 1, &r);
}

void VulkanCommandList::ClearDepthStencil(rhi::Texture* texture,
                                          rhi::TextureState state,
                                          const rhi::TextureRange& range,
                                          float depth,
                                          u8 stencil) {
  const VkClearDepthStencilValue v{depth, stencil};
  const VkImageSubresourceRange r{ToVkAspect(range.aspect), range.base_mip,
                                  range.mips, range.base_layer, range.layers};
  vkCmdClearDepthStencilImage(cmd, Tex(texture)->image,
                              ToVkLayout(state, range.aspect), &v, 1, &r);
}

void VulkanCommandList::FillBuffer(rhi::Buffer* buffer,
                                   u64 offset,
                                   u64 bytes,
                                   u32 value) {
  vkCmdFillBuffer(cmd, Buf(buffer)->buffer, offset, bytes, value);
}

void VulkanCommandList::UpdateBuffer(rhi::Buffer* buffer,
                                     u64 offset,
                                     u64 bytes,
                                     const void* data) {
  vkCmdUpdateBuffer(cmd, Buf(buffer)->buffer, offset, bytes, data);
}

void VulkanCommandList::Barrier(u32 src_access,
                                u32 dst_access,
                                const rhi::TextureBarrier* textures,
                                u32 num_textures) {
  const VkPipelineStageFlags shader = device_.ShaderStages();
  VkPipelineStageFlags src_stages = StagesFor(src_access, shader);
  VkPipelineStageFlags dst_stages = StagesFor(dst_access, shader);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = AccessFor(src_access);
  mb.dstAccessMask = AccessFor(dst_access);
  constexpr u32 kMax = 32;
  VkImageMemoryBarrier ib[kMax];
  const u32 n = num_textures > kMax ? kMax : num_textures;
  for (u32 i = 0; i < n; i++) {
    const rhi::TextureBarrier& t = textures[i];
    VkAccessFlags src_a, dst_a;
    VkPipelineStageFlags src_s, dst_s;
    StateAccess(t.before, shader, src_a, src_s);
    StateAccess(t.after, shader, dst_a, dst_s);
    src_stages |= src_s;
    dst_stages |= dst_s;
    ib[i] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib[i].srcAccessMask = src_a;
    ib[i].dstAccessMask = dst_a;
    ib[i].oldLayout = ToVkLayout(t.before, t.range.aspect);
    ib[i].newLayout = ToVkLayout(t.after, t.range.aspect);
    ib[i].srcQueueFamilyIndex = ib[i].dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    ib[i].image = Tex(t.texture)->image;
    ib[i].subresourceRange = {ToVkAspect(t.range.aspect), t.range.base_mip,
                              t.range.mips, t.range.base_layer, t.range.layers};
  }
  if (!src_stages)
    src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  if (!dst_stages)
    dst_stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  const bool memory = mb.srcAccessMask || mb.dstAccessMask;
  vkCmdPipelineBarrier(cmd, src_stages, dst_stages, 0, memory ? 1 : 0, &mb, 0,
                       nullptr, n, ib);
  if (num_textures > kMax)
    Barrier(0, 0, textures + kMax, num_textures - kMax);
}

void VulkanCommandList::ResetTimestamps(rhi::TimestampPool* pool,
                                        u32 first,
                                        u32 count) {
  vkCmdResetQueryPool(cmd, static_cast<VulkanTimestampPool*>(pool)->pool, first,
                      count);
}

void VulkanCommandList::WriteTimestamp(rhi::TimestampPool* pool,
                                       u32 index,
                                       bool start) {
  vkCmdWriteTimestamp(cmd,
                      start ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                            : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      static_cast<VulkanTimestampPool*>(pool)->pool, index);
}

void VulkanCommandList::PushLabel(const char* label) {
  if (!device_.cmd_begin_label)
    return;
  VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  l.pLabelName = label;
  device_.cmd_begin_label(cmd, &l);
}

void VulkanCommandList::PopLabel() {
  if (device_.cmd_end_label)
    device_.cmd_end_label(cmd);
}

void VulkanCommandList::InsertLabel(const char* label) {
  if (!device_.cmd_insert_label)
    return;
  VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  l.pLabelName = label;
  device_.cmd_insert_label(cmd, &l);
}

void VulkanCommandList::Checkpoint(u64 marker) {
  if (device_.cmd_set_checkpoint)
    device_.cmd_set_checkpoint(
        cmd, reinterpret_cast<const void*>(static_cast<uintptr_t>(marker)));
}

}  // namespace gpu::vk::rhi_impl
