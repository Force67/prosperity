/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The Vulkan objects behind the rhi classes. Private to vk_rhi_*.cc.

#include <vulkan/vulkan.h>

#include "base/arch.h"
#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "base/threading/mutex.h"
#include "gpu/rhi/device.h"
#include "gpu/vulkan/vk_memory_span.h"
#include "gpu/vulkan/vk_rhi.h"

namespace gpu::vk::rhi_impl {

class VulkanDevice;

struct Allocation {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;
  u32 type = 0;
  bool dedicated = false;
};

// Device-local image memory: small images share 256 MiB blocks.
class ImageAllocator {
 public:
  bool Allocate(VulkanDevice& device, VkImage image, Allocation& out);
  void Free(VulkanDevice& device, Allocation& allocation);

 private:
  struct Block {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    u32 type = 0;
    MemorySpanAllocator spans;
  };
  base::Mutex mutex_;
  base::Vector<Block> blocks_;
};

class VulkanBuffer final : public rhi::Buffer {
 public:
  VulkanBuffer(const rhi::BufferDesc& desc) { desc_ = desc; }
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  // A sparse buffer's committed pages, and the blocks they are carved from.
  base::HashSet<u64> committed;
  base::Vector<VkDeviceMemory> blocks;
  u64 block_used = 0;
  u32 memory_type = 0;
  void SetMapped(u8* p) { mapped_ = p; }
  void SetAddress(u64 a) { address_ = a; }
};

class VulkanTexture final : public rhi::Texture {
 public:
  VulkanTexture(const rhi::TextureDesc& desc) { desc_ = desc; }
  VkImage image = VK_NULL_HANDLE;
  Allocation allocation;
  VkImageAspectFlags aspects = 0;
};

class VulkanView final : public rhi::TextureView {
 public:
  VulkanView(rhi::Texture* texture, const rhi::TextureViewDesc& desc) {
    texture_ = texture;
    desc_ = desc;
  }
  VkImageView view = VK_NULL_HANDLE;
};

class VulkanSampler final : public rhi::Sampler {
 public:
  VkSampler sampler = VK_NULL_HANDLE;
};

class VulkanBindGroupLayout final : public rhi::BindGroupLayout {
 public:
  VulkanBindGroupLayout(const rhi::BindGroupLayoutDesc& desc) { desc_ = desc; }
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  // Per binding: the descriptor type, in desc().bindings order.
  base::Vector<VkDescriptorType> types;
  u32 dynamic_count = 0;
};

class VulkanBindGroup final : public rhi::BindGroup {
 public:
  VulkanBindGroupLayout* layout = nullptr;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
};

class VulkanPipelineLayout final : public rhi::PipelineLayout {
 public:
  VulkanPipelineLayout(const rhi::PipelineLayoutDesc& desc) { desc_ = desc; }
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkShaderStageFlags push_stages = 0;
};

class VulkanPipeline final : public rhi::Pipeline {
 public:
  VkPipeline pipeline = VK_NULL_HANDLE;
  VulkanPipelineLayout* layout = nullptr;
  VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
};

class VulkanTimestampPool final : public rhi::TimestampPool {
 public:
  VkQueryPool pool = VK_NULL_HANDLE;
  u32 count = 0;
};

class VulkanCommandList final : public rhi::CommandList {
 public:
  explicit VulkanCommandList(VulkanDevice& device) : device_(device) {}
  ~VulkanCommandList() override;

  VkCommandBuffer cmd = VK_NULL_HANDLE;

  void Begin() override;
  void End() override;
  void BeginRenderPass(const rhi::RenderPassDesc& pass) override;
  void EndRenderPass() override;
  void SetPipeline(rhi::Pipeline* pipeline) override;
  void SetBindGroup(u32 index,
                    rhi::BindGroup* group,
                    const u32* dynamic_offsets,
                    u32 num_offsets) override;
  void PushBindGroup(u32 index,
                     rhi::BindGroupLayout* layout,
                     const rhi::BindingWrite* writes,
                     u32 num_writes) override;
  void SetPushConstants(u32 offset, u32 bytes, const void* data) override;
  void SetVertexBuffers(u32 first,
                        u32 count,
                        rhi::Buffer* const* buffers,
                        const u64* offsets) override;
  void SetIndexBuffer(rhi::Buffer* buffer,
                      u64 offset,
                      rhi::IndexType type) override;
  void SetViewport(float x,
                   float y,
                   float width,
                   float height,
                   float min_depth,
                   float max_depth) override;
  void SetScissor(i32 x, i32 y, u32 width, u32 height) override;
  void SetBlendConstants(const float rgba[4]) override;
  void Draw(u32 vertex_count,
            u32 instance_count,
            u32 first_vertex,
            u32 first_instance) override;
  void DrawIndexed(u32 index_count,
                   u32 instance_count,
                   u32 first_index,
                   i32 vertex_offset,
                   u32 first_instance) override;
  void DrawMeshTasks(u32 x, u32 y, u32 z) override;
  void Dispatch(u32 x, u32 y, u32 z) override;
  void DispatchBase(u32 bx, u32 by, u32 bz, u32 x, u32 y, u32 z) override;
  void DispatchIndirect(rhi::Buffer* args, u64 offset) override;
  void ClearAttachment(u32 attachment,
                       const rhi::ClearColor& color,
                       float depth,
                       u8 stencil,
                       u8 aspect,
                       i32 x,
                       i32 y,
                       u32 width,
                       u32 height) override;
  void CopyBuffer(rhi::Buffer* dst,
                  u64 dst_offset,
                  rhi::Buffer* src,
                  u64 src_offset,
                  u64 bytes) override;
  void CopyBufferToTexture(rhi::Texture* dst,
                           rhi::Buffer* src,
                           const rhi::BufferTextureCopy* regions,
                           u32 count) override;
  void CopyTextureToBuffer(rhi::Buffer* dst,
                           rhi::Texture* src,
                           const rhi::BufferTextureCopy* regions,
                           u32 count) override;
  void CopyTexture(rhi::Texture* dst,
                   const rhi::TextureRegion& dst_region,
                   rhi::Texture* src,
                   const rhi::TextureRegion& src_region) override;
  void BlitTexture(rhi::Texture* dst,
                   const rhi::TextureRegion& dst_region,
                   rhi::Texture* src,
                   const rhi::TextureRegion& src_region,
                   rhi::Filter filter) override;
  void ClearTexture(rhi::Texture* texture,
                    rhi::TextureState state,
                    const rhi::TextureRange& range,
                    const rhi::ClearColor& color) override;
  void ClearDepthStencil(rhi::Texture* texture,
                         rhi::TextureState state,
                         const rhi::TextureRange& range,
                         float depth,
                         u8 stencil) override;
  void FillBuffer(rhi::Buffer* buffer,
                  u64 offset,
                  u64 bytes,
                  u32 value) override;
  void UpdateBuffer(rhi::Buffer* buffer,
                    u64 offset,
                    u64 bytes,
                    const void* data) override;
  void Barrier(u32 src_access,
               u32 dst_access,
               const rhi::TextureBarrier* textures,
               u32 num_textures) override;
  void ResetTimestamps(rhi::TimestampPool* pool, u32 first, u32 count) override;
  void WriteTimestamp(rhi::TimestampPool* pool, u32 index, bool start) override;
  void PushLabel(const char* label) override;
  void PopLabel() override;
  void InsertLabel(const char* label) override;
  void Checkpoint(u64 marker) override;

 private:
  VulkanDevice& device_;
  VulkanPipeline* pipeline_ = nullptr;
  // Sets for PushBindGroup on a device without push descriptors; reset at
  // Begin, which is only legal once the previous submission retired.
  base::Vector<VkDescriptorPool> transient_pools_;
  u32 transient_pool_ = 0;
};

class VulkanDevice final : public rhi::Device {
 public:
  ~VulkanDevice() override;
  bool Init(const VulkanOptions& options);

  const rhi::Caps& caps() const override { return caps_; }
  rhi::Buffer* CreateBuffer(const rhi::BufferDesc& desc) override;
  rhi::Texture* CreateTexture(const rhi::TextureDesc& desc) override;
  rhi::TextureView* CreateView(rhi::Texture* texture,
                               const rhi::TextureViewDesc& desc) override;
  rhi::Sampler* CreateSampler(const rhi::SamplerDesc& desc) override;
  rhi::BindGroupLayout* CreateBindGroupLayout(
      const rhi::BindGroupLayoutDesc& desc) override;
  rhi::BindGroup* CreateBindGroup(const rhi::BindGroupDesc& desc) override;
  void UpdateBindGroup(rhi::BindGroup* group,
                       const rhi::BindingWrite* writes,
                       u32 count) override;
  rhi::PipelineLayout* CreatePipelineLayout(
      const rhi::PipelineLayoutDesc& desc) override;
  rhi::Pipeline* CreateGraphicsPipeline(
      const rhi::GraphicsPipelineDesc& desc) override;
  rhi::Pipeline* CreateComputePipeline(
      const rhi::ComputePipelineDesc& desc) override;
  rhi::TimestampPool* CreateTimestampPool(u32 count) override;
  rhi::CommandList* CreateCommandList() override;
  void Destroy(rhi::Object* object) override;
  void SetName(rhi::Object* object, const char* name) override;
  bool SupportsFormat(rhi::Format format, u32 usage) const override;
  bool CommitSparse(rhi::Buffer* buffer, u64 offset, u64 bytes) override;
  bool SupportsBlit(rhi::Format src, rhi::Format dst) const override;
  void QueryMemoryBudget(u64* used, u64* budget) const override;
  u64 Submit(rhi::CommandList* const* lists, u32 count) override;
  bool IsComplete(u64 submission) override;
  bool Wait(u64 submission, u64 timeout_ns) override;
  void WaitIdle() override;
  u64 LastSubmission() const override { return submitted_; }
  bool ReadTimestamps(rhi::TimestampPool* pool,
                      u32 first,
                      u32 count,
                      u64* out) override;
  void Maintain() override;
  void ReportDeviceLoss() override;
  // RenderDoc names a Vulkan device by its instance's dispatch pointer.
  void* CaptureHandle() const override {
    return *reinterpret_cast<void**>(native.instance);
  }

  // Fills `write` for one binding; `info` storage must outlive the update.
  void FillWrite(VkDescriptorType type,
                 const rhi::BindingWrite& in,
                 VkWriteDescriptorSet& write,
                 VkDescriptorImageInfo& image,
                 VkDescriptorBufferInfo& buffer) const;
  u32 FindMemoryType(u32 bits, VkMemoryPropertyFlags want) const;
  VkPipelineStageFlags ShaderStages() const { return shader_stages_; }

  NativeDevice native;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  PFN_vkCmdBeginRenderingKHR cmd_begin_rendering = nullptr;
  PFN_vkCmdEndRenderingKHR cmd_end_rendering = nullptr;
  PFN_vkCmdPushDescriptorSetKHR cmd_push_descriptor_set = nullptr;
  PFN_vkCmdDrawMeshTasksEXT cmd_draw_mesh_tasks = nullptr;
  PFN_vkCmdSetCheckpointNV cmd_set_checkpoint = nullptr;
  PFN_vkSetDebugUtilsObjectNameEXT set_object_name = nullptr;
  PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin_label = nullptr;
  PFN_vkCmdEndDebugUtilsLabelEXT cmd_end_label = nullptr;
  PFN_vkCmdInsertDebugUtilsLabelEXT cmd_insert_label = nullptr;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  base::Mutex queue_mutex;

 private:
  VkDescriptorPool GrowSetPool();
  void SavePipelineCache(bool force);
  // A build that took a second or more is written out now, off this thread:
  // waiting for a quiet frame end loses it to a run that never has one.
  void SaveAfterSlowBuild(u64 started);
  void WritePipelineCache(bool force);

  rhi::Caps caps_;
  base::String device_name_;
  base::String pipeline_cache_path_;
  VkPipelineStageFlags shader_stages_ = 0;
  VkSemaphore timeline_ = VK_NULL_HANDLE;
  u64 submitted_ = 0;
  ImageAllocator images_;
  base::Mutex set_pool_mutex_;
  base::Vector<VkDescriptorPool> set_pools_;
  base::Atomic<u64> last_pipeline_build_ns_{0};
  u64 last_cache_write_ns_ = 0;
  size_t last_cache_size_ = 0;
  // A periodic save runs on a thread of its own; a forced one waits for it.
  base::Atomic<bool> cache_saving_{false};
  base::Atomic<bool> save_requested_{false};
  bool fault_reported_ = false;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  float max_lod_bias_ = 0.0f;
  float max_anisotropy_ = 1.0f;
};

inline VulkanDevice& Impl(rhi::Device* device) {
  return *static_cast<VulkanDevice*>(device);
}

VkShaderStageFlags ToVkStages(u32 stages);
VkDescriptorType ToVkDescriptorType(rhi::BindingType type);
VkImageAspectFlags ToVkAspect(u8 aspect);

}  // namespace gpu::vk::rhi_impl
