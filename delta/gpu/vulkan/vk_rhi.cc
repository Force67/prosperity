/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_rhi.h"

#include <vulkan/vulkan.h>

#include <cstring>
#include <string>
#include <vector>

#include <base/logging.h>

namespace gpu::vk {
namespace {

using namespace gpu::rhi;  // NOLINT: this file is one backend of that seam

// Bail out of the enclosing function on a Vulkan error.
#define RHI_VK(expr, on_fail)                                     \
  do {                                                            \
    const VkResult _r = (expr);                                   \
    if (_r != VK_SUCCESS) {                                       \
      BASE_LOGI("rhivk", "{} failed: {}", #expr, (int)_r);        \
      on_fail;                                                    \
    }                                                             \
  } while (0)

VkFormat ToVk(Format f) {
  switch (f) {
    case Format::R8Unorm: return VK_FORMAT_R8_UNORM;
    case Format::R8Uint: return VK_FORMAT_R8_UINT;
    case Format::RG8Unorm: return VK_FORMAT_R8G8_UNORM;
    case Format::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::BGRA8Srgb: return VK_FORMAT_B8G8R8A8_SRGB;
    case Format::R16Uint: return VK_FORMAT_R16_UINT;
    case Format::R16Float: return VK_FORMAT_R16_SFLOAT;
    case Format::RG16Float: return VK_FORMAT_R16G16_SFLOAT;
    case Format::RGBA16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::RGBA16Unorm: return VK_FORMAT_R16G16B16A16_UNORM;
    case Format::R32Uint: return VK_FORMAT_R32_UINT;
    case Format::R32Float: return VK_FORMAT_R32_SFLOAT;
    case Format::RG32Float: return VK_FORMAT_R32G32_SFLOAT;
    case Format::RGB32Float: return VK_FORMAT_R32G32B32_SFLOAT;
    case Format::RGBA32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case Format::RGB10A2Unorm: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Format::RG11B10Float: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case Format::D16Unorm: return VK_FORMAT_D16_UNORM;
    case Format::D32Float: return VK_FORMAT_D32_SFLOAT;
    case Format::D32FloatS8Uint: return VK_FORMAT_D32_SFLOAT_S8_UINT;
    case Format::BC1Unorm: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case Format::BC2Unorm: return VK_FORMAT_BC2_UNORM_BLOCK;
    case Format::BC3Unorm: return VK_FORMAT_BC3_UNORM_BLOCK;
    case Format::BC4Unorm: return VK_FORMAT_BC4_UNORM_BLOCK;
    case Format::BC5Unorm: return VK_FORMAT_BC5_UNORM_BLOCK;
    case Format::BC7Unorm: return VK_FORMAT_BC7_UNORM_BLOCK;
    case Format::Unknown:
    case Format::kCount: break;
  }
  return VK_FORMAT_UNDEFINED;
}

VkCompareOp ToVk(CompareOp op) {
  switch (op) {
    case CompareOp::Never: return VK_COMPARE_OP_NEVER;
    case CompareOp::Less: return VK_COMPARE_OP_LESS;
    case CompareOp::Equal: return VK_COMPARE_OP_EQUAL;
    case CompareOp::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case CompareOp::Greater: return VK_COMPARE_OP_GREATER;
    case CompareOp::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
    case CompareOp::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case CompareOp::Always: return VK_COMPARE_OP_ALWAYS;
  }
  return VK_COMPARE_OP_ALWAYS;
}

VkBlendFactor ToVk(BlendFactor f) {
  switch (f) {
    case BlendFactor::Zero: return VK_BLEND_FACTOR_ZERO;
    case BlendFactor::One: return VK_BLEND_FACTOR_ONE;
    case BlendFactor::SrcColor: return VK_BLEND_FACTOR_SRC_COLOR;
    case BlendFactor::OneMinusSrcColor:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case BlendFactor::DstColor: return VK_BLEND_FACTOR_DST_COLOR;
    case BlendFactor::OneMinusDstColor:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case BlendFactor::SrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
    case BlendFactor::OneMinusSrcAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case BlendFactor::DstAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
    case BlendFactor::OneMinusDstAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case BlendFactor::ConstantColor: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case BlendFactor::OneMinusConstantColor:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case BlendFactor::ConstantAlpha: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case BlendFactor::OneMinusConstantAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case BlendFactor::SrcAlphaSaturate:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
  }
  return VK_BLEND_FACTOR_ONE;
}

VkBlendOp ToVk(BlendOp op) {
  switch (op) {
    case BlendOp::Add: return VK_BLEND_OP_ADD;
    case BlendOp::Subtract: return VK_BLEND_OP_SUBTRACT;
    case BlendOp::ReverseSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case BlendOp::Min: return VK_BLEND_OP_MIN;
    case BlendOp::Max: return VK_BLEND_OP_MAX;
  }
  return VK_BLEND_OP_ADD;
}

VkPrimitiveTopology ToVk(PrimitiveTopology t) {
  switch (t) {
    case PrimitiveTopology::PointList: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case PrimitiveTopology::LineList: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case PrimitiveTopology::LineStrip: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case PrimitiveTopology::TriangleList:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case PrimitiveTopology::TriangleStrip:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  }
  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkFilter ToVkFilter(FilterMode m) {
  return m == FilterMode::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerAddressMode ToVk(AddressMode m) {
  switch (m) {
    case AddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case AddressMode::MirrorRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case AddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case AddressMode::ClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  }
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkDescriptorType ToVk(BindingType type, bool dynamic) {
  switch (type) {
    case BindingType::UniformBuffer:
      return dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                     : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case BindingType::StorageBuffer:
      return dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case BindingType::SampledTexture:
      return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case BindingType::StorageTexture: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case BindingType::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
  }
  return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}

VkShaderStageFlags ToVkStages(u32 stages) {
  VkShaderStageFlags out = 0;
  if (stages & kStageVertex)
    out |= VK_SHADER_STAGE_VERTEX_BIT;
  if (stages & kStagePixel)
    out |= VK_SHADER_STAGE_FRAGMENT_BIT;
  if (stages & kStageCompute)
    out |= VK_SHADER_STAGE_COMPUTE_BIT;
  return out;
}

VkAttachmentLoadOp ToVk(LoadOp op) {
  switch (op) {
    case LoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::Discard: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
  return VK_ATTACHMENT_LOAD_OP_LOAD;
}

VkAttachmentStoreOp ToVk(StoreOp op) {
  return op == StoreOp::Store ? VK_ATTACHMENT_STORE_OP_STORE
                              : VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

// What a state means to the pipeline: the layout an image holds in it, and the
// access and stage a barrier into or out of it has to name.
struct StateInfo {
  VkImageLayout layout;
  VkAccessFlags access;
  VkPipelineStageFlags stage;
};

StateInfo StateOf(ResourceState s) {
  constexpr VkPipelineStageFlags kAllShaders =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  switch (s) {
    case ResourceState::Undefined:
      return {VK_IMAGE_LAYOUT_UNDEFINED, 0,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    case ResourceState::Common:
      return {VK_IMAGE_LAYOUT_GENERAL, 0,
              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    case ResourceState::RenderTarget:
      return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    case ResourceState::DepthWrite:
      return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT};
    case ResourceState::DepthRead:
      return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_SHADER_READ_BIT,
              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | kAllShaders};
    case ResourceState::ShaderRead:
      return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT,
              kAllShaders};
    case ResourceState::UnorderedAccess:
      return {VK_IMAGE_LAYOUT_GENERAL,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
              kAllShaders};
    case ResourceState::CopySrc:
      return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ResourceState::CopyDst:
      return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ResourceState::VertexInput:
      return {VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
    case ResourceState::IndexInput:
      return {VK_IMAGE_LAYOUT_UNDEFINED, VK_ACCESS_INDEX_READ_BIT,
              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
    case ResourceState::IndirectArgument:
      return {VK_IMAGE_LAYOUT_UNDEFINED,
              VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
              VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT};
  }
  return {VK_IMAGE_LAYOUT_GENERAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
}

VkImageAspectFlags AspectOf(Format f) {
  if (!FormatIsDepth(f))
    return VK_IMAGE_ASPECT_COLOR_BIT;
  return VK_IMAGE_ASPECT_DEPTH_BIT |
         (FormatHasStencil(f) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

class VulkanDevice;

class VulkanBuffer : public Buffer {
 public:
  VulkanBuffer(VulkanDevice& device, const BufferDesc& desc);
  ~VulkanBuffer() override;

  void* Map() override { return mapped_; }

  VkBuffer handle() const { return buffer_; }  // NOLINT: accessor
  bool valid() const { return buffer_ != VK_NULL_HANDLE; }  // NOLINT

 private:
  VulkanDevice& device_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  void* mapped_ = nullptr;
};

// A view of a texture, made on demand and kept for the texture's life. The key
// is everything the view is built from, so two bindings that want the same
// view share one.
struct ViewKey {
  VkFormat format;
  u32 base_mip, mips, base_layer, layers;
  VkImageViewType type;
  VkImageAspectFlags aspect;

  bool operator==(const ViewKey& o) const {
    return format == o.format && base_mip == o.base_mip && mips == o.mips &&
           base_layer == o.base_layer && layers == o.layers &&
           type == o.type && aspect == o.aspect;
  }
};

class VulkanTexture : public Texture {
 public:
  VulkanTexture(VulkanDevice& device, const TextureDesc& desc);
  ~VulkanTexture() override;

  VkImage image() const { return image_; }  // NOLINT: accessor
  bool valid() const { return image_ != VK_NULL_HANDLE; }  // NOLINT

  // View covering one mip and layer, in the texture's own format: what a
  // render pass attaches.
  VkImageView AttachmentView(u32 mip, u32 layer);
  // View a shader reads or writes through.
  VkImageView ShaderView(Format format,
                         u32 base_mip,
                         u32 mips,
                         u32 base_layer,
                         u32 layers);

 private:
  VkImageView GetView(const ViewKey& key);

  VulkanDevice& device_;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  std::vector<std::pair<ViewKey, VkImageView>> views_;
};

class VulkanSampler : public Sampler {
 public:
  VulkanSampler(VulkanDevice& device, const SamplerDesc& desc);
  ~VulkanSampler() override;

  VkSampler handle() const { return sampler_; }  // NOLINT: accessor

 private:
  VulkanDevice& device_;
  VkSampler sampler_ = VK_NULL_HANDLE;
};

class VulkanShaderModule : public ShaderModule {
 public:
  VulkanShaderModule(VulkanDevice& device, const ShaderDesc& desc);
  ~VulkanShaderModule() override;

  VkShaderModule handle() const { return module_; }  // NOLINT: accessor
  const char* entry() const { return entry_.c_str(); }  // NOLINT: accessor

 private:
  VulkanDevice& device_;
  VkShaderModule module_ = VK_NULL_HANDLE;
  std::string entry_;
};

class VulkanBindGroupLayout : public BindGroupLayout {
 public:
  VulkanBindGroupLayout(VulkanDevice& device,
                        const BindGroupLayoutDesc& desc);
  ~VulkanBindGroupLayout() override;

  VkDescriptorSetLayout handle() const { return layout_; }  // NOLINT

 private:
  VulkanDevice& device_;
  VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
};

class VulkanBindGroup : public BindGroup {
 public:
  VulkanBindGroup(VulkanDevice& device, const BindGroupDesc& desc);
  ~VulkanBindGroup() override;

  VkDescriptorSet handle() const { return set_; }  // NOLINT: accessor

 private:
  VulkanDevice& device_;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
};

class VulkanPipeline : public Pipeline {
 public:
  VulkanPipeline(VulkanDevice& device, const GraphicsPipelineDesc& desc);
  VulkanPipeline(VulkanDevice& device, const ComputePipelineDesc& desc);
  ~VulkanPipeline() override;

  VkPipeline handle() const { return pipeline_; }  // NOLINT: accessor
  VkPipelineLayout layout() const { return layout_; }  // NOLINT: accessor
  VkPipelineBindPoint bind_point() const { return bind_point_; }  // NOLINT
  VkShaderStageFlags push_stages() const { return push_stages_; }  // NOLINT
  bool valid() const { return pipeline_ != VK_NULL_HANDLE; }  // NOLINT

 private:
  bool CreateLayout(const BindGroupLayout* const* groups,
                    u32 num_groups,
                    u32 push_bytes,
                    VkShaderStageFlags push_stages);

  VulkanDevice& device_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipelineBindPoint bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;
  VkShaderStageFlags push_stages_ = 0;
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice& device, VkCommandBuffer cmd);
  ~VulkanCommandList() override;

  void Begin() override;
  void End() override;
  void BeginRenderPass(const RenderPassDesc& pass) override;
  void EndRenderPass() override;
  void SetPipeline(Pipeline* pipeline) override;
  void SetBindGroup(u32 index,
                    BindGroup* group,
                    const u32* offsets,
                    u32 num_offsets) override;
  void SetPushConstants(u32 offset, u32 bytes, const void* data) override;
  void SetVertexBuffer(u32 slot, Buffer* buffer, u64 offset) override;
  void SetIndexBuffer(Buffer* buffer, u64 offset, IndexType type) override;
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
  void Dispatch(u32 groups_x, u32 groups_y, u32 groups_z) override;
  void CopyBuffer(Buffer* dst,
                  u64 dst_offset,
                  Buffer* src,
                  u64 src_offset,
                  u64 bytes) override;
  void CopyBufferToTexture(Texture* dst,
                           const TextureRegion& region,
                           Buffer* src,
                           u64 src_offset,
                           u32 src_row_pitch) override;
  void CopyTextureToBuffer(Buffer* dst,
                           u64 dst_offset,
                           u32 dst_row_pitch,
                           Texture* src,
                           const TextureRegion& region) override;
  void FillBuffer(Buffer* buffer, u64 offset, u64 bytes, u32 value) override;
  void Transition(Buffer* buffer, ResourceState state) override;
  void Transition(Texture* texture, ResourceState state) override;
  void PushDebugLabel(const char* label) override;
  void PopDebugLabel() override;

  VkCommandBuffer handle() const { return cmd_; }  // NOLINT: accessor

 private:
  // Shader writes made visible to the next shader access, the barrier a
  // transition into a state the resource already holds asks for.
  void ShaderWriteBarrier();

  VulkanDevice& device_;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VulkanPipeline* pipeline_ = nullptr;
};

class VulkanDevice : public Device {
 public:
  VulkanDevice() = default;
  ~VulkanDevice() override;

  bool Create();

  const char* backend_name() const override { return "vulkan"; }  // NOLINT
  const DeviceCaps& caps() const override { return caps_; }       // NOLINT

  std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override;
  std::unique_ptr<Texture> CreateTexture(const TextureDesc& desc) override;
  std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override;
  std::unique_ptr<ShaderModule> CreateShader(const ShaderDesc& desc) override;
  std::unique_ptr<BindGroupLayout> CreateBindGroupLayout(
      const BindGroupLayoutDesc& desc) override;
  std::unique_ptr<BindGroup> CreateBindGroup(
      const BindGroupDesc& desc) override;
  std::unique_ptr<Pipeline> CreateGraphicsPipeline(
      const GraphicsPipelineDesc& desc) override;
  std::unique_ptr<Pipeline> CreateComputePipeline(
      const ComputePipelineDesc& desc) override;
  std::unique_ptr<CommandList> CreateCommandList() override;

  u64 Submit(CommandList* list) override;
  void Wait(u64 submission) override;
  void WaitIdle() override;

  VkDevice device() const { return device_; }  // NOLINT: accessor
  // A descriptor set plus the pool it came from, so the group that owns it can
  // give it back.
  struct SetAllocation {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
  };
  SetAllocation AllocateSet(VkDescriptorSetLayout layout);
  u32 FindMemoryType(u32 type_bits, VkMemoryPropertyFlags props) const;
  // The empty layout that fills a gap in a pipeline's bind group list, since a
  // Vulkan pipeline layout has no holes.
  VkDescriptorSetLayout empty_set_layout() const {  // NOLINT: accessor
    return empty_set_layout_;
  }

 private:
  bool CreateInstance();
  bool PickPhysicalDevice();
  bool CreateLogicalDevice();
  bool CreatePools();

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  u32 queue_family_ = 0;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> descriptor_pools_;
  VkDescriptorSetLayout empty_set_layout_ = VK_NULL_HANDLE;
  VkSemaphore timeline_ = VK_NULL_HANDLE;
  u64 submissions_ = 0;
  VkPhysicalDeviceMemoryProperties memory_props_{};
  DeviceCaps caps_;
  std::string adapter_name_;
};

VulkanBuffer::VulkanBuffer(VulkanDevice& device, const BufferDesc& desc)
    : device_(device) {
  desc_ = desc;

  VkBufferUsageFlags usage = 0;
  if (desc.usage & kBufferVertex)
    usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (desc.usage & kBufferIndex)
    usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if (desc.usage & kBufferConstant)
    usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (desc.usage & kBufferStorage)
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (desc.usage & kBufferCopySrc)
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (desc.usage & kBufferCopyDst)
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (desc.usage & kBufferIndirect)
    usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = desc.size ? desc.size : 1;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  RHI_VK(vkCreateBuffer(device_.device(), &bi, nullptr, &buffer_), return);

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device_.device(), buffer_, &req);

  VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (desc.memory == MemoryKind::Upload) {
    props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  } else if (desc.memory == MemoryKind::Readback) {
    // Cached, not write-combined: the CPU reads this memory, and reading
    // write-combined memory costs tens of milliseconds a frame.
    props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = device_.FindMemoryType(req.memoryTypeBits, props);
  if (ai.memoryTypeIndex == ~0u && desc.memory == MemoryKind::Readback) {
    props &= ~VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    ai.memoryTypeIndex = device_.FindMemoryType(req.memoryTypeBits, props);
  }
  if (ai.memoryTypeIndex == ~0u) {
    BASE_LOGI("rhivk", "no memory type for buffer usage {:#x}", desc.usage);
    vkDestroyBuffer(device_.device(), buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    return;
  }
  RHI_VK(vkAllocateMemory(device_.device(), &ai, nullptr, &memory_), {
    vkDestroyBuffer(device_.device(), buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    return;
  });
  vkBindBufferMemory(device_.device(), buffer_, memory_, 0);

  if (desc.memory != MemoryKind::Device)
    vkMapMemory(device_.device(), memory_, 0, VK_WHOLE_SIZE, 0, &mapped_);
}

VulkanBuffer::~VulkanBuffer() {
  if (mapped_)
    vkUnmapMemory(device_.device(), memory_);
  if (buffer_)
    vkDestroyBuffer(device_.device(), buffer_, nullptr);
  if (memory_)
    vkFreeMemory(device_.device(), memory_, nullptr);
}

VulkanTexture::VulkanTexture(VulkanDevice& device, const TextureDesc& desc)
    : device_(device) {
  desc_ = desc;

  VkImageUsageFlags usage = 0;
  if (desc.usage & kTextureSampled)
    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (desc.usage & kTextureRenderTarget)
    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (desc.usage & kTextureDepthStencil)
    usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (desc.usage & kTextureStorage)
    usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (desc.usage & kTextureCopySrc)
    usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (desc.usage & kTextureCopyDst)
    usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = desc.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  ii.format = ToVk(desc.format);
  ii.extent = {desc.width, desc.height, desc.depth};
  ii.mipLevels = desc.mip_levels;
  ii.arrayLayers = desc.depth > 1 ? 1 : desc.array_layers;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = usage;
  ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  RHI_VK(vkCreateImage(device_.device(), &ii, nullptr, &image_), return);

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(device_.device(), image_, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = device_.FindMemoryType(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  RHI_VK(vkAllocateMemory(device_.device(), &ai, nullptr, &memory_), {
    vkDestroyImage(device_.device(), image_, nullptr);
    image_ = VK_NULL_HANDLE;
    return;
  });
  vkBindImageMemory(device_.device(), image_, memory_, 0);
}

VulkanTexture::~VulkanTexture() {
  for (auto& [key, view] : views_)
    vkDestroyImageView(device_.device(), view, nullptr);
  if (image_)
    vkDestroyImage(device_.device(), image_, nullptr);
  if (memory_)
    vkFreeMemory(device_.device(), memory_, nullptr);
}

VkImageView VulkanTexture::GetView(const ViewKey& key) {
  for (const auto& [k, view] : views_) {
    if (k == key)
      return view;
  }
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = image_;
  vi.viewType = key.type;
  vi.format = key.format;
  vi.subresourceRange = {key.aspect, key.base_mip, key.mips, key.base_layer,
                         key.layers};
  VkImageView view = VK_NULL_HANDLE;
  RHI_VK(vkCreateImageView(device_.device(), &vi, nullptr, &view),
         return VK_NULL_HANDLE);
  views_.emplace_back(key, view);
  return view;
}

VkImageView VulkanTexture::AttachmentView(u32 mip, u32 layer) {
  return GetView({ToVk(desc_.format), mip, 1, layer, 1,
                  VK_IMAGE_VIEW_TYPE_2D, AspectOf(desc_.format)});
}

VkImageView VulkanTexture::ShaderView(Format format,
                                      u32 base_mip,
                                      u32 mips,
                                      u32 base_layer,
                                      u32 layers) {
  const Format use = format == Format::Unknown ? desc_.format : format;
  VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;
  if (desc_.depth > 1)
    type = VK_IMAGE_VIEW_TYPE_3D;
  else if (layers > 1)
    type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  // A depth image is sampled through its depth plane alone: a view carrying
  // both aspects is not a legal shader resource.
  VkImageAspectFlags aspect = AspectOf(use);
  if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT)
    aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
  return GetView({ToVk(use), base_mip, mips, base_layer, layers, type,
                  aspect});
}

VulkanSampler::VulkanSampler(VulkanDevice& device, const SamplerDesc& desc)
    : device_(device) {
  VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  si.magFilter = ToVkFilter(desc.mag_filter);
  si.minFilter = ToVkFilter(desc.min_filter);
  si.mipmapMode = desc.mip_filter == FilterMode::Linear
                      ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                      : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  si.addressModeU = ToVk(desc.address_u);
  si.addressModeV = ToVk(desc.address_v);
  si.addressModeW = ToVk(desc.address_w);
  si.mipLodBias = desc.lod_bias;
  si.anisotropyEnable = desc.max_anisotropy > 1;
  si.maxAnisotropy = static_cast<float>(desc.max_anisotropy);
  si.compareEnable = desc.compare_enable;
  si.compareOp = ToVk(desc.compare);
  si.minLod = desc.min_lod;
  si.maxLod = desc.max_lod;
  switch (desc.border) {
    case BorderColor::TransparentBlack:
      si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      break;
    case BorderColor::OpaqueBlack:
      si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
      break;
    case BorderColor::OpaqueWhite:
      si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      break;
  }
  RHI_VK(vkCreateSampler(device_.device(), &si, nullptr, &sampler_), return);
}

VulkanSampler::~VulkanSampler() {
  if (sampler_)
    vkDestroySampler(device_.device(), sampler_, nullptr);
}

VulkanShaderModule::VulkanShaderModule(VulkanDevice& device,
                                       const ShaderDesc& desc)
    : device_(device), entry_(desc.entry ? desc.entry : "main") {
  if (!desc.spirv || !desc.spirv_bytes) {
    BASE_LOGI("rhivk", "shader '{}' has no SPIR-V",
              desc.name ? desc.name : "?");
    return;
  }
  VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  mi.codeSize = desc.spirv_bytes;
  mi.pCode = desc.spirv;
  RHI_VK(vkCreateShaderModule(device_.device(), &mi, nullptr, &module_),
         return);
}

VulkanShaderModule::~VulkanShaderModule() {
  if (module_)
    vkDestroyShaderModule(device_.device(), module_, nullptr);
}

VulkanBindGroupLayout::VulkanBindGroupLayout(VulkanDevice& device,
                                             const BindGroupLayoutDesc& desc)
    : device_(device) {
  desc_ = desc;
  VkDescriptorSetLayoutBinding bindings[kMaxBindGroupEntries];
  for (u32 i = 0; i < desc.num_entries; i++) {
    const BindGroupEntry& e = desc.entries[i];
    bindings[i] = {};
    bindings[i].binding = e.binding;
    bindings[i].descriptorType = ToVk(e.type, e.dynamic);
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = ToVkStages(e.stages);
  }
  VkDescriptorSetLayoutCreateInfo li{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  li.bindingCount = desc.num_entries;
  li.pBindings = bindings;
  RHI_VK(vkCreateDescriptorSetLayout(device_.device(), &li, nullptr, &layout_),
         return);
}

VulkanBindGroupLayout::~VulkanBindGroupLayout() {
  if (layout_)
    vkDestroyDescriptorSetLayout(device_.device(), layout_, nullptr);
}

VulkanBindGroup::VulkanBindGroup(VulkanDevice& device,
                                 const BindGroupDesc& desc)
    : device_(device) {
  const auto* layout = static_cast<const VulkanBindGroupLayout*>(desc.layout);
  if (!layout)
    return;
  const VulkanDevice::SetAllocation allocation =
      device.AllocateSet(layout->handle());
  pool_ = allocation.pool;
  set_ = allocation.set;
  if (!set_)
    return;

  VkWriteDescriptorSet writes[kMaxBindGroupEntries]{};
  VkDescriptorBufferInfo buffers[kMaxBindGroupEntries]{};
  VkDescriptorImageInfo images[kMaxBindGroupEntries]{};
  u32 count = 0;

  for (u32 i = 0; i < desc.num_bindings; i++) {
    const BindGroupBinding& b = desc.bindings[i];
    // The layout is what says how a binding is consumed; the binding itself
    // only supplies the resource.
    const BindGroupEntry* entry = nullptr;
    for (u32 e = 0; e < layout->desc().num_entries; e++) {
      if (layout->desc().entries[e].binding == b.binding)
        entry = &layout->desc().entries[e];
    }
    if (!entry)
      continue;

    VkWriteDescriptorSet& w = writes[count];
    w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set_;
    w.dstBinding = b.binding;
    w.descriptorCount = 1;
    w.descriptorType = ToVk(entry->type, entry->dynamic);

    switch (entry->type) {
      case BindingType::UniformBuffer:
      case BindingType::StorageBuffer: {
        auto* buffer = static_cast<VulkanBuffer*>(b.buffer);
        if (!buffer)
          continue;
        buffers[count].buffer = buffer->handle();
        buffers[count].offset = b.offset;
        buffers[count].range =
            b.size ? b.size : buffer->desc().size - b.offset;
        w.pBufferInfo = &buffers[count];
        break;
      }
      case BindingType::SampledTexture:
      case BindingType::StorageTexture: {
        auto* texture = static_cast<VulkanTexture*>(b.texture);
        if (!texture)
          continue;
        images[count].imageView = texture->ShaderView(
            b.view_format, b.base_mip, b.mips, b.base_layer, b.layers);
        images[count].imageLayout =
            entry->type == BindingType::StorageTexture
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        w.pImageInfo = &images[count];
        break;
      }
      case BindingType::Sampler: {
        auto* sampler = static_cast<VulkanSampler*>(b.sampler);
        if (!sampler)
          continue;
        images[count].sampler = sampler->handle();
        w.pImageInfo = &images[count];
        break;
      }
    }
    count++;
  }
  if (count)
    vkUpdateDescriptorSets(device.device(), count, writes, 0, nullptr);
}

VulkanBindGroup::~VulkanBindGroup() {
  if (set_)
    vkFreeDescriptorSets(device_.device(), pool_, 1, &set_);
}

bool VulkanPipeline::CreateLayout(const BindGroupLayout* const* groups,
                                  u32 num_groups,
                                  u32 push_bytes,
                                  VkShaderStageFlags push_stages) {
  VkDescriptorSetLayout set_layouts[kMaxBindGroups];
  for (u32 i = 0; i < num_groups; i++) {
    const auto* group = static_cast<const VulkanBindGroupLayout*>(groups[i]);
    set_layouts[i] =
        group ? group->handle() : device_.empty_set_layout();
  }

  VkPushConstantRange push{};
  push.stageFlags = push_stages;
  push.offset = 0;
  push.size = push_bytes;

  VkPipelineLayoutCreateInfo pi{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pi.setLayoutCount = num_groups;
  pi.pSetLayouts = set_layouts;
  pi.pushConstantRangeCount = push_bytes ? 1 : 0;
  pi.pPushConstantRanges = &push;
  push_stages_ = push_stages;
  RHI_VK(vkCreatePipelineLayout(device_.device(), &pi, nullptr, &layout_),
         return false);
  return true;
}

VulkanPipeline::VulkanPipeline(VulkanDevice& device,
                               const GraphicsPipelineDesc& desc)
    : device_(device) {
  const auto* vs = static_cast<const VulkanShaderModule*>(desc.vertex);
  const auto* ps = static_cast<const VulkanShaderModule*>(desc.pixel);
  if (!vs || !vs->handle()) {
    BASE_LOGI("rhivk", "pipeline '{}' has no vertex shader",
              desc.name ? desc.name : "?");
    return;
  }
  if (!CreateLayout(desc.bind_groups, desc.num_bind_groups,
                    desc.push_constant_bytes,
                    VK_SHADER_STAGE_VERTEX_BIT |
                        VK_SHADER_STAGE_FRAGMENT_BIT)) {
    return;
  }

  VkPipelineShaderStageCreateInfo stages[2]{};
  u32 num_stages = 0;
  stages[num_stages] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[num_stages].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[num_stages].module = vs->handle();
  stages[num_stages].pName = vs->entry();
  num_stages++;
  if (ps && ps->handle()) {
    stages[num_stages] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[num_stages].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[num_stages].module = ps->handle();
    stages[num_stages].pName = ps->entry();
    num_stages++;
  }

  VkVertexInputBindingDescription vbs[kMaxVertexBuffers]{};
  for (u32 i = 0; i < desc.num_vertex_buffers; i++) {
    vbs[i].binding = i;
    vbs[i].stride = desc.vertex_buffers[i].stride;
    vbs[i].inputRate = desc.vertex_buffers[i].per_instance
                           ? VK_VERTEX_INPUT_RATE_INSTANCE
                           : VK_VERTEX_INPUT_RATE_VERTEX;
  }
  VkVertexInputAttributeDescription vas[kMaxVertexAttributes]{};
  for (u32 i = 0; i < desc.num_attributes; i++) {
    vas[i].location = desc.attributes[i].location;
    vas[i].binding = desc.attributes[i].buffer;
    vas[i].format = ToVk(desc.attributes[i].format);
    vas[i].offset = desc.attributes[i].offset;
  }
  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = desc.num_vertex_buffers;
  vi.pVertexBindingDescriptions = vbs;
  vi.vertexAttributeDescriptionCount = desc.num_attributes;
  vi.pVertexAttributeDescriptions = vas;

  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = ToVk(desc.topology);

  VkPipelineViewportStateCreateInfo vp{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.depthClampEnable = !desc.raster.depth_clip;
  switch (desc.raster.cull) {
    case CullMode::None: rs.cullMode = VK_CULL_MODE_NONE; break;
    case CullMode::Front: rs.cullMode = VK_CULL_MODE_FRONT_BIT; break;
    case CullMode::Back: rs.cullMode = VK_CULL_MODE_BACK_BIT; break;
  }
  rs.frontFace = desc.raster.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                       : VK_FRONT_FACE_CLOCKWISE;
  rs.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo ds{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = desc.depth.test_enable;
  ds.depthWriteEnable = desc.depth.write_enable;
  ds.depthCompareOp = ToVk(desc.depth.compare);
  ds.maxDepthBounds = 1.0f;

  VkPipelineColorBlendAttachmentState blends[kMaxColorTargets]{};
  for (u32 i = 0; i < desc.num_color_targets; i++) {
    const BlendState& b = desc.blend[i];
    blends[i].blendEnable = b.enable;
    blends[i].srcColorBlendFactor = ToVk(b.src_color);
    blends[i].dstColorBlendFactor = ToVk(b.dst_color);
    blends[i].colorBlendOp = ToVk(b.color_op);
    blends[i].srcAlphaBlendFactor = ToVk(b.src_alpha);
    blends[i].dstAlphaBlendFactor = ToVk(b.dst_alpha);
    blends[i].alphaBlendOp = ToVk(b.alpha_op);
    blends[i].colorWriteMask = b.write_mask;
  }
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = desc.num_color_targets;
  cb.pAttachments = blends;

  const VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                     VK_DYNAMIC_STATE_SCISSOR,
                                     VK_DYNAMIC_STATE_BLEND_CONSTANTS};
  VkPipelineDynamicStateCreateInfo dy{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = sizeof(dynamics) / sizeof(dynamics[0]);
  dy.pDynamicStates = dynamics;

  VkFormat color_formats[kMaxColorTargets]{};
  for (u32 i = 0; i < desc.num_color_targets; i++)
    color_formats[i] = ToVk(desc.color_formats[i]);
  VkPipelineRenderingCreateInfo ri{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  ri.colorAttachmentCount = desc.num_color_targets;
  ri.pColorAttachmentFormats = color_formats;
  ri.depthAttachmentFormat = ToVk(desc.depth_format);
  if (FormatHasStencil(desc.depth_format))
    ri.stencilAttachmentFormat = ri.depthAttachmentFormat;

  VkGraphicsPipelineCreateInfo gi{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &ri;
  gi.stageCount = num_stages;
  gi.pStages = stages;
  gi.pVertexInputState = &vi;
  gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp;
  gi.pRasterizationState = &rs;
  gi.pMultisampleState = &ms;
  gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cb;
  gi.pDynamicState = &dy;
  gi.layout = layout_;
  RHI_VK(vkCreateGraphicsPipelines(device_.device(), VK_NULL_HANDLE, 1, &gi,
                                   nullptr, &pipeline_),
         return);
}

VulkanPipeline::VulkanPipeline(VulkanDevice& device,
                               const ComputePipelineDesc& desc)
    : device_(device), bind_point_(VK_PIPELINE_BIND_POINT_COMPUTE) {
  const auto* cs = static_cast<const VulkanShaderModule*>(desc.compute);
  if (!cs || !cs->handle())
    return;
  if (!CreateLayout(desc.bind_groups, desc.num_bind_groups,
                    desc.push_constant_bytes, VK_SHADER_STAGE_COMPUTE_BIT)) {
    return;
  }
  VkComputePipelineCreateInfo ci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = cs->handle();
  ci.stage.pName = cs->entry();
  ci.layout = layout_;
  RHI_VK(vkCreateComputePipelines(device_.device(), VK_NULL_HANDLE, 1, &ci,
                                  nullptr, &pipeline_),
         return);
}

VulkanPipeline::~VulkanPipeline() {
  if (pipeline_)
    vkDestroyPipeline(device_.device(), pipeline_, nullptr);
  if (layout_)
    vkDestroyPipelineLayout(device_.device(), layout_, nullptr);
}

VulkanCommandList::VulkanCommandList(VulkanDevice& device, VkCommandBuffer cmd)
    : device_(device), cmd_(cmd) {}

VulkanCommandList::~VulkanCommandList() = default;

void VulkanCommandList::Begin() {
  vkResetCommandBuffer(cmd_, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd_, &bi);
  pipeline_ = nullptr;
}

void VulkanCommandList::End() {
  vkEndCommandBuffer(cmd_);
}

void VulkanCommandList::BeginRenderPass(const RenderPassDesc& pass) {
  VkRenderingAttachmentInfo colors[kMaxColorTargets]{};
  for (u32 i = 0; i < pass.num_color; i++) {
    const ColorAttachment& a = pass.color[i];
    auto* texture = static_cast<VulkanTexture*>(a.texture);
    Transition(texture, ResourceState::RenderTarget);
    colors[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colors[i].imageView = texture->AttachmentView(a.mip, a.layer);
    colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colors[i].loadOp = ToVk(a.load);
    colors[i].storeOp = ToVk(a.store);
    memcpy(colors[i].clearValue.color.float32, a.clear, sizeof(a.clear));
  }

  VkRenderingAttachmentInfo depth{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  VkRenderingAttachmentInfo stencil = depth;
  bool has_stencil = false;
  if (pass.has_depth && pass.depth.texture) {
    auto* texture = static_cast<VulkanTexture*>(pass.depth.texture);
    Transition(texture, pass.depth.read_only ? ResourceState::DepthRead
                                             : ResourceState::DepthWrite);
    depth.imageView = texture->AttachmentView(0, 0);
    depth.imageLayout =
        pass.depth.read_only
            ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.loadOp = ToVk(pass.depth.load);
    depth.storeOp = ToVk(pass.depth.store);
    depth.clearValue.depthStencil.depth = pass.depth.clear_depth;
    if (FormatHasStencil(texture->desc().format)) {
      stencil = depth;
      stencil.clearValue.depthStencil.stencil = pass.depth.clear_stencil;
      has_stencil = true;
    }
  }

  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {pass.width, pass.height}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = pass.num_color;
  ri.pColorAttachments = colors;
  ri.pDepthAttachment = depth.imageView ? &depth : nullptr;
  ri.pStencilAttachment = has_stencil ? &stencil : nullptr;
  vkCmdBeginRendering(cmd_, &ri);
}

void VulkanCommandList::EndRenderPass() {
  vkCmdEndRendering(cmd_);
}

void VulkanCommandList::SetPipeline(Pipeline* pipeline) {
  pipeline_ = static_cast<VulkanPipeline*>(pipeline);
  if (!pipeline_ || !pipeline_->valid())
    return;
  vkCmdBindPipeline(cmd_, pipeline_->bind_point(), pipeline_->handle());
}

void VulkanCommandList::SetBindGroup(u32 index,
                                     BindGroup* group,
                                     const u32* offsets,
                                     u32 num_offsets) {
  if (!pipeline_ || !group)
    return;
  VkDescriptorSet set = static_cast<VulkanBindGroup*>(group)->handle();
  if (!set)
    return;
  vkCmdBindDescriptorSets(cmd_, pipeline_->bind_point(), pipeline_->layout(),
                          index, 1, &set, num_offsets, offsets);
}

void VulkanCommandList::SetPushConstants(u32 offset,
                                         u32 bytes,
                                         const void* data) {
  if (!pipeline_ || !bytes)
    return;
  vkCmdPushConstants(cmd_, pipeline_->layout(), pipeline_->push_stages(),
                     offset, bytes, data);
}

void VulkanCommandList::SetVertexBuffer(u32 slot,
                                        Buffer* buffer,
                                        u64 offset) {
  if (!buffer)
    return;
  VkBuffer handle = static_cast<VulkanBuffer*>(buffer)->handle();
  VkDeviceSize vk_offset = offset;
  vkCmdBindVertexBuffers(cmd_, slot, 1, &handle, &vk_offset);
}

void VulkanCommandList::SetIndexBuffer(Buffer* buffer,
                                       u64 offset,
                                       IndexType type) {
  if (!buffer)
    return;
  vkCmdBindIndexBuffer(cmd_, static_cast<VulkanBuffer*>(buffer)->handle(),
                       offset,
                       type == IndexType::U16 ? VK_INDEX_TYPE_UINT16
                                              : VK_INDEX_TYPE_UINT32);
}

void VulkanCommandList::SetViewport(float x,
                                    float y,
                                    float width,
                                    float height,
                                    float min_depth,
                                    float max_depth) {
  // The seam's clip space has +Y up, which is D3D12's; Vulkan's has +Y down.
  // A viewport of negative height, anchored at the rectangle's bottom edge,
  // is the flip -- and it is the same flip the guest renderer already applies
  // so that a target sampled as a texture lines up with the one it was
  // rasterized into.
  VkViewport vp{x, y + height, width, -height, min_depth, max_depth};
  vkCmdSetViewport(cmd_, 0, 1, &vp);
}

void VulkanCommandList::SetScissor(i32 x, i32 y, u32 width, u32 height) {
  VkRect2D rect{{x, y}, {width, height}};
  vkCmdSetScissor(cmd_, 0, 1, &rect);
}

void VulkanCommandList::SetBlendConstants(const float rgba[4]) {
  vkCmdSetBlendConstants(cmd_, rgba);
}

void VulkanCommandList::Draw(u32 vertex_count,
                             u32 instance_count,
                             u32 first_vertex,
                             u32 first_instance) {
  vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanCommandList::DrawIndexed(u32 index_count,
                                    u32 instance_count,
                                    u32 first_index,
                                    i32 vertex_offset,
                                    u32 first_instance) {
  vkCmdDrawIndexed(cmd_, index_count, instance_count, first_index,
                   vertex_offset, first_instance);
}

void VulkanCommandList::Dispatch(u32 groups_x, u32 groups_y, u32 groups_z) {
  vkCmdDispatch(cmd_, groups_x, groups_y, groups_z);
}

void VulkanCommandList::CopyBuffer(Buffer* dst,
                                   u64 dst_offset,
                                   Buffer* src,
                                   u64 src_offset,
                                   u64 bytes) {
  VkBufferCopy copy{src_offset, dst_offset, bytes};
  vkCmdCopyBuffer(cmd_, static_cast<VulkanBuffer*>(src)->handle(),
                  static_cast<VulkanBuffer*>(dst)->handle(), 1, &copy);
}

void VulkanCommandList::CopyBufferToTexture(Texture* dst,
                                            const TextureRegion& region,
                                            Buffer* src,
                                            u64 src_offset,
                                            u32 src_row_pitch) {
  auto* texture = static_cast<VulkanTexture*>(dst);
  const Format format = texture->desc().format;
  const u32 block = FormatBlockDim(format);
  VkBufferImageCopy copy{};
  copy.bufferOffset = src_offset;
  copy.bufferRowLength =
      src_row_pitch
          ? static_cast<u32>(src_row_pitch / FormatTexelBytes(format) * block)
          : 0;
  copy.imageSubresource = {AspectOf(format), region.mip, region.layer, 1};
  copy.imageOffset = {static_cast<i32>(region.x), static_cast<i32>(region.y),
                      static_cast<i32>(region.z)};
  copy.imageExtent = {region.width, region.height, region.depth};
  vkCmdCopyBufferToImage(cmd_, static_cast<VulkanBuffer*>(src)->handle(),
                         texture->image(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
}

void VulkanCommandList::CopyTextureToBuffer(Buffer* dst,
                                            u64 dst_offset,
                                            u32 dst_row_pitch,
                                            Texture* src,
                                            const TextureRegion& region) {
  auto* texture = static_cast<VulkanTexture*>(src);
  const Format format = texture->desc().format;
  const u32 block = FormatBlockDim(format);
  VkImageAspectFlags aspect = AspectOf(format);
  if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT)
    aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
  VkBufferImageCopy copy{};
  copy.bufferOffset = dst_offset;
  copy.bufferRowLength =
      dst_row_pitch
          ? static_cast<u32>(dst_row_pitch / FormatTexelBytes(format) * block)
          : 0;
  copy.imageSubresource = {aspect, region.mip, region.layer, 1};
  copy.imageOffset = {static_cast<i32>(region.x), static_cast<i32>(region.y),
                      static_cast<i32>(region.z)};
  copy.imageExtent = {region.width, region.height, region.depth};
  vkCmdCopyImageToBuffer(cmd_, texture->image(),
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         static_cast<VulkanBuffer*>(dst)->handle(), 1, &copy);
}

void VulkanCommandList::FillBuffer(Buffer* buffer,
                                   u64 offset,
                                   u64 bytes,
                                   u32 value) {
  vkCmdFillBuffer(cmd_, static_cast<VulkanBuffer*>(buffer)->handle(), offset,
                  bytes, value);
}

void VulkanCommandList::Transition(Buffer* buffer, ResourceState state) {
  if (!buffer)
    return;
  if (buffer->state() == state) {
    // Asking for a state a resource already holds only means something for
    // unordered access, where it is a request to make earlier writes visible.
    if (state == ResourceState::UnorderedAccess)
      ShaderWriteBarrier();
    return;
  }
  const StateInfo from = StateOf(buffer->state());
  const StateInfo to = StateOf(state);
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcAccessMask = from.access;
  barrier.dstAccessMask = to.access;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = static_cast<VulkanBuffer*>(buffer)->handle();
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd_, from.stage, to.stage, 0, 0, nullptr, 1, &barrier,
                       0, nullptr);
  buffer->set_state(state);
}

void VulkanCommandList::Transition(Texture* texture, ResourceState state) {
  if (!texture)
    return;
  if (texture->state() == state) {
    if (state == ResourceState::UnorderedAccess)
      ShaderWriteBarrier();
    return;
  }
  const StateInfo from = StateOf(texture->state());
  const StateInfo to = StateOf(state);
  const TextureDesc& desc = texture->desc();
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = from.access;
  barrier.dstAccessMask = to.access;
  barrier.oldLayout = from.layout;
  barrier.newLayout = to.layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = static_cast<VulkanTexture*>(texture)->image();
  barrier.subresourceRange = {AspectOf(desc.format), 0, desc.mip_levels, 0,
                              desc.depth > 1 ? 1 : desc.array_layers};
  vkCmdPipelineBarrier(cmd_, from.stage, to.stage, 0, 0, nullptr, 0, nullptr,
                       1, &barrier);
  texture->set_state(state);
}

void VulkanCommandList::ShaderWriteBarrier() {
  constexpr VkPipelineStageFlags kShaders =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd_, kShaders, kShaders, 0, 1, &barrier, 0, nullptr, 0,
                       nullptr);
}

void VulkanCommandList::PushDebugLabel(const char*) {}
void VulkanCommandList::PopDebugLabel() {}

VulkanDevice::~VulkanDevice() {
  if (!device_)
    return;
  vkDeviceWaitIdle(device_);
  for (VkDescriptorPool pool : descriptor_pools_)
    vkDestroyDescriptorPool(device_, pool, nullptr);
  if (empty_set_layout_)
    vkDestroyDescriptorSetLayout(device_, empty_set_layout_, nullptr);
  if (timeline_)
    vkDestroySemaphore(device_, timeline_, nullptr);
  if (command_pool_)
    vkDestroyCommandPool(device_, command_pool_, nullptr);
  vkDestroyDevice(device_, nullptr);
  if (instance_)
    vkDestroyInstance(instance_, nullptr);
}

bool VulkanDevice::CreateInstance() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "prosperity";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  RHI_VK(vkCreateInstance(&ic, nullptr, &instance_), return false);
  return true;
}

bool VulkanDevice::PickPhysicalDevice() {
  u32 count = 0;
  vkEnumeratePhysicalDevices(instance_, &count, nullptr);
  if (!count)
    return false;
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance_, &count, devices.data());

  // Prefer a discrete GPU, then any device that reports 1.3.
  int best_score = -1;
  for (VkPhysicalDevice candidate : devices) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(candidate, &props);
    if (props.apiVersion < VK_API_VERSION_1_3)
      continue;
    int score = 1;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score = 3;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score = 2;
    if (score > best_score) {
      best_score = score;
      physical_ = candidate;
      adapter_name_ = props.deviceName;
      caps_.max_texture_2d = props.limits.maxImageDimension2D;
      caps_.max_bound_textures =
          props.limits.maxPerStageDescriptorSampledImages;
      caps_.constant_buffer_offset_alignment = static_cast<u32>(
          props.limits.minUniformBufferOffsetAlignment);
      caps_.storage_buffer_offset_alignment = static_cast<u32>(
          props.limits.minStorageBufferOffsetAlignment);
      caps_.timestamp_period_ns = props.limits.timestampPeriod;
    }
  }
  if (!physical_)
    return false;

  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical_, &features);
  caps_.independent_blend = features.independentBlend;
  caps_.anisotropic_filtering = features.samplerAnisotropy;
  caps_.adapter_name = adapter_name_.c_str();
  vkGetPhysicalDeviceMemoryProperties(physical_, &memory_props_);

  u32 families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
  std::vector<VkQueueFamilyProperties> props(families);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, props.data());
  for (u32 i = 0; i < families; i++) {
    if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      queue_family_ = i;
      return true;
    }
  }
  return false;
}

bool VulkanDevice::CreateLogicalDevice() {
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qi.queueFamilyIndex = queue_family_;
  qi.queueCount = 1;
  qi.pQueuePriorities = &priority;

  VkPhysicalDeviceVulkan13Features f13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.dynamicRendering = VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.timelineSemaphore = VK_TRUE;
  f12.pNext = &f13;
  VkPhysicalDeviceFeatures2 f2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &f12;
  f2.features.samplerAnisotropy = caps_.anisotropic_filtering;
  f2.features.independentBlend = caps_.independent_blend;

  VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  di.pNext = &f2;
  di.queueCreateInfoCount = 1;
  di.pQueueCreateInfos = &qi;
  RHI_VK(vkCreateDevice(physical_, &di, nullptr, &device_), return false);
  vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

  VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  ti.initialValue = 0;
  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  si.pNext = &ti;
  RHI_VK(vkCreateSemaphore(device_, &si, nullptr, &timeline_), return false);
  return true;
}

bool VulkanDevice::CreatePools() {
  VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  ci.queueFamilyIndex = queue_family_;
  RHI_VK(vkCreateCommandPool(device_, &ci, nullptr, &command_pool_),
         return false);

  VkDescriptorSetLayoutCreateInfo li{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  RHI_VK(vkCreateDescriptorSetLayout(device_, &li, nullptr,
                                     &empty_set_layout_),
         return false);
  return true;
}

bool VulkanDevice::Create() {
  if (!CreateInstance())
    return false;
  if (!PickPhysicalDevice()) {
    BASE_LOGI("rhivk", "no Vulkan 1.3 device");
    return false;
  }
  if (!CreateLogicalDevice())
    return false;
  if (!CreatePools())
    return false;
  BASE_LOGI("rhivk", "device up: {}", adapter_name_.c_str());
  return true;
}

u32 VulkanDevice::FindMemoryType(u32 type_bits,
                                 VkMemoryPropertyFlags props) const {
  for (u32 i = 0; i < memory_props_.memoryTypeCount; i++) {
    if (!(type_bits & (1u << i)))
      continue;
    if ((memory_props_.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  return ~0u;
}

VulkanDevice::SetAllocation VulkanDevice::AllocateSet(
    VkDescriptorSetLayout layout) {
  for (int attempt = 0; attempt < 2; attempt++) {
    if (!descriptor_pools_.empty()) {
      VkDescriptorSetAllocateInfo ai{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      ai.descriptorPool = descriptor_pools_.back();
      ai.descriptorSetCount = 1;
      ai.pSetLayouts = &layout;
      VkDescriptorSet set = VK_NULL_HANDLE;
      if (vkAllocateDescriptorSets(device_, &ai, &set) == VK_SUCCESS)
        return {descriptor_pools_.back(), set};
    }
    // Out of room (or no pool yet): another pool, then try once more.
    constexpr u32 kPerPool = 1024;
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kPerPool},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kPerPool},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kPerPool},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, kPerPool},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kPerPool},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kPerPool},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kPerPool},
    };
    VkDescriptorPoolCreateInfo pi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = kPerPool;
    pi.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    pi.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    RHI_VK(vkCreateDescriptorPool(device_, &pi, nullptr, &pool), return {});
    descriptor_pools_.push_back(pool);
  }
  return {};
}

std::unique_ptr<Buffer> VulkanDevice::CreateBuffer(const BufferDesc& desc) {
  auto buffer = std::make_unique<VulkanBuffer>(*this, desc);
  if (!buffer->valid())
    return nullptr;
  return buffer;
}

std::unique_ptr<Texture> VulkanDevice::CreateTexture(
    const TextureDesc& desc) {
  auto texture = std::make_unique<VulkanTexture>(*this, desc);
  if (!texture->valid())
    return nullptr;
  return texture;
}

std::unique_ptr<Sampler> VulkanDevice::CreateSampler(
    const SamplerDesc& desc) {
  return std::make_unique<VulkanSampler>(*this, desc);
}

std::unique_ptr<ShaderModule> VulkanDevice::CreateShader(
    const ShaderDesc& desc) {
  auto shader = std::make_unique<VulkanShaderModule>(*this, desc);
  if (!shader->handle())
    return nullptr;
  return shader;
}

std::unique_ptr<BindGroupLayout> VulkanDevice::CreateBindGroupLayout(
    const BindGroupLayoutDesc& desc) {
  auto layout = std::make_unique<VulkanBindGroupLayout>(*this, desc);
  if (!layout->handle())
    return nullptr;
  return layout;
}

std::unique_ptr<BindGroup> VulkanDevice::CreateBindGroup(
    const BindGroupDesc& desc) {
  auto group = std::make_unique<VulkanBindGroup>(*this, desc);
  if (!group->handle())
    return nullptr;
  return group;
}

std::unique_ptr<Pipeline> VulkanDevice::CreateGraphicsPipeline(
    const GraphicsPipelineDesc& desc) {
  auto pipeline = std::make_unique<VulkanPipeline>(*this, desc);
  if (!pipeline->valid())
    return nullptr;
  return pipeline;
}

std::unique_ptr<Pipeline> VulkanDevice::CreateComputePipeline(
    const ComputePipelineDesc& desc) {
  auto pipeline = std::make_unique<VulkanPipeline>(*this, desc);
  if (!pipeline->valid())
    return nullptr;
  return pipeline;
}

std::unique_ptr<CommandList> VulkanDevice::CreateCommandList() {
  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = command_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  RHI_VK(vkAllocateCommandBuffers(device_, &ai, &cmd), return nullptr);
  return std::make_unique<VulkanCommandList>(*this, cmd);
}

u64 VulkanDevice::Submit(CommandList* list) {
  auto* vk_list = static_cast<VulkanCommandList*>(list);
  const uint64_t value = ++submissions_;
  VkCommandBuffer cmd = vk_list->handle();

  VkTimelineSemaphoreSubmitInfo ts{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  ts.signalSemaphoreValueCount = 1;
  ts.pSignalSemaphoreValues = &value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &ts;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &timeline_;
  RHI_VK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), return 0);
  return value;
}

void VulkanDevice::Wait(u64 submission) {
  if (!submission)
    return;
  const uint64_t value = submission;
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1;
  wi.pSemaphores = &timeline_;
  wi.pValues = &value;
  vkWaitSemaphores(device_, &wi, UINT64_MAX);
}

void VulkanDevice::WaitIdle() {
  vkDeviceWaitIdle(device_);
}

}  // namespace

std::unique_ptr<rhi::Device> CreateVulkanDevice() {
  auto device = std::make_unique<VulkanDevice>();
  if (!device->Create())
    return nullptr;
  return device;
}

}  // namespace gpu::vk
