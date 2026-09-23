/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/d3d12/d3d12_rhi.h"

#if !DELTA_HAVE_D3D12

namespace gpu::d3d12 {

std::unique_ptr<rhi::Device> CreateD3D12Device() {
  return nullptr;
}

}  // namespace gpu::d3d12

#else

// Everything that is not D3D12 comes first: the Windows compatibility headers
// define min and max as macros, which the standard library does not survive.
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include <base/logging.h>

#if defined(_WIN32)
#include <d3d12.h>
#include <d3dcompiler.h>
#else
// vkd3d ships the same D3D12 surface for Linux: vkd3d_windows.h supplies the
// COM and calling-convention shims (and must come first, or the generated
// header reaches for <windows.h>), vkd3d_utils.h the loader entry points
// d3d12.dll and d3dcompiler.dll export on Windows. INITGUID turns the IID
// declarations in the generated header into definitions; exactly one
// translation unit may do that, and this is the only one that includes them.
// WIDL_EXPLICIT_AGGREGATE_RETURNS: a method returning a struct by value is the
// C++ spelling, but the vtable behind it takes a hidden out-pointer. Without
// this the two disagree and the first such call (a descriptor heap's handle)
// jumps into nothing. The header still offers the by-value form as a wrapper,
// so the calls below read the same on either platform.
#define INITGUID
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <vkd3d_windows.h>
#include <vkd3d_d3d12.h>
#include <vkd3d_utils.h>
#undef min
#undef max
#endif

namespace gpu::d3d12 {
namespace {

using namespace gpu::rhi;  // NOLINT: this file is one backend of that seam

template <typename T>
void ReleaseCom(T*& object) {
  if (object) {
    object->Release();
    object = nullptr;
  }
}

DXGI_FORMAT ToDxgi(Format f) {
  switch (f) {
    case Format::R8Unorm: return DXGI_FORMAT_R8_UNORM;
    case Format::R8Uint: return DXGI_FORMAT_R8_UINT;
    case Format::RG8Unorm: return DXGI_FORMAT_R8G8_UNORM;
    case Format::RGBA8Unorm: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA8Srgb: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case Format::BGRA8Unorm: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case Format::BGRA8Srgb: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case Format::R16Uint: return DXGI_FORMAT_R16_UINT;
    case Format::R16Float: return DXGI_FORMAT_R16_FLOAT;
    case Format::RG16Float: return DXGI_FORMAT_R16G16_FLOAT;
    case Format::RGBA16Float: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case Format::RGBA16Unorm: return DXGI_FORMAT_R16G16B16A16_UNORM;
    case Format::R32Uint: return DXGI_FORMAT_R32_UINT;
    case Format::R32Float: return DXGI_FORMAT_R32_FLOAT;
    case Format::RG32Float: return DXGI_FORMAT_R32G32_FLOAT;
    case Format::RGB32Float: return DXGI_FORMAT_R32G32B32_FLOAT;
    case Format::RGBA32Float: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case Format::RGB10A2Unorm: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case Format::RG11B10Float: return DXGI_FORMAT_R11G11B10_FLOAT;
    case Format::D16Unorm: return DXGI_FORMAT_D16_UNORM;
    case Format::D32Float: return DXGI_FORMAT_D32_FLOAT;
    case Format::D32FloatS8Uint: return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case Format::BC1Unorm: return DXGI_FORMAT_BC1_UNORM;
    case Format::BC2Unorm: return DXGI_FORMAT_BC2_UNORM;
    case Format::BC3Unorm: return DXGI_FORMAT_BC3_UNORM;
    case Format::BC4Unorm: return DXGI_FORMAT_BC4_UNORM;
    case Format::BC5Unorm: return DXGI_FORMAT_BC5_UNORM;
    case Format::BC7Unorm: return DXGI_FORMAT_BC7_UNORM;
    case Format::Unknown:
    case Format::kCount: break;
  }
  return DXGI_FORMAT_UNKNOWN;
}

// A depth texture that is also sampled cannot be created in its depth format:
// the resource is typeless and the depth and shader views reinterpret it. This
// is the one place where the D3D12 resource format is not ToDxgi().
struct DepthFormats {
  DXGI_FORMAT resource;
  DXGI_FORMAT view;   // DSV
  DXGI_FORMAT shader; // SRV
};

DepthFormats DepthFormatsFor(Format f) {
  switch (f) {
    case Format::D16Unorm:
      return {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM,
              DXGI_FORMAT_R16_UNORM};
    case Format::D32Float:
      return {DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT,
              DXGI_FORMAT_R32_FLOAT};
    case Format::D32FloatS8Uint:
      return {DXGI_FORMAT_R32G8X24_TYPELESS,
              DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
              DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS};
    default: break;
  }
  const DXGI_FORMAT plain = ToDxgi(f);
  return {plain, plain, plain};
}

D3D12_COMPARISON_FUNC ToD3D(CompareOp op) {
  switch (op) {
    case CompareOp::Never: return D3D12_COMPARISON_FUNC_NEVER;
    case CompareOp::Less: return D3D12_COMPARISON_FUNC_LESS;
    case CompareOp::Equal: return D3D12_COMPARISON_FUNC_EQUAL;
    case CompareOp::LessEqual: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    case CompareOp::Greater: return D3D12_COMPARISON_FUNC_GREATER;
    case CompareOp::NotEqual: return D3D12_COMPARISON_FUNC_NOT_EQUAL;
    case CompareOp::GreaterEqual: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    case CompareOp::Always: return D3D12_COMPARISON_FUNC_ALWAYS;
  }
  return D3D12_COMPARISON_FUNC_ALWAYS;
}

// D3D12 has one blend factor enum for colour and one interpretation for alpha:
// the *_COLOR factors are illegal in the alpha slots, so alpha uses the
// matching *_ALPHA factor. `alpha` selects that translation.
D3D12_BLEND ToD3D(BlendFactor f, bool alpha) {
  switch (f) {
    case BlendFactor::Zero: return D3D12_BLEND_ZERO;
    case BlendFactor::One: return D3D12_BLEND_ONE;
    case BlendFactor::SrcColor:
      return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
    case BlendFactor::OneMinusSrcColor:
      return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
    case BlendFactor::DstColor:
      return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
    case BlendFactor::OneMinusDstColor:
      return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
    case BlendFactor::SrcAlpha: return D3D12_BLEND_SRC_ALPHA;
    case BlendFactor::OneMinusSrcAlpha: return D3D12_BLEND_INV_SRC_ALPHA;
    case BlendFactor::DstAlpha: return D3D12_BLEND_DEST_ALPHA;
    case BlendFactor::OneMinusDstAlpha: return D3D12_BLEND_INV_DEST_ALPHA;
    case BlendFactor::ConstantColor:
    case BlendFactor::ConstantAlpha:
      return D3D12_BLEND_BLEND_FACTOR;
    case BlendFactor::OneMinusConstantColor:
    case BlendFactor::OneMinusConstantAlpha:
      return D3D12_BLEND_INV_BLEND_FACTOR;
    case BlendFactor::SrcAlphaSaturate: return D3D12_BLEND_SRC_ALPHA_SAT;
  }
  return D3D12_BLEND_ONE;
}

D3D12_BLEND_OP ToD3D(BlendOp op) {
  switch (op) {
    case BlendOp::Add: return D3D12_BLEND_OP_ADD;
    case BlendOp::Subtract: return D3D12_BLEND_OP_SUBTRACT;
    case BlendOp::ReverseSubtract: return D3D12_BLEND_OP_REV_SUBTRACT;
    case BlendOp::Min: return D3D12_BLEND_OP_MIN;
    case BlendOp::Max: return D3D12_BLEND_OP_MAX;
  }
  return D3D12_BLEND_OP_ADD;
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE ToD3DType(PrimitiveTopology t) {
  switch (t) {
    case PrimitiveTopology::PointList:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    case PrimitiveTopology::LineList:
    case PrimitiveTopology::LineStrip:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    case PrimitiveTopology::TriangleList:
    case PrimitiveTopology::TriangleStrip:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  }
  return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

D3D12_PRIMITIVE_TOPOLOGY ToD3DTopology(PrimitiveTopology t) {
  switch (t) {
    case PrimitiveTopology::PointList:
      return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case PrimitiveTopology::LineList:
      return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case PrimitiveTopology::LineStrip:
      return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case PrimitiveTopology::TriangleList:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    case PrimitiveTopology::TriangleStrip:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
  }
  return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
}

D3D12_TEXTURE_ADDRESS_MODE ToD3D(AddressMode m) {
  switch (m) {
    case AddressMode::Repeat: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case AddressMode::MirrorRepeat: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case AddressMode::ClampToEdge: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case AddressMode::ClampToBorder: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
  }
  return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
}

D3D12_RESOURCE_STATES ToD3D(ResourceState s) {
  switch (s) {
    case ResourceState::Undefined:
    case ResourceState::Common:
      return D3D12_RESOURCE_STATE_COMMON;
    case ResourceState::RenderTarget:
      return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case ResourceState::DepthWrite: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case ResourceState::DepthRead: return D3D12_RESOURCE_STATE_DEPTH_READ;
    case ResourceState::ShaderRead:
      return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case ResourceState::UnorderedAccess:
      return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case ResourceState::CopySrc: return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case ResourceState::CopyDst: return D3D12_RESOURCE_STATE_COPY_DEST;
    case ResourceState::VertexInput:
      return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    case ResourceState::IndexInput:
      return D3D12_RESOURCE_STATE_INDEX_BUFFER;
    case ResourceState::IndirectArgument:
      return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
  }
  return D3D12_RESOURCE_STATE_COMMON;
}

// Which kind of descriptor a binding becomes, which is also its HLSL register
// class: b for a constant buffer, t for a sampled texture, u for a storage
// buffer or image, s for a sampler.
D3D12_DESCRIPTOR_RANGE_TYPE ToD3D(BindingType type) {
  switch (type) {
    case BindingType::UniformBuffer: return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case BindingType::StorageBuffer:
    case BindingType::StorageTexture:
      return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    case BindingType::SampledTexture: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case BindingType::Sampler: return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
  }
  return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
}

u64 AlignUp(u64 value, u64 alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

class D3D12Device;

// A bump allocator over one descriptor heap. The CPU heaps back the descriptors
// a bind group owns, and do not reclaim them when it is destroyed, so a caller
// that creates groups per draw will wrap the heap; the shader-visible heaps are
// rings a command list copies into, and they wrap without waiting, so they must
// be large enough to hold every descriptor a frame binds. Both want a real
// allocator once the renderer moves onto this seam and starts creating groups
// at frame rate.
class DescriptorHeap {
 public:
  bool Create(ID3D12Device* device,
              D3D12_DESCRIPTOR_HEAP_TYPE type,
              u32 capacity,
              bool shader_visible);
  void Destroy() { ReleaseCom(heap_); }

  u32 Allocate(u32 count);
  void Reset() { cursor_ = 0; }

  D3D12_CPU_DESCRIPTOR_HANDLE Cpu(u32 index) const;
  D3D12_GPU_DESCRIPTOR_HANDLE Gpu(u32 index) const;
  ID3D12DescriptorHeap* heap() const { return heap_; }  // NOLINT: accessor

 private:
  ID3D12DescriptorHeap* heap_ = nullptr;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_{};
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_start_{};
  u32 stride_ = 0;
  u32 capacity_ = 0;
  u32 cursor_ = 0;
};

bool DescriptorHeap::Create(ID3D12Device* device,
                            D3D12_DESCRIPTOR_HEAP_TYPE type,
                            u32 capacity,
                            bool shader_visible) {
  D3D12_DESCRIPTOR_HEAP_DESC hd{};
  hd.Type = type;
  hd.NumDescriptors = capacity;
  hd.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                            : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_ID3D12DescriptorHeap,
                                          (void**)&heap_))) {
    return false;
  }
  stride_ = device->GetDescriptorHandleIncrementSize(type);
  capacity_ = capacity;
  cpu_start_ = heap_->GetCPUDescriptorHandleForHeapStart();
  if (shader_visible)
    gpu_start_ = heap_->GetGPUDescriptorHandleForHeapStart();
  return true;
}

u32 DescriptorHeap::Allocate(u32 count) {
  if (cursor_ + count > capacity_)
    cursor_ = 0;
  const u32 base = cursor_;
  cursor_ += count;
  return base;
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeap::Cpu(u32 index) const {
  D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu_start_;
  handle.ptr += static_cast<SIZE_T>(index) * stride_;
  return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::Gpu(u32 index) const {
  D3D12_GPU_DESCRIPTOR_HANDLE handle = gpu_start_;
  handle.ptr += static_cast<UINT64>(index) * stride_;
  return handle;
}

class D3D12Buffer : public Buffer {
 public:
  D3D12Buffer(D3D12Device& device, const BufferDesc& desc);
  ~D3D12Buffer() override;

  void* Map() override { return mapped_; }

  ID3D12Resource* resource() const { return resource_; }  // NOLINT
  bool valid() const { return resource_ != nullptr; }     // NOLINT
  // An upload or readback resource is created in the one state it may ever be
  // in; barriers on it are illegal.
  bool barriers_allowed() const {  // NOLINT: accessor
    return desc_.memory == MemoryKind::Device;
  }
  D3D12_GPU_VIRTUAL_ADDRESS address() const {  // NOLINT: accessor
    return resource_->GetGPUVirtualAddress();
  }

 private:
  ID3D12Resource* resource_ = nullptr;
  void* mapped_ = nullptr;
};

class D3D12Texture : public Texture {
 public:
  D3D12Texture(D3D12Device& device, const TextureDesc& desc);
  ~D3D12Texture() override;

  ID3D12Resource* resource() const { return resource_; }  // NOLINT
  bool valid() const { return resource_ != nullptr; }     // NOLINT
  DXGI_FORMAT resource_format() const { return resource_format_; }  // NOLINT

  // Render or depth target view of one mip and layer, made on demand.
  D3D12_CPU_DESCRIPTOR_HANDLE TargetView(u32 mip, u32 layer);

 private:
  D3D12Device& device_;
  ID3D12Resource* resource_ = nullptr;
  DXGI_FORMAT resource_format_ = DXGI_FORMAT_UNKNOWN;
  // Keyed by mip | layer << 16.
  std::vector<std::pair<u32, D3D12_CPU_DESCRIPTOR_HANDLE>> target_views_;
};

class D3D12Sampler : public Sampler {
 public:
  explicit D3D12Sampler(const SamplerDesc& desc);

  const D3D12_SAMPLER_DESC& native() const { return native_; }  // NOLINT

 private:
  D3D12_SAMPLER_DESC native_{};
};

class D3D12ShaderModule : public ShaderModule {
 public:
  explicit D3D12ShaderModule(const ShaderDesc& desc);
  ~D3D12ShaderModule() override;

  D3D12_SHADER_BYTECODE bytecode() const;  // NOLINT: accessor
  bool valid() const;                      // NOLINT: accessor

 private:
  ID3DBlob* blob_ = nullptr;
  const void* external_ = nullptr;
  u64 external_bytes_ = 0;
};

class D3D12BindGroupLayout : public BindGroupLayout {
 public:
  explicit D3D12BindGroupLayout(const BindGroupLayoutDesc& desc);

  // Where a binding's descriptor sits within its table. -1 when the binding is
  // not in this layout.
  int ViewSlot(u32 binding) const;
  int SamplerSlot(u32 binding) const;
  const BindGroupEntry* Entry(u32 binding) const;
  u32 num_views() const { return num_views_; }        // NOLINT: accessor
  u32 num_samplers() const { return num_samplers_; }  // NOLINT: accessor

 private:
  u32 num_views_ = 0;
  u32 num_samplers_ = 0;
};

// What a dynamic binding needs to have its descriptor rebuilt with a per-draw
// offset. Everything else is baked into the staging descriptors once.
struct DynamicBinding {
  u32 slot = 0;  // descriptor index within the group's view table
  BindingType type = BindingType::UniformBuffer;
  D3D12Buffer* buffer = nullptr;
  u64 base_offset = 0;
  u64 size = 0;
};

class D3D12BindGroup : public BindGroup {
 public:
  D3D12BindGroup(D3D12Device& device, const BindGroupDesc& desc);

  u32 view_base() const { return view_base_; }            // NOLINT
  u32 sampler_base() const { return sampler_base_; }      // NOLINT
  u32 num_views() const { return num_views_; }            // NOLINT
  u32 num_samplers() const { return num_samplers_; }      // NOLINT
  u32 group_index() const { return group_index_; }        // NOLINT
  const std::vector<DynamicBinding>& dynamics() const {   // NOLINT
    return dynamics_;
  }

 private:
  u32 view_base_ = 0;
  u32 sampler_base_ = 0;
  u32 num_views_ = 0;
  u32 num_samplers_ = 0;
  u32 group_index_ = 0;
  std::vector<DynamicBinding> dynamics_;
};

class D3D12Pipeline : public Pipeline {
 public:
  D3D12Pipeline(D3D12Device& device, const GraphicsPipelineDesc& desc);
  D3D12Pipeline(D3D12Device& device, const ComputePipelineDesc& desc);
  ~D3D12Pipeline() override;

  ID3D12PipelineState* state() const { return state_; }     // NOLINT
  ID3D12RootSignature* root() const { return root_; }       // NOLINT
  bool compute() const { return compute_; }                 // NOLINT
  bool valid() const { return state_ != nullptr; }          // NOLINT
  D3D12_PRIMITIVE_TOPOLOGY topology() const {               // NOLINT
    return topology_;
  }
  // D3D12 carries the vertex stride in the buffer view rather than in the
  // pipeline, so binding a vertex buffer has to read it from here.
  u32 vertex_stride(u32 slot) const {                       // NOLINT
    return slot < kMaxVertexBuffers ? vertex_stride_[slot] : 0;
  }
  int view_root_param(u32 group) const {                    // NOLINT
    return view_root_param_[group];
  }
  int sampler_root_param(u32 group) const {                 // NOLINT
    return sampler_root_param_[group];
  }
  int push_root_param() const { return push_root_param_; }  // NOLINT

 private:
  // Builds the root signature from the bind group layouts: one descriptor
  // table per group for its constant buffers, textures and storage
  // resources, a second for its samplers (D3D12 keeps those in their own
  // heap), and one set of root constants at the end.
  bool CreateRootSignature(const BindGroupLayout* const* groups,
                           u32 num_groups,
                           u32 push_bytes,
                           bool graphics);

  D3D12Device& device_;
  ID3D12PipelineState* state_ = nullptr;
  ID3D12RootSignature* root_ = nullptr;
  bool compute_ = false;
  D3D12_PRIMITIVE_TOPOLOGY topology_ = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
  int view_root_param_[kMaxBindGroups] = {-1, -1, -1, -1};
  int sampler_root_param_[kMaxBindGroups] = {-1, -1, -1, -1};
  int push_root_param_ = -1;
  u32 vertex_stride_[kMaxVertexBuffers] = {};
};

class D3D12CommandList : public CommandList {
 public:
  D3D12CommandList(D3D12Device& device,
                   ID3D12CommandAllocator* allocator,
                   ID3D12GraphicsCommandList* list);
  ~D3D12CommandList() override;

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

  ID3D12GraphicsCommandList* list() const { return list_; }  // NOLINT

 private:
  void BindHeaps();

  D3D12Device& device_;
  ID3D12CommandAllocator* allocator_ = nullptr;
  ID3D12GraphicsCommandList* list_ = nullptr;
  D3D12Pipeline* pipeline_ = nullptr;
};

class D3D12Device : public Device {
 public:
  D3D12Device() = default;
  ~D3D12Device() override;

  bool Create();

  const char* backend_name() const override { return "d3d12"; }  // NOLINT
  const DeviceCaps& caps() const override { return caps_; }      // NOLINT

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

  ID3D12Device* native() const { return device_; }  // NOLINT: accessor
  DescriptorHeap& rtv_heap() { return rtv_heap_; }
  DescriptorHeap& dsv_heap() { return dsv_heap_; }
  DescriptorHeap& staging_view_heap() { return staging_view_heap_; }
  DescriptorHeap& staging_sampler_heap() { return staging_sampler_heap_; }
  DescriptorHeap& gpu_view_heap() { return gpu_view_heap_; }
  DescriptorHeap& gpu_sampler_heap() { return gpu_sampler_heap_; }

 private:
  ID3D12Device* device_ = nullptr;
  ID3D12CommandQueue* queue_ = nullptr;
  ID3D12Fence* fence_ = nullptr;
  u64 submissions_ = 0;
  DescriptorHeap rtv_heap_;
  DescriptorHeap dsv_heap_;
  DescriptorHeap staging_view_heap_;
  DescriptorHeap staging_sampler_heap_;
  DescriptorHeap gpu_view_heap_;
  DescriptorHeap gpu_sampler_heap_;
  DeviceCaps caps_;
};

D3D12Buffer::D3D12Buffer(D3D12Device& device, const BufferDesc& desc) {
  desc_ = desc;

  D3D12_HEAP_PROPERTIES heap{};
  D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON;
  switch (desc.memory) {
    case MemoryKind::Device:
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      break;
    case MemoryKind::Upload:
      heap.Type = D3D12_HEAP_TYPE_UPLOAD;
      initial = D3D12_RESOURCE_STATE_GENERIC_READ;
      break;
    case MemoryKind::Readback:
      heap.Type = D3D12_HEAP_TYPE_READBACK;
      initial = D3D12_RESOURCE_STATE_COPY_DEST;
      break;
  }

  D3D12_RESOURCE_DESC rd{};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = desc.size ? desc.size : 1;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.Format = DXGI_FORMAT_UNKNOWN;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  if ((desc.usage & kBufferStorage) && desc.memory == MemoryKind::Device)
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  if (FAILED(device.native()->CreateCommittedResource(
          &heap, D3D12_HEAP_FLAG_NONE, &rd, initial, nullptr,
          IID_ID3D12Resource, (void**)&resource_))) {
    BASE_LOGI("rhid3d", "buffer of {} bytes failed", desc.size);
    return;
  }
  if (desc.memory != MemoryKind::Device) {
    const D3D12_RANGE none{0, 0};
    resource_->Map(0, desc.memory == MemoryKind::Upload ? &none : nullptr,
                   &mapped_);
  }
  state_ = desc.memory == MemoryKind::Readback ? ResourceState::CopyDst
                                               : ResourceState::Common;
}

D3D12Buffer::~D3D12Buffer() {
  ReleaseCom(resource_);
}

D3D12Texture::D3D12Texture(D3D12Device& device, const TextureDesc& desc)
    : device_(device) {
  desc_ = desc;

  const bool depth = FormatIsDepth(desc.format);
  const DepthFormats formats = DepthFormatsFor(desc.format);
  // A depth surface the shader also samples has to be typeless, so its depth
  // and shader views can disagree about how to read it.
  resource_format_ = depth && (desc.usage & kTextureSampled)
                         ? formats.resource
                         : ToDxgi(desc.format);

  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;

  D3D12_RESOURCE_DESC rd{};
  rd.Dimension = desc.depth > 1 ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  rd.Width = desc.width;
  rd.Height = desc.height;
  rd.DepthOrArraySize = static_cast<UINT16>(
      desc.depth > 1 ? desc.depth : desc.array_layers);
  rd.MipLevels = static_cast<UINT16>(desc.mip_levels);
  rd.Format = resource_format_;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  if (desc.usage & kTextureRenderTarget)
    rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  if (desc.usage & kTextureDepthStencil)
    rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
  if (desc.usage & kTextureStorage)
    rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if ((desc.usage & kTextureDepthStencil) && !(desc.usage & kTextureSampled))
    rd.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;

  if (FAILED(device.native()->CreateCommittedResource(
          &heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
          nullptr, IID_ID3D12Resource, (void**)&resource_))) {
    BASE_LOGI("rhid3d", "texture {}x{} {} failed", desc.width, desc.height,
              FormatName(desc.format));
    return;
  }
  state_ = ResourceState::Common;
}

D3D12Texture::~D3D12Texture() {
  ReleaseCom(resource_);
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Texture::TargetView(u32 mip, u32 layer) {
  const u32 key = mip | (layer << 16);
  for (const auto& [k, handle] : target_views_) {
    if (k == key)
      return handle;
  }

  const bool depth = FormatIsDepth(desc_.format);
  DescriptorHeap& heap = depth ? device_.dsv_heap() : device_.rtv_heap();
  const u32 index = heap.Allocate(1);
  const D3D12_CPU_DESCRIPTOR_HANDLE handle = heap.Cpu(index);
  const bool arrayed = desc_.array_layers > 1;

  if (depth) {
    D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = DepthFormatsFor(desc_.format).view;
    if (arrayed) {
      dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
      dv.Texture2DArray.MipSlice = mip;
      dv.Texture2DArray.FirstArraySlice = layer;
      dv.Texture2DArray.ArraySize = 1;
    } else {
      dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
      dv.Texture2D.MipSlice = mip;
    }
    device_.native()->CreateDepthStencilView(resource_, &dv, handle);
  } else {
    D3D12_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = ToDxgi(desc_.format);
    if (desc_.depth > 1) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
      rv.Texture3D.MipSlice = mip;
      rv.Texture3D.FirstWSlice = layer;
      rv.Texture3D.WSize = 1;
    } else if (arrayed) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
      rv.Texture2DArray.MipSlice = mip;
      rv.Texture2DArray.FirstArraySlice = layer;
      rv.Texture2DArray.ArraySize = 1;
    } else {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
      rv.Texture2D.MipSlice = mip;
    }
    device_.native()->CreateRenderTargetView(resource_, &rv, handle);
  }
  target_views_.emplace_back(key, handle);
  return handle;
}

D3D12Sampler::D3D12Sampler(const SamplerDesc& desc) {
  // D3D12 packs the three filters into one enum value. Only the combinations
  // the guest can ask for are built here: point/linear per stage, and the
  // comparison variants.
  UINT filter = 0;
  if (desc.min_filter == FilterMode::Linear)
    filter |= 0x10;
  if (desc.mag_filter == FilterMode::Linear)
    filter |= 0x4;
  if (desc.mip_filter == FilterMode::Linear)
    filter |= 0x1;
  if (desc.max_anisotropy > 1)
    filter = D3D12_FILTER_ANISOTROPIC;
  if (desc.compare_enable)
    filter |= 0x80;  // the comparison reduction bit

  native_.Filter = static_cast<D3D12_FILTER>(filter);
  native_.AddressU = ToD3D(desc.address_u);
  native_.AddressV = ToD3D(desc.address_v);
  native_.AddressW = ToD3D(desc.address_w);
  native_.MipLODBias = desc.lod_bias;
  native_.MaxAnisotropy = desc.max_anisotropy;
  native_.ComparisonFunc = desc.compare_enable
                               ? ToD3D(desc.compare)
                               : D3D12_COMPARISON_FUNC_NEVER;
  native_.MinLOD = desc.min_lod;
  native_.MaxLOD = desc.max_lod;
  const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  const float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float* border = clear;
  if (desc.border == BorderColor::OpaqueBlack)
    border = black;
  else if (desc.border == BorderColor::OpaqueWhite)
    border = white;
  memcpy(native_.BorderColor, border, sizeof(native_.BorderColor));
}

D3D12ShaderModule::D3D12ShaderModule(const ShaderDesc& desc) {
  if (desc.dxbc && desc.dxbc_bytes) {
    external_ = desc.dxbc;
    external_bytes_ = desc.dxbc_bytes;
    return;
  }
  if (!desc.hlsl || !desc.hlsl_target) {
    BASE_LOGI("rhid3d", "shader '{}' has neither DXBC nor HLSL",
              desc.name ? desc.name : "?");
    return;
  }
  ID3DBlob* errors = nullptr;
  const HRESULT hr = D3DCompile(desc.hlsl, strlen(desc.hlsl),
                                desc.name ? desc.name : "shader", nullptr,
                                nullptr, desc.entry ? desc.entry : "main",
                                desc.hlsl_target, 0, 0, &blob_, &errors);
  if (FAILED(hr)) {
    BASE_LOGI("rhid3d", "compiling '{}' as {} failed: {}",
              desc.name ? desc.name : "?", desc.hlsl_target,
              errors ? (const char*)errors->GetBufferPointer() : "no message");
  }
  ReleaseCom(errors);
}

D3D12ShaderModule::~D3D12ShaderModule() {
  ReleaseCom(blob_);
}

bool D3D12ShaderModule::valid() const {
  return blob_ != nullptr || external_ != nullptr;
}

D3D12_SHADER_BYTECODE D3D12ShaderModule::bytecode() const {
  D3D12_SHADER_BYTECODE code{};
  if (blob_) {
    code.pShaderBytecode = blob_->GetBufferPointer();
    code.BytecodeLength = blob_->GetBufferSize();
  } else {
    code.pShaderBytecode = external_;
    code.BytecodeLength = external_bytes_;
  }
  return code;
}

D3D12BindGroupLayout::D3D12BindGroupLayout(const BindGroupLayoutDesc& desc) {
  desc_ = desc;
  for (u32 i = 0; i < desc.num_entries; i++) {
    if (desc.entries[i].type == BindingType::Sampler)
      num_samplers_++;
    else
      num_views_++;
  }
}

const BindGroupEntry* D3D12BindGroupLayout::Entry(u32 binding) const {
  for (u32 i = 0; i < desc_.num_entries; i++) {
    if (desc_.entries[i].binding == binding)
      return &desc_.entries[i];
  }
  return nullptr;
}

int D3D12BindGroupLayout::ViewSlot(u32 binding) const {
  int slot = 0;
  for (u32 i = 0; i < desc_.num_entries; i++) {
    if (desc_.entries[i].type == BindingType::Sampler)
      continue;
    if (desc_.entries[i].binding == binding)
      return slot;
    slot++;
  }
  return -1;
}

int D3D12BindGroupLayout::SamplerSlot(u32 binding) const {
  int slot = 0;
  for (u32 i = 0; i < desc_.num_entries; i++) {
    if (desc_.entries[i].type != BindingType::Sampler)
      continue;
    if (desc_.entries[i].binding == binding)
      return slot;
    slot++;
  }
  return -1;
}

D3D12BindGroup::D3D12BindGroup(D3D12Device& device,
                               const BindGroupDesc& desc) {
  const auto* layout = static_cast<const D3D12BindGroupLayout*>(desc.layout);
  if (!layout)
    return;
  group_index_ = layout->desc().index;
  num_views_ = layout->num_views();
  num_samplers_ = layout->num_samplers();
  if (num_views_)
    view_base_ = device.staging_view_heap().Allocate(num_views_);
  if (num_samplers_)
    sampler_base_ = device.staging_sampler_heap().Allocate(num_samplers_);

  for (u32 i = 0; i < desc.num_bindings; i++) {
    const BindGroupBinding& b = desc.bindings[i];
    const BindGroupEntry* entry = layout->Entry(b.binding);
    if (!entry)
      continue;

    if (entry->type == BindingType::Sampler) {
      const int slot = layout->SamplerSlot(b.binding);
      auto* sampler = static_cast<D3D12Sampler*>(b.sampler);
      if (slot < 0 || !sampler)
        continue;
      device.native()->CreateSampler(
          &sampler->native(),
          device.staging_sampler_heap().Cpu(sampler_base_ + slot));
      continue;
    }

    const int slot = layout->ViewSlot(b.binding);
    if (slot < 0)
      continue;
    const D3D12_CPU_DESCRIPTOR_HANDLE handle =
        device.staging_view_heap().Cpu(view_base_ + slot);

    switch (entry->type) {
      case BindingType::UniformBuffer: {
        auto* buffer = static_cast<D3D12Buffer*>(b.buffer);
        if (!buffer)
          continue;
        const u64 size =
            b.size ? b.size : buffer->desc().size - b.offset;
        if (entry->dynamic) {
          dynamics_.push_back({static_cast<u32>(slot), entry->type, buffer,
                               b.offset, size});
        }
        D3D12_CONSTANT_BUFFER_VIEW_DESC cv{};
        cv.BufferLocation = buffer->address() + b.offset;
        cv.SizeInBytes = static_cast<UINT>(AlignUp(size, 256));
        device.native()->CreateConstantBufferView(&cv, handle);
        break;
      }
      case BindingType::StorageBuffer: {
        auto* buffer = static_cast<D3D12Buffer*>(b.buffer);
        if (!buffer)
          continue;
        const u64 size =
            b.size ? b.size : buffer->desc().size - b.offset;
        if (entry->dynamic) {
          dynamics_.push_back({static_cast<u32>(slot), entry->type, buffer,
                               b.offset, size});
        }
        // A raw (byte address) view, which is what a shader indexing the
        // buffer itself wants; a structured view would need the stride the
        // guest never states.
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = DXGI_FORMAT_R32_TYPELESS;
        uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.FirstElement = b.offset / 4;
        uv.Buffer.NumElements = static_cast<UINT>(size / 4);
        uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device.native()->CreateUnorderedAccessView(buffer->resource(), nullptr,
                                                   &uv, handle);
        break;
      }
      case BindingType::SampledTexture: {
        auto* texture = static_cast<D3D12Texture*>(b.texture);
        if (!texture)
          continue;
        const TextureDesc& td = texture->desc();
        const Format view_format =
            b.view_format == Format::Unknown ? td.format : b.view_format;
        D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = FormatIsDepth(view_format)
                        ? DepthFormatsFor(view_format).shader
                        : ToDxgi(view_format);
        sv.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (td.depth > 1) {
          sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
          sv.Texture3D.MostDetailedMip = b.base_mip;
          sv.Texture3D.MipLevels = b.mips;
        } else if (b.layers > 1) {
          sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
          sv.Texture2DArray.MostDetailedMip = b.base_mip;
          sv.Texture2DArray.MipLevels = b.mips;
          sv.Texture2DArray.FirstArraySlice = b.base_layer;
          sv.Texture2DArray.ArraySize = b.layers;
        } else {
          sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
          sv.Texture2D.MostDetailedMip = b.base_mip;
          sv.Texture2D.MipLevels = b.mips;
        }
        device.native()->CreateShaderResourceView(texture->resource(), &sv,
                                                  handle);
        break;
      }
      case BindingType::StorageTexture: {
        auto* texture = static_cast<D3D12Texture*>(b.texture);
        if (!texture)
          continue;
        const Format view_format = b.view_format == Format::Unknown
                                       ? texture->desc().format
                                       : b.view_format;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = ToDxgi(view_format);
        if (texture->desc().depth > 1) {
          uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
          uv.Texture3D.MipSlice = b.base_mip;
          uv.Texture3D.WSize = texture->desc().depth;
        } else {
          uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
          uv.Texture2D.MipSlice = b.base_mip;
        }
        device.native()->CreateUnorderedAccessView(texture->resource(),
                                                   nullptr, &uv, handle);
        break;
      }
      case BindingType::Sampler: break;  // handled above
    }
  }
}

bool D3D12Pipeline::CreateRootSignature(const BindGroupLayout* const* groups,
                                        u32 num_groups,
                                        u32 push_bytes,
                                        bool graphics) {
  std::vector<D3D12_ROOT_PARAMETER> params;
  // One vector per table: the ranges have to stay alive until serialization.
  std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> ranges;
  ranges.reserve(num_groups * 2 + 1);

  for (u32 g = 0; g < num_groups; g++) {
    const auto* layout = static_cast<const D3D12BindGroupLayout*>(groups[g]);
    if (!layout)
      continue;
    const BindGroupLayoutDesc& ld = layout->desc();

    std::vector<D3D12_DESCRIPTOR_RANGE> views;
    std::vector<D3D12_DESCRIPTOR_RANGE> samplers;
    u32 view_slot = 0;
    u32 sampler_slot = 0;
    for (u32 i = 0; i < ld.num_entries; i++) {
      const BindGroupEntry& e = ld.entries[i];
      D3D12_DESCRIPTOR_RANGE range{};
      range.RangeType = ToD3D(e.type);
      range.NumDescriptors = 1;
      range.BaseShaderRegister = e.binding;
      range.RegisterSpace = ld.index;
      if (e.type == BindingType::Sampler) {
        range.OffsetInDescriptorsFromTableStart = sampler_slot++;
        samplers.push_back(range);
      } else {
        range.OffsetInDescriptorsFromTableStart = view_slot++;
        views.push_back(range);
      }
    }

    if (!views.empty()) {
      ranges.push_back(std::move(views));
      D3D12_ROOT_PARAMETER param{};
      param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      param.DescriptorTable.NumDescriptorRanges =
          static_cast<UINT>(ranges.back().size());
      param.DescriptorTable.pDescriptorRanges = ranges.back().data();
      view_root_param_[g] = static_cast<int>(params.size());
      params.push_back(param);
    }
    if (!samplers.empty()) {
      ranges.push_back(std::move(samplers));
      D3D12_ROOT_PARAMETER param{};
      param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      param.DescriptorTable.NumDescriptorRanges =
          static_cast<UINT>(ranges.back().size());
      param.DescriptorTable.pDescriptorRanges = ranges.back().data();
      sampler_root_param_[g] = static_cast<int>(params.size());
      params.push_back(param);
    }
  }

  if (push_bytes) {
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    param.Constants.ShaderRegister = 0;
    param.Constants.RegisterSpace = kPushConstantSpace;
    param.Constants.Num32BitValues = (push_bytes + 3) / 4;
    push_root_param_ = static_cast<int>(params.size());
    params.push_back(param);
  }

  D3D12_ROOT_SIGNATURE_DESC rs{};
  rs.NumParameters = static_cast<UINT>(params.size());
  rs.pParameters = params.empty() ? nullptr : params.data();
  rs.Flags = graphics
                 ? D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                 : D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ID3DBlob* blob = nullptr;
  ID3DBlob* errors = nullptr;
  HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &blob, &errors);
  if (FAILED(hr)) {
    BASE_LOGI("rhid3d", "root signature failed: {}",
              errors ? (const char*)errors->GetBufferPointer() : "no message");
    ReleaseCom(errors);
    return false;
  }
  ReleaseCom(errors);
  hr = device_.native()->CreateRootSignature(
      0, blob->GetBufferPointer(), blob->GetBufferSize(),
      IID_ID3D12RootSignature, (void**)&root_);
  ReleaseCom(blob);
  return SUCCEEDED(hr);
}

D3D12Pipeline::D3D12Pipeline(D3D12Device& device,
                             const GraphicsPipelineDesc& desc)
    : device_(device) {
  const auto* vs = static_cast<const D3D12ShaderModule*>(desc.vertex);
  const auto* ps = static_cast<const D3D12ShaderModule*>(desc.pixel);
  if (!vs || !vs->valid()) {
    BASE_LOGI("rhid3d", "pipeline '{}' has no vertex shader",
              desc.name ? desc.name : "?");
    return;
  }
  if (!CreateRootSignature(desc.bind_groups, desc.num_bind_groups,
                           desc.push_constant_bytes, true)) {
    return;
  }
  topology_ = ToD3DTopology(desc.topology);
  for (u32 i = 0; i < desc.num_vertex_buffers; i++)
    vertex_stride_[i] = desc.vertex_buffers[i].stride;

  // Vertex inputs are matched by semantic. One name with the attribute's
  // location as its index is the convention the shaders on this seam use, so
  // an HLSL input reads `float4 pos : ATTRIB0`.
  D3D12_INPUT_ELEMENT_DESC elements[kMaxVertexAttributes]{};
  for (u32 i = 0; i < desc.num_attributes; i++) {
    const VertexAttribute& a = desc.attributes[i];
    elements[i].SemanticName = "ATTRIB";
    elements[i].SemanticIndex = a.location;
    elements[i].Format = ToDxgi(a.format);
    elements[i].InputSlot = a.buffer;
    elements[i].AlignedByteOffset = a.offset;
    elements[i].InputSlotClass =
        desc.vertex_buffers[a.buffer].per_instance
            ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
            : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    elements[i].InstanceDataStepRate =
        desc.vertex_buffers[a.buffer].per_instance ? 1 : 0;
  }

  D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
  pd.pRootSignature = root_;
  pd.VS = vs->bytecode();
  if (ps && ps->valid())
    pd.PS = ps->bytecode();
  pd.InputLayout.pInputElementDescs = elements;
  pd.InputLayout.NumElements = desc.num_attributes;
  pd.PrimitiveTopologyType = ToD3DType(desc.topology);
  pd.SampleMask = 0xFFFFFFFFu;
  pd.SampleDesc.Count = 1;

  pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  switch (desc.raster.cull) {
    case CullMode::None:
      pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
      break;
    case CullMode::Front:
      pd.RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
      break;
    case CullMode::Back:
      pd.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
      break;
  }
  pd.RasterizerState.FrontCounterClockwise = desc.raster.front_ccw;
  pd.RasterizerState.DepthClipEnable = desc.raster.depth_clip;

  pd.DepthStencilState.DepthEnable = desc.depth.test_enable;
  pd.DepthStencilState.DepthWriteMask = desc.depth.write_enable
                                            ? D3D12_DEPTH_WRITE_MASK_ALL
                                            : D3D12_DEPTH_WRITE_MASK_ZERO;
  pd.DepthStencilState.DepthFunc = ToD3D(desc.depth.compare);
  pd.DSVFormat = desc.depth_format == Format::Unknown
                     ? DXGI_FORMAT_UNKNOWN
                     : DepthFormatsFor(desc.depth_format).view;

  pd.BlendState.IndependentBlendEnable = desc.num_color_targets > 1;
  pd.NumRenderTargets = desc.num_color_targets;
  for (u32 i = 0; i < desc.num_color_targets; i++) {
    const BlendState& b = desc.blend[i];
    D3D12_RENDER_TARGET_BLEND_DESC& rt = pd.BlendState.RenderTarget[i];
    rt.BlendEnable = b.enable;
    rt.SrcBlend = ToD3D(b.src_color, false);
    rt.DestBlend = ToD3D(b.dst_color, false);
    rt.BlendOp = ToD3D(b.color_op);
    rt.SrcBlendAlpha = ToD3D(b.src_alpha, true);
    rt.DestBlendAlpha = ToD3D(b.dst_alpha, true);
    rt.BlendOpAlpha = ToD3D(b.alpha_op);
    rt.LogicOp = D3D12_LOGIC_OP_NOOP;
    rt.RenderTargetWriteMask = b.write_mask;
    pd.RTVFormats[i] = ToDxgi(desc.color_formats[i]);
  }

  if (FAILED(device_.native()->CreateGraphicsPipelineState(
          &pd, IID_ID3D12PipelineState, (void**)&state_))) {
    BASE_LOGI("rhid3d", "graphics pipeline '{}' failed",
              desc.name ? desc.name : "?");
  }
}

D3D12Pipeline::D3D12Pipeline(D3D12Device& device,
                             const ComputePipelineDesc& desc)
    : device_(device), compute_(true) {
  const auto* cs = static_cast<const D3D12ShaderModule*>(desc.compute);
  if (!cs || !cs->valid())
    return;
  if (!CreateRootSignature(desc.bind_groups, desc.num_bind_groups,
                           desc.push_constant_bytes, false)) {
    return;
  }
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
  pd.pRootSignature = root_;
  pd.CS = cs->bytecode();
  if (FAILED(device_.native()->CreateComputePipelineState(
          &pd, IID_ID3D12PipelineState, (void**)&state_))) {
    BASE_LOGI("rhid3d", "compute pipeline '{}' failed",
              desc.name ? desc.name : "?");
  }
}

D3D12Pipeline::~D3D12Pipeline() {
  ReleaseCom(state_);
  ReleaseCom(root_);
}

D3D12CommandList::D3D12CommandList(D3D12Device& device,
                                   ID3D12CommandAllocator* allocator,
                                   ID3D12GraphicsCommandList* list)
    : device_(device), allocator_(allocator), list_(list) {}

D3D12CommandList::~D3D12CommandList() {
  ReleaseCom(list_);
  ReleaseCom(allocator_);
}

void D3D12CommandList::BindHeaps() {
  ID3D12DescriptorHeap* heaps[2] = {device_.gpu_view_heap().heap(),
                                    device_.gpu_sampler_heap().heap()};
  list_->SetDescriptorHeaps(2, heaps);
}

void D3D12CommandList::Begin() {
  allocator_->Reset();
  list_->Reset(allocator_, nullptr);
  pipeline_ = nullptr;
  BindHeaps();
}

void D3D12CommandList::End() {
  list_->Close();
}

void D3D12CommandList::BeginRenderPass(const RenderPassDesc& pass) {
  D3D12_CPU_DESCRIPTOR_HANDLE rtvs[kMaxColorTargets]{};
  for (u32 i = 0; i < pass.num_color; i++) {
    const ColorAttachment& a = pass.color[i];
    auto* texture = static_cast<D3D12Texture*>(a.texture);
    Transition(texture, ResourceState::RenderTarget);
    rtvs[i] = texture->TargetView(a.mip, a.layer);
  }

  D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
  bool has_depth = false;
  if (pass.has_depth && pass.depth.texture) {
    auto* texture = static_cast<D3D12Texture*>(pass.depth.texture);
    Transition(texture, pass.depth.read_only ? ResourceState::DepthRead
                                             : ResourceState::DepthWrite);
    dsv = texture->TargetView(0, 0);
    has_depth = true;
  }

  list_->OMSetRenderTargets(pass.num_color, rtvs, FALSE,
                            has_depth ? &dsv : nullptr);

  // D3D12 has no load op: a pass that wants a cleared attachment clears it
  // here, which is the same work the driver does behind Vulkan's loadOp.
  for (u32 i = 0; i < pass.num_color; i++) {
    if (pass.color[i].load == LoadOp::Clear)
      list_->ClearRenderTargetView(rtvs[i], pass.color[i].clear, 0, nullptr);
  }
  if (has_depth && pass.depth.load == LoadOp::Clear) {
    UINT flags = D3D12_CLEAR_FLAG_DEPTH;
    if (FormatHasStencil(pass.depth.texture->desc().format))
      flags |= D3D12_CLEAR_FLAG_STENCIL;
    list_->ClearDepthStencilView(dsv, static_cast<D3D12_CLEAR_FLAGS>(flags),
                                 pass.depth.clear_depth,
                                 pass.depth.clear_stencil, 0, nullptr);
  }
}

void D3D12CommandList::EndRenderPass() {}

void D3D12CommandList::SetPipeline(Pipeline* pipeline) {
  pipeline_ = static_cast<D3D12Pipeline*>(pipeline);
  if (!pipeline_ || !pipeline_->valid())
    return;
  list_->SetPipelineState(pipeline_->state());
  if (pipeline_->compute()) {
    list_->SetComputeRootSignature(pipeline_->root());
  } else {
    list_->SetGraphicsRootSignature(pipeline_->root());
    list_->IASetPrimitiveTopology(pipeline_->topology());
  }
}

void D3D12CommandList::SetBindGroup(u32 index,
                                    BindGroup* group,
                                    const u32* offsets,
                                    u32 num_offsets) {
  if (!pipeline_ || !group)
    return;
  auto* d3d_group = static_cast<D3D12BindGroup*>(group);

  if (d3d_group->num_views()) {
    // Shader-visible descriptors have to live in the heap the command list
    // bound, so the group's staged copies move into this frame's ring.
    const u32 base =
        device_.gpu_view_heap().Allocate(d3d_group->num_views());
    device_.native()->CopyDescriptorsSimple(
        d3d_group->num_views(), device_.gpu_view_heap().Cpu(base),
        device_.staging_view_heap().Cpu(d3d_group->view_base()),
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // A dynamic binding's offset is not known until the draw, so its
    // descriptor is written now rather than copied.
    u32 next = 0;
    for (const DynamicBinding& dynamic : d3d_group->dynamics()) {
      const u32 offset = next < num_offsets ? offsets[next] : 0;
      next++;
      const D3D12_CPU_DESCRIPTOR_HANDLE handle =
          device_.gpu_view_heap().Cpu(base + dynamic.slot);
      if (dynamic.type == BindingType::UniformBuffer) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cv{};
        cv.BufferLocation = dynamic.buffer->address() + dynamic.base_offset +
                            offset;
        cv.SizeInBytes = static_cast<UINT>(AlignUp(dynamic.size, 256));
        device_.native()->CreateConstantBufferView(&cv, handle);
      } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = DXGI_FORMAT_R32_TYPELESS;
        uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.FirstElement = (dynamic.base_offset + offset) / 4;
        uv.Buffer.NumElements = static_cast<UINT>(dynamic.size / 4);
        uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_.native()->CreateUnorderedAccessView(
            dynamic.buffer->resource(), nullptr, &uv, handle);
      }
    }

    const int param = pipeline_->view_root_param(index);
    if (param >= 0) {
      const D3D12_GPU_DESCRIPTOR_HANDLE gpu =
          device_.gpu_view_heap().Gpu(base);
      if (pipeline_->compute())
        list_->SetComputeRootDescriptorTable(param, gpu);
      else
        list_->SetGraphicsRootDescriptorTable(param, gpu);
    }
  }

  if (d3d_group->num_samplers()) {
    const u32 base =
        device_.gpu_sampler_heap().Allocate(d3d_group->num_samplers());
    device_.native()->CopyDescriptorsSimple(
        d3d_group->num_samplers(), device_.gpu_sampler_heap().Cpu(base),
        device_.staging_sampler_heap().Cpu(d3d_group->sampler_base()),
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    const int param = pipeline_->sampler_root_param(index);
    if (param >= 0) {
      const D3D12_GPU_DESCRIPTOR_HANDLE gpu =
          device_.gpu_sampler_heap().Gpu(base);
      if (pipeline_->compute())
        list_->SetComputeRootDescriptorTable(param, gpu);
      else
        list_->SetGraphicsRootDescriptorTable(param, gpu);
    }
  }
}

void D3D12CommandList::SetPushConstants(u32 offset,
                                        u32 bytes,
                                        const void* data) {
  if (!pipeline_ || pipeline_->push_root_param() < 0 || !bytes)
    return;
  const UINT num = (bytes + 3) / 4;
  if (pipeline_->compute()) {
    list_->SetComputeRoot32BitConstants(pipeline_->push_root_param(), num,
                                        data, offset / 4);
  } else {
    list_->SetGraphicsRoot32BitConstants(pipeline_->push_root_param(), num,
                                         data, offset / 4);
  }
}

void D3D12CommandList::SetVertexBuffer(u32 slot,
                                       Buffer* buffer,
                                       u64 offset) {
  if (!buffer)
    return;
  auto* d3d_buffer = static_cast<D3D12Buffer*>(buffer);
  D3D12_VERTEX_BUFFER_VIEW view{};
  view.BufferLocation = d3d_buffer->address() + offset;
  view.SizeInBytes = static_cast<UINT>(d3d_buffer->desc().size - offset);
  view.StrideInBytes = pipeline_ ? pipeline_->vertex_stride(slot) : 0;
  list_->IASetVertexBuffers(slot, 1, &view);
}

void D3D12CommandList::SetIndexBuffer(Buffer* buffer,
                                      u64 offset,
                                      IndexType type) {
  if (!buffer)
    return;
  auto* d3d_buffer = static_cast<D3D12Buffer*>(buffer);
  D3D12_INDEX_BUFFER_VIEW view{};
  view.BufferLocation = d3d_buffer->address() + offset;
  view.SizeInBytes = static_cast<UINT>(d3d_buffer->desc().size - offset);
  view.Format = type == IndexType::U16 ? DXGI_FORMAT_R16_UINT
                                       : DXGI_FORMAT_R32_UINT;
  list_->IASetIndexBuffer(&view);
}

void D3D12CommandList::SetViewport(float x,
                                   float y,
                                   float width,
                                   float height,
                                   float min_depth,
                                   float max_depth) {
  D3D12_VIEWPORT vp{x, y, width, height, min_depth, max_depth};
  list_->RSSetViewports(1, &vp);
}

void D3D12CommandList::SetScissor(i32 x, i32 y, u32 width, u32 height) {
  D3D12_RECT rect{x, y, x + static_cast<LONG>(width),
                  y + static_cast<LONG>(height)};
  list_->RSSetScissorRects(1, &rect);
}

void D3D12CommandList::SetBlendConstants(const float rgba[4]) {
  list_->OMSetBlendFactor(rgba);
}

void D3D12CommandList::Draw(u32 vertex_count,
                            u32 instance_count,
                            u32 first_vertex,
                            u32 first_instance) {
  list_->DrawInstanced(vertex_count, instance_count, first_vertex,
                       first_instance);
}

void D3D12CommandList::DrawIndexed(u32 index_count,
                                   u32 instance_count,
                                   u32 first_index,
                                   i32 vertex_offset,
                                   u32 first_instance) {
  list_->DrawIndexedInstanced(index_count, instance_count, first_index,
                              vertex_offset, first_instance);
}

void D3D12CommandList::Dispatch(u32 groups_x, u32 groups_y, u32 groups_z) {
  list_->Dispatch(groups_x, groups_y, groups_z);
}

void D3D12CommandList::CopyBuffer(Buffer* dst,
                                  u64 dst_offset,
                                  Buffer* src,
                                  u64 src_offset,
                                  u64 bytes) {
  list_->CopyBufferRegion(static_cast<D3D12Buffer*>(dst)->resource(),
                          dst_offset,
                          static_cast<D3D12Buffer*>(src)->resource(),
                          src_offset, bytes);
}

void D3D12CommandList::CopyBufferToTexture(Texture* dst,
                                           const TextureRegion& region,
                                           Buffer* src,
                                           u64 src_offset,
                                           u32 src_row_pitch) {
  auto* texture = static_cast<D3D12Texture*>(dst);
  const TextureDesc& td = texture->desc();

  D3D12_TEXTURE_COPY_LOCATION to{};
  to.pResource = texture->resource();
  to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  to.SubresourceIndex = region.mip + region.layer * td.mip_levels;

  D3D12_TEXTURE_COPY_LOCATION from{};
  from.pResource = static_cast<D3D12Buffer*>(src)->resource();
  from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  from.PlacedFootprint.Offset = src_offset;
  from.PlacedFootprint.Footprint.Format = texture->resource_format();
  from.PlacedFootprint.Footprint.Width = region.width;
  from.PlacedFootprint.Footprint.Height = region.height;
  from.PlacedFootprint.Footprint.Depth = region.depth;
  from.PlacedFootprint.Footprint.RowPitch =
      src_row_pitch
          ? src_row_pitch
          : static_cast<UINT>(FormatRowBytes(td.format, region.width));

  list_->CopyTextureRegion(&to, region.x, region.y, region.z, &from, nullptr);
}

void D3D12CommandList::CopyTextureToBuffer(Buffer* dst,
                                           u64 dst_offset,
                                           u32 dst_row_pitch,
                                           Texture* src,
                                           const TextureRegion& region) {
  auto* texture = static_cast<D3D12Texture*>(src);
  const TextureDesc& td = texture->desc();

  D3D12_TEXTURE_COPY_LOCATION to{};
  to.pResource = static_cast<D3D12Buffer*>(dst)->resource();
  to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  to.PlacedFootprint.Offset = dst_offset;
  to.PlacedFootprint.Footprint.Format = texture->resource_format();
  to.PlacedFootprint.Footprint.Width = region.width;
  to.PlacedFootprint.Footprint.Height = region.height;
  to.PlacedFootprint.Footprint.Depth = region.depth;
  to.PlacedFootprint.Footprint.RowPitch =
      dst_row_pitch
          ? dst_row_pitch
          : static_cast<UINT>(FormatRowBytes(td.format, region.width));

  D3D12_TEXTURE_COPY_LOCATION from{};
  from.pResource = texture->resource();
  from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  from.SubresourceIndex = region.mip + region.layer * td.mip_levels;

  D3D12_BOX box{};
  box.left = region.x;
  box.top = region.y;
  box.front = region.z;
  box.right = region.x + region.width;
  box.bottom = region.y + region.height;
  box.back = region.z + region.depth;
  list_->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
}

void D3D12CommandList::FillBuffer(Buffer* buffer,
                                  u64 offset,
                                  u64 bytes,
                                  u32 value) {
  auto* d3d_buffer = static_cast<D3D12Buffer*>(buffer);
  // Clearing needs the view twice: shader-visible for the GPU to write
  // through, and CPU-only for the driver to read the descriptor from.
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
  uv.Format = DXGI_FORMAT_R32_TYPELESS;
  uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uv.Buffer.FirstElement = offset / 4;
  uv.Buffer.NumElements = static_cast<UINT>(bytes / 4);
  uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

  const u32 staging = device_.staging_view_heap().Allocate(1);
  const u32 visible = device_.gpu_view_heap().Allocate(1);
  device_.native()->CreateUnorderedAccessView(
      d3d_buffer->resource(), nullptr, &uv,
      device_.staging_view_heap().Cpu(staging));
  device_.native()->CreateUnorderedAccessView(
      d3d_buffer->resource(), nullptr, &uv,
      device_.gpu_view_heap().Cpu(visible));

  const UINT values[4] = {value, value, value, value};
  list_->ClearUnorderedAccessViewUint(
      device_.gpu_view_heap().Gpu(visible),
      device_.staging_view_heap().Cpu(staging), d3d_buffer->resource(),
      values, 0, nullptr);
}

void D3D12CommandList::Transition(Buffer* buffer, ResourceState state) {
  if (!buffer)
    return;
  auto* d3d_buffer = static_cast<D3D12Buffer*>(buffer);
  // Upload and readback resources are created in the only state they may
  // hold, so their transitions are bookkeeping and nothing else.
  if (!d3d_buffer->barriers_allowed()) {
    buffer->set_state(state);
    return;
  }
  if (buffer->state() == state) {
    // Asking for a state a resource already holds only means something for
    // unordered access, where it is a request to make earlier writes visible.
    if (state == ResourceState::UnorderedAccess) {
      D3D12_RESOURCE_BARRIER barrier{};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.UAV.pResource = d3d_buffer->resource();
      list_->ResourceBarrier(1, &barrier);
    }
    return;
  }
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = d3d_buffer->resource();
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = ToD3D(buffer->state());
  barrier.Transition.StateAfter = ToD3D(state);
  list_->ResourceBarrier(1, &barrier);
  buffer->set_state(state);
}

void D3D12CommandList::Transition(Texture* texture, ResourceState state) {
  if (!texture)
    return;
  auto* d3d_texture = static_cast<D3D12Texture*>(texture);
  if (texture->state() == state) {
    if (state == ResourceState::UnorderedAccess) {
      D3D12_RESOURCE_BARRIER barrier{};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.UAV.pResource = d3d_texture->resource();
      list_->ResourceBarrier(1, &barrier);
    }
    return;
  }
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = d3d_texture->resource();
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = ToD3D(texture->state());
  barrier.Transition.StateAfter = ToD3D(state);
  list_->ResourceBarrier(1, &barrier);
  texture->set_state(state);
}

void D3D12CommandList::PushDebugLabel(const char*) {}
void D3D12CommandList::PopDebugLabel() {}

D3D12Device::~D3D12Device() {
  if (device_)
    WaitIdle();
  rtv_heap_.Destroy();
  dsv_heap_.Destroy();
  staging_view_heap_.Destroy();
  staging_sampler_heap_.Destroy();
  gpu_view_heap_.Destroy();
  gpu_sampler_heap_.Destroy();
  ReleaseCom(fence_);
  ReleaseCom(queue_);
  ReleaseCom(device_);
}

bool D3D12Device::Create() {
  if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                               IID_ID3D12Device, (void**)&device_))) {
    BASE_LOGI("rhid3d", "no D3D12 device");
    return false;
  }

  D3D12_COMMAND_QUEUE_DESC qd{};
  qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (FAILED(device_->CreateCommandQueue(&qd, IID_ID3D12CommandQueue,
                                         (void**)&queue_))) {
    return false;
  }
  if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_ID3D12Fence, (void**)&fence_))) {
    return false;
  }

  // Sized for a frame's worth of descriptors: the shader-visible heaps are
  // rings that wrap without waiting, so a frame that binds more than this
  // would overwrite descriptors still in flight.
  if (!rtv_heap_.Create(device_, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1024, false) ||
      !dsv_heap_.Create(device_, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 512, false) ||
      !staging_view_heap_.Create(
          device_, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16384, false) ||
      !staging_sampler_heap_.Create(device_, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                                    1024, false) ||
      !gpu_view_heap_.Create(device_, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                             16384, true) ||
      !gpu_sampler_heap_.Create(device_, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                                2048, true)) {
    BASE_LOGI("rhid3d", "descriptor heaps failed");
    return false;
  }

  caps_.adapter_name = "d3d12";
  caps_.max_texture_2d = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
  caps_.max_bound_textures =
      D3D12_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
  caps_.constant_buffer_offset_alignment =
      D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
  caps_.storage_buffer_offset_alignment = 16;
  caps_.texture_row_pitch_alignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
  caps_.texture_copy_offset_alignment =
      D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
  caps_.independent_blend = true;
  caps_.anisotropic_filtering = true;
  caps_.host_import = false;
  BASE_LOGI("rhid3d", "device up");
  return true;
}

std::unique_ptr<Buffer> D3D12Device::CreateBuffer(const BufferDesc& desc) {
  auto buffer = std::make_unique<D3D12Buffer>(*this, desc);
  if (!buffer->valid())
    return nullptr;
  return buffer;
}

std::unique_ptr<Texture> D3D12Device::CreateTexture(const TextureDesc& desc) {
  auto texture = std::make_unique<D3D12Texture>(*this, desc);
  if (!texture->valid())
    return nullptr;
  return texture;
}

std::unique_ptr<Sampler> D3D12Device::CreateSampler(const SamplerDesc& desc) {
  return std::make_unique<D3D12Sampler>(desc);
}

std::unique_ptr<ShaderModule> D3D12Device::CreateShader(
    const ShaderDesc& desc) {
  auto shader = std::make_unique<D3D12ShaderModule>(desc);
  if (!shader->valid())
    return nullptr;
  return shader;
}

std::unique_ptr<BindGroupLayout> D3D12Device::CreateBindGroupLayout(
    const BindGroupLayoutDesc& desc) {
  return std::make_unique<D3D12BindGroupLayout>(desc);
}

std::unique_ptr<BindGroup> D3D12Device::CreateBindGroup(
    const BindGroupDesc& desc) {
  if (!desc.layout)
    return nullptr;
  return std::make_unique<D3D12BindGroup>(*this, desc);
}

std::unique_ptr<Pipeline> D3D12Device::CreateGraphicsPipeline(
    const GraphicsPipelineDesc& desc) {
  auto pipeline = std::make_unique<D3D12Pipeline>(*this, desc);
  if (!pipeline->valid())
    return nullptr;
  return pipeline;
}

std::unique_ptr<Pipeline> D3D12Device::CreateComputePipeline(
    const ComputePipelineDesc& desc) {
  auto pipeline = std::make_unique<D3D12Pipeline>(*this, desc);
  if (!pipeline->valid())
    return nullptr;
  return pipeline;
}

std::unique_ptr<CommandList> D3D12Device::CreateCommandList() {
  ID3D12CommandAllocator* allocator = nullptr;
  if (FAILED(device_->CreateCommandAllocator(
          D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator,
          (void**)&allocator))) {
    return nullptr;
  }
  ID3D12GraphicsCommandList* list = nullptr;
  if (FAILED(device_->CreateCommandList(
          0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
          IID_ID3D12GraphicsCommandList, (void**)&list))) {
    ReleaseCom(allocator);
    return nullptr;
  }
  // Created open; the contract is that recording starts at Begin().
  list->Close();
  return std::make_unique<D3D12CommandList>(*this, allocator, list);
}

u64 D3D12Device::Submit(CommandList* list) {
  auto* d3d_list = static_cast<D3D12CommandList*>(list);
  ID3D12CommandList* lists[1] = {d3d_list->list()};
  queue_->ExecuteCommandLists(1, lists);
  const u64 value = ++submissions_;
  queue_->Signal(fence_, value);
  return value;
}

void D3D12Device::Wait(u64 submission) {
  // Polled rather than event-driven: the event handle is a Windows primitive
  // that the Linux build would have to emulate, and this is not a hot path.
  while (fence_->GetCompletedValue() < submission)
    std::this_thread::sleep_for(std::chrono::microseconds(50));
}

void D3D12Device::WaitIdle() {
  if (!submissions_)
    return;
  Wait(submissions_);
  gpu_view_heap_.Reset();
  gpu_sampler_heap_.Reset();
}

}  // namespace

std::unique_ptr<rhi::Device> CreateD3D12Device() {
  auto device = std::make_unique<D3D12Device>();
  if (!device->Create())
    return nullptr;
  return device;
}

}  // namespace gpu::d3d12

#endif  // DELTA_HAVE_D3D12
