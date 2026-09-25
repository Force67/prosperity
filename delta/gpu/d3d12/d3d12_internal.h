#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The D3D12 objects behind the rhi classes. Private to d3d12_*.cc.
 *
 * Binding: every PipelineLayout is one root signature,
 *   [push constants (root constants, or a root CBV when they do not fit)]
 *   [raster: y sign] [draw: base vertex/instance] [dispatch: group count]
 *   per group: [CBV/SRV/UAV table] [sampler table] [root CBV per dynamic UBO]
 * A group's descriptors live in a CPU-only heap and are copied into a
 * shader-visible ring at SetBindGroup; the ring is handed out in chunks a
 * command list keeps until it records again (its previous submission has
 * retired by then). Sampler tables are deduplicated by content into the
 * 2048-entry shader-visible sampler heap.
 */

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/arch.h"
#include "gpu/d3d12/d3d12_rhi.h"
#include "gpu/d3d12/d3d12_shader.h"
#include "gpu/rhi/device.h"

#include "gpu/d3d12/d3d12_api.h"

namespace gpu::d3d12::impl {

class D3D12Device;

template <typename T>
void SafeRelease(T*& object) {
  if (object) {
    object->Release();
    object = nullptr;
  }
}

// Format mapping (d3d12_format.cc).
struct DxgiInfo {
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;    // views, RTVs, DSVs
  DXGI_FORMAT typeless = DXGI_FORMAT_UNKNOWN;  // resources that alias views
  DXGI_FORMAT srv = DXGI_FORMAT_UNKNOWN;       // sampling (depth: the plane)
  DXGI_FORMAT stencil_srv = DXGI_FORMAT_UNKNOWN;
  // A vertex format DXGI lacks, fed as its integer twin and converted in the
  // shader (1 = unsigned, 2 = signed), or with red and blue swapped.
  u8 scaled = 0;
  bool swap_rb = false;
  bool texture = true;  // usable for textures at all
};
const DxgiInfo& Dxgi(rhi::Format format);
DXGI_FORMAT VertexFormat(rhi::Format format);

// Contiguous ranges of CPU-only descriptors.
struct CpuRange {
  D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
  u32 page = ~0u;
  u32 offset = 0;
  u32 count = 0;
  bool valid() const { return page != ~0u; }  // NOLINT: accessor
};

class CpuDescriptorPool {
 public:
  void Init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type,
            u32 page_size);
  void Shutdown();
  bool Allocate(u32 count, CpuRange* out);
  void Free(CpuRange& range);
  D3D12_CPU_DESCRIPTOR_HANDLE At(const CpuRange& r, u32 index) const {
    return {r.cpu.ptr + static_cast<size_t>(index) * increment_};
  }
  u32 increment() const { return increment_; }  // NOLINT: accessor

 private:
  struct Page {
    ID3D12DescriptorHeap* heap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE start{};
    std::map<u32, u32> free;  // offset -> count
  };
  ID3D12Device* device_ = nullptr;
  D3D12_DESCRIPTOR_HEAP_TYPE type_{};
  u32 page_size_ = 0;
  u32 increment_ = 0;
  std::mutex mutex_;
  std::vector<Page> pages_;
};

class D3D12Buffer final : public rhi::Buffer {
 public:
  explicit D3D12Buffer(const rhi::BufferDesc& desc) { desc_ = desc; }
  void set_mapped(u8* p) { mapped_ = p; }
  ID3D12Resource* resource = nullptr;
  D3D12_GPU_VIRTUAL_ADDRESS va = 0;
  u64 alloc_size = 0;
  // Upload and readback heaps never leave their initial state.
  bool fixed_state = false;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

class D3D12Texture final : public rhi::Texture {
 public:
  explicit D3D12Texture(const rhi::TextureDesc& desc) { desc_ = desc; }
  u32 array_size() const {  // NOLINT: accessor
    return desc_.dim == rhi::TextureDim::k3D ? 1u : desc_.layers;
  }
  u32 Subresource(u32 mip, u32 layer, u32 plane) const {
    return mip + layer * desc_.mips + plane * desc_.mips * array_size();
  }
  ID3D12Resource* resource = nullptr;
  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
  u32 planes = 1;
  // The last state each subresource was moved into while recording: only
  // consulted for a kUndefined "before", the one side the caller cannot name.
  std::vector<D3D12_RESOURCE_STATES> states;
  // Per (mip, layer) attachment descriptors for clears and blits.
  std::mutex view_mutex;
  std::unordered_map<u32, CpuRange> rtvs, dsvs;
};

class D3D12View final : public rhi::TextureView {
 public:
  D3D12View(rhi::Texture* texture, const rhi::TextureViewDesc& desc) {
    texture_ = texture;
    desc_ = desc;
  }
  CpuRange srv, uav, rtv;
  CpuRange dsv[4];  // by read-only depth | read-only stencil << 1
};

class D3D12Sampler final : public rhi::Sampler {
 public:
  u32 id = 0;  // index into the device's deduplicated sampler table
};

struct GroupEntry {
  u32 binding = 0;
  rhi::BindingType type = rhi::BindingType::kSampledTexture;
  bool read_only = false;
  u32 slot = 0;              // in the CBV/SRV/UAV table
  u32 sampler_slot = ~0u;    // in the sampler table
  u32 dynamic_index = ~0u;   // in SetBindGroup's offsets
};

class D3D12BindGroupLayout final : public rhi::BindGroupLayout {
 public:
  explicit D3D12BindGroupLayout(const rhi::BindGroupLayoutDesc& desc) {
    desc_ = desc;
  }
  const GroupEntry* Find(u32 binding) const;
  std::vector<GroupEntry> entries;  // desc().bindings order
  std::vector<u32> dynamic_entries;  // entry index per dynamic index
  u32 table_size = 0;
  u32 sampler_count = 0;
};

// What a bound group needs from the buffers it references.
struct BufferUse {
  class D3D12Buffer* buffer = nullptr;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

class D3D12BindGroup final : public rhi::BindGroup {
 public:
  D3D12BindGroupLayout* layout = nullptr;
  CpuRange table;  // CPU-only copy of every table slot
  std::vector<u32> samplers;  // sampler ids by sampler slot
  // Per dynamic index: the buffer and the write's own offset/range.
  std::vector<rhi::BindingWrite> dynamic;
  std::vector<BufferUse> uses;
  // Cached shader-visible sampler table and the cache generation it is from.
  u32 sampler_table = ~0u;
  u64 sampler_generation = 0;
};

struct RootGroup {
  i32 table = -1;
  i32 sampler_table = -1;
  std::vector<i32> root_cbv;  // per dynamic index; -1 = in the table
};

class D3D12PipelineLayout final : public rhi::PipelineLayout {
 public:
  explicit D3D12PipelineLayout(const rhi::PipelineLayoutDesc& desc) {
    desc_ = desc;
  }
  ID3D12RootSignature* root = nullptr;
  i32 push_constants = -1;  // root constants
  i32 push_cbv = -1;        // push constants from upload memory
  u32 push_dwords = 0;
  i32 raster = -1, draw = -1, dispatch = -1;
  std::vector<RootGroup> groups;
  std::vector<std::pair<u32, u32>> read_only_storage;  // (set, binding)
};

class D3D12Pipeline final : public rhi::Pipeline {
 public:
  ID3D12PipelineState* pso = nullptr;
  // Primitive restart with 32-bit indices (pso takes 16-bit).
  ID3D12PipelineState* pso_restart32 = nullptr;
  D3D12PipelineLayout* layout = nullptr;
  bool compute = false;
  D3D_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
  bool fan = false;  // drawn as an indexed list built at record time
  u32 strides[16] = {};
  u8 stencil_ref = 0;
  bool uses_raster = false;
  bool uses_draw_params = false;
  bool uses_workgroup_count = false;
};

class D3D12TimestampPool final : public rhi::TimestampPool {
 public:
  ID3D12QueryHeap* heap = nullptr;
  ID3D12Resource* readback = nullptr;
  const u64* results = nullptr;
  u32 count = 0;
};

// A chunk of shader-visible descriptors owned by one command list.
struct RingChunk {
  u32 first = 0;
  u32 used = 0;
};

struct UploadBlock {
  ID3D12Resource* resource = nullptr;
  u8* mapped = nullptr;
  D3D12_GPU_VIRTUAL_ADDRESS va = 0;
  u64 size = 0;
  u64 used = 0;
};

struct UploadAlloc {
  ID3D12Resource* resource = nullptr;
  u64 offset = 0;
  u8* cpu = nullptr;
  D3D12_GPU_VIRTUAL_ADDRESS va = 0;
};

class D3D12CommandList final : public rhi::CommandList {
 public:
  explicit D3D12CommandList(D3D12Device& device) : device_(device) {}
  ~D3D12CommandList() override;
  bool Init();

  ID3D12CommandAllocator* allocator = nullptr;
  ID3D12GraphicsCommandList* cmd = nullptr;

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
  void ResetTimestamps(rhi::TimestampPool* pool,
                       u32 first,
                       u32 count) override;
  void WriteTimestamp(rhi::TimestampPool* pool, u32 index) override;
  void PushLabel(const char* label) override;
  void PopLabel() override;
  void InsertLabel(const char* label) override;

 private:
  struct BoundGroup {
    D3D12BindGroupLayout* layout = nullptr;
    D3D12_GPU_DESCRIPTOR_HANDLE table{};
    D3D12_GPU_DESCRIPTOR_HANDLE samplers{};
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> root_cbv;
    std::vector<BufferUse> uses;
    bool dirty = false;
    u64 validated_epoch = ~0ull;
  };

  // Shader-visible CBV/SRV/UAV slots valid until this list records again.
  bool AllocateDescriptors(u32 count,
                           D3D12_CPU_DESCRIPTOR_HANDLE* cpu,
                           D3D12_GPU_DESCRIPTOR_HANDLE* gpu);
  UploadAlloc AllocateUpload(u64 bytes, u64 alignment);
  // A DEFAULT-heap region for repacking copies, in COPY_DEST.
  UploadAlloc AllocateScratch(u64 bytes);
  void ReleaseTransient();

  // Buffer states are per list: every buffer decays to COMMON between
  // ExecuteCommandLists calls, and Submit runs each list in its own.
  void Require(D3D12Buffer* buffer, D3D12_RESOURCE_STATES state);
  void RequireScratch(ID3D12Resource* scratch, D3D12_RESOURCE_STATES state);
  void Transition(ID3D12Resource* resource,
                  u32 subresource,
                  D3D12_RESOURCE_STATES before,
                  D3D12_RESOURCE_STATES after);
  void TransitionTexture(D3D12Texture* texture,
                         const rhi::TextureRange& range,
                         D3D12_RESOURCE_STATES after,
                         const D3D12_RESOURCE_STATES* before = nullptr);
  void FlushBarriers();

  void BindTable(u32 index,
                 D3D12BindGroupLayout* layout,
                 D3D12_GPU_DESCRIPTOR_HANDLE table,
                 D3D12_GPU_DESCRIPTOR_HANDLE samplers,
                 std::vector<D3D12_GPU_VIRTUAL_ADDRESS> root_cbv,
                 std::vector<BufferUse> uses);
  void WriteBufferView(D3D12_CPU_DESCRIPTOR_HANDLE dst,
                       const GroupEntry& entry,
                       D3D12Buffer* buffer,
                       u64 offset,
                       u64 range);
  void FlushState(bool compute);
  void ApplyVertexBuffers();

  void CopyBufferTexture(D3D12Texture* texture,
                         D3D12Buffer* buffer,
                         const rhi::BufferTextureCopy& region,
                         bool to_texture);
  CpuRange AttachmentView(D3D12Texture* texture, u32 mip, u32 layer,
                          bool depth);
  void ClearByCopy(D3D12Texture* texture,
                   const rhi::TextureRange& range,
                   const rhi::ClearColor& color);

  D3D12Device& device_;
  bool recording_ = false;

  std::vector<RingChunk> chunks_;
  std::vector<UploadBlock> upload_;
  size_t upload_block_ = 0;
  std::vector<UploadBlock> scratch_;
  ID3D12DescriptorHeap* cpu_scratch_ = nullptr;  // for UAV clears
  u32 cpu_scratch_next_ = 0;

  std::vector<D3D12_RESOURCE_BARRIER> barriers_;
  std::unordered_map<D3D12Buffer*, D3D12_RESOURCE_STATES> buffer_states_;
  std::unordered_map<ID3D12Resource*, D3D12_RESOURCE_STATES> scratch_states_;
  u64 transition_epoch_ = 0;

  D3D12Pipeline* pipeline_ = nullptr;
  ID3D12RootSignature* root_ = nullptr;  // as last set on the list
  bool root_compute_ = false;
  ID3D12PipelineState* pso_ = nullptr;
  bool state_lost_ = false;  // an internal pass replaced the pipeline state
  BoundGroup groups_[8];
  u32 push_[64] = {};
  bool push_dirty_ = false;
  struct VertexBinding {
    D3D12Buffer* buffer = nullptr;
    u64 offset = 0;
  } vertex_[16];
  u32 vertex_count_ = 0;
  bool vertex_dirty_ = false;
  D3D12Buffer* index_buffer_ = nullptr;
  u64 index_offset_ = 0;
  rhi::IndexType index_type_ = rhi::IndexType::kUint16;
  bool index_dirty_ = false;
  float y_sign_ = 1.0f;
  D3D12_VIEWPORT viewport_{};
  D3D12_RECT scissor_{};
  float blend_[4] = {};

  u32 pass_color_count_ = 0;
  D3D12_CPU_DESCRIPTOR_HANDLE pass_rtvs_[8]{};
  D3D12_CPU_DESCRIPTOR_HANDLE pass_dsv_{};
  D3D12_CPU_DESCRIPTOR_HANDLE pass_dsv_clear_{};  // writable, for clears
  bool pass_has_dsv_ = false;
};

class D3D12Device final : public rhi::Device {
 public:
  ~D3D12Device() override;
  bool Init(const D3D12Options& options);

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
  u64 Submit(rhi::CommandList* const* lists, u32 count) override;
  bool IsComplete(u64 submission) override;
  bool Wait(u64 submission, u64 timeout_ns) override;
  void WaitIdle() override;
  u64 LastSubmission() const override { return submitted_; }
  bool ReadTimestamps(rhi::TimestampPool* pool,
                      u32 first,
                      u32 count,
                      u64* out) override;
  void ReportDeviceLoss() override;

  // Buffer views into a CPU descriptor (d3d12_device.cc).
  void WriteCbv(D3D12_CPU_DESCRIPTOR_HANDLE dst, D3D12Buffer* buffer,
                u64 offset, u64 range);
  void WriteRawView(D3D12_CPU_DESCRIPTOR_HANDLE dst, D3D12Buffer* buffer,
                    u64 offset, u64 range, bool uav);
  void WriteNull(D3D12_CPU_DESCRIPTOR_HANDLE dst, rhi::BindingType type,
                 bool read_only);

  // Shader-visible descriptor chunks for command lists.
  bool AcquireChunk(RingChunk* out);
  void ReleaseChunk(const RingChunk& chunk);
  D3D12_CPU_DESCRIPTOR_HANDLE RingCpu(u32 index) const {
    return {ring_cpu_.ptr + static_cast<size_t>(index) * view_increment};
  }
  D3D12_GPU_DESCRIPTOR_HANDLE RingGpu(u32 index) const {
    return {ring_gpu_.ptr + static_cast<u64>(index) * view_increment};
  }
  // A shader-visible sampler table holding these sampler ids.
  D3D12_GPU_DESCRIPTOR_HANDLE SamplerTable(const std::vector<u32>& ids,
                                           u32* cached,
                                           u64* generation);

  ID3D12Resource* CreateBufferResource(u64 size, D3D12_HEAP_TYPE heap,
                                       D3D12_RESOURCE_FLAGS flags,
                                       D3D12_RESOURCE_STATES state);
  D3D12_RESOURCE_STATES TextureState(const D3D12Texture* texture,
                                     rhi::TextureState state) const;

  // The internal blit pipeline for one render-target format.
  ID3D12PipelineState* BlitPipeline(DXGI_FORMAT format);
  ID3D12RootSignature* blit_root() const { return blit_root_; }  // NOLINT

  ID3D12Device* device = nullptr;
  ID3D12CommandQueue* queue = nullptr;
  ID3D12DescriptorHeap* ring_heap = nullptr;
  ID3D12DescriptorHeap* sampler_heap = nullptr;
  CpuDescriptorPool views;     // CBV/SRV/UAV
  CpuDescriptorPool rtvs;
  CpuDescriptorPool dsvs;
  CpuDescriptorPool samplers;  // CPU copies of each unique sampler
  u32 view_increment = 0;
  u32 sampler_increment = 0;
  bool debug_labels = false;

 private:
  ID3D12RootSignature* SerializeRoot(const D3D12_ROOT_SIGNATURE_DESC& desc);
  // SPIR-V -> DXIL, cached by module and options.
  bool CompileStage(const rhi::ShaderCode& code,
                    const LowerOptions& options,
                    std::vector<u8>* dxil,
                    LoweredShader* info);
  bool InitBlit();

  rhi::Caps caps_;
  std::string device_name_;
  u32 shader_model_ = 60;
  ID3D12Fence* fence_ = nullptr;
  std::mutex queue_mutex_;
  u64 submitted_ = 0;

  D3D12_CPU_DESCRIPTOR_HANDLE ring_cpu_{};
  D3D12_GPU_DESCRIPTOR_HANDLE ring_gpu_{};
  u32 ring_size_ = 0;
  std::mutex chunk_mutex_;
  std::vector<u32> free_chunks_;

  std::mutex sampler_mutex_;
  std::map<std::string, u32> sampler_ids_;  // packed desc -> id
  std::vector<CpuRange> sampler_cpu_;       // by id
  std::map<std::vector<u32>, u32> sampler_tables_;
  u32 sampler_heap_used_ = 0;
  u64 sampler_generation_ = 1;

  std::mutex shader_mutex_;
  struct ShaderEntry {
    std::vector<u8> dxil;
    LoweredShader info;
    bool ok = false;
  };
  std::unordered_map<u64, ShaderEntry> shaders_;

  ID3D12RootSignature* blit_root_ = nullptr;
  std::vector<u8> blit_vs_, blit_ps_;
  std::mutex blit_mutex_;
  std::unordered_map<u32, ID3D12PipelineState*> blit_psos_;
  bool fault_reported_ = false;
};

inline D3D12Buffer* Buf(rhi::Buffer* b) {
  return static_cast<D3D12Buffer*>(b);
}
inline D3D12Texture* Tex(rhi::Texture* t) {
  return static_cast<D3D12Texture*>(t);
}
inline D3D12View* View(rhi::TextureView* v) {
  return static_cast<D3D12View*>(v);
}

constexpr u32 kChunkSize = 4096;
constexpr u32 kSamplerHeapSize = 2048;

}  // namespace gpu::d3d12::impl
