#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The device abstraction the renderer runs on. Everything above it (render
 * targets keyed by guest address, the texture cache, the compute working set)
 * is one body of code shared by every backend; everything below it is one
 * graphics API.
 *
 * The model is Vulkan's, trimmed to what the renderer uses:
 *
 *   Objects are created by the Device and destroyed with Device::Destroy. The
 *   caller keeps them alive until the GPU work that uses them has retired.
 *
 *   Texture state is explicit. The caller names both sides of each transition
 *   (it already tracks them), which D3D12 and OpenGL can take or ignore.
 *
 *   Binding is by group: a Vulkan descriptor set, a D3D12 descriptor table,
 *   a list of OpenGL binding points. Dynamic buffer bindings take their offset
 *   at bind time.
 *
 *   Shaders are SPIR-V. Vulkan consumes it directly; OpenGL and D3D12 lower
 *   it to GLSL / HLSL when a pipeline is created (see rhi/shader_lowering.h).
 *
 * A backend is created by rhi::CreateDevice (rhi/backend.h).
 */

#include "base/arch.h"
#include "gpu/rhi/types.h"

namespace gpu::rhi {

class Object {
 public:
  virtual ~Object() = default;
};

class Buffer : public Object {
 public:
  const BufferDesc& desc() const { return desc_; }  // NOLINT: accessor
  // Host pointer for kUpload/kReadback memory (and imported host memory),
  // stable for the buffer's life. Null for kDevice memory.
  u8* mapped() const { return mapped_; }  // NOLINT: accessor
  // The shader-visible address, with kBufferAddress; 0 otherwise.
  u64 address() const { return address_; }  // NOLINT: accessor

 protected:
  BufferDesc desc_;
  u8* mapped_ = nullptr;
  u64 address_ = 0;
};

class Texture : public Object {
 public:
  const TextureDesc& desc() const { return desc_; }  // NOLINT: accessor

 protected:
  TextureDesc desc_;
};

class TextureView : public Object {
 public:
  const TextureViewDesc& desc() const { return desc_; }  // NOLINT: accessor
  Texture* texture() const { return texture_; }          // NOLINT: accessor

 protected:
  TextureViewDesc desc_;
  Texture* texture_ = nullptr;
};

class Sampler : public Object {};
class BindGroupLayout : public Object {
 public:
  const BindGroupLayoutDesc& desc() const { return desc_; }  // NOLINT

 protected:
  BindGroupLayoutDesc desc_;
};
class BindGroup : public Object {};
class PipelineLayout : public Object {
 public:
  const PipelineLayoutDesc& desc() const { return desc_; }  // NOLINT

 protected:
  PipelineLayoutDesc desc_;
};
class Pipeline : public Object {};
class TimestampPool : public Object {};

// One recorded sequence of GPU work: Begin, record, End, Device::Submit. It
// may be recorded again once that submission has retired.
class CommandList : public Object {
 public:
  virtual void Begin() = 0;
  virtual void End() = 0;

  // Draws are legal only inside a pass; copies, clears and dispatches only
  // outside one.
  virtual void BeginRenderPass(const RenderPassDesc& pass) = 0;
  virtual void EndRenderPass() = 0;

  virtual void SetPipeline(Pipeline* pipeline) = 0;
  // One offset per dynamic binding of the group, in binding order.
  virtual void SetBindGroup(u32 index,
                            BindGroup* group,
                            const u32* dynamic_offsets = nullptr,
                            u32 num_offsets = 0) = 0;
  // A group that lives only in this list: written and bound in one step.
  // Dynamic bindings in `layout` take their full offset from the writes.
  virtual void PushBindGroup(u32 index,
                             BindGroupLayout* layout,
                             const BindingWrite* writes,
                             u32 num_writes) = 0;
  virtual void SetPushConstants(u32 offset, u32 bytes, const void* data) = 0;

  virtual void SetVertexBuffers(u32 first,
                                u32 count,
                                Buffer* const* buffers,
                                const u64* offsets) = 0;
  virtual void SetIndexBuffer(Buffer* buffer, u64 offset, IndexType type) = 0;
  // Top-left origin. The renderer rasterises y-up by passing a negative
  // height with y at the bottom edge, as Vulkan does.
  virtual void SetViewport(float x,
                           float y,
                           float width,
                           float height,
                           float min_depth,
                           float max_depth) = 0;
  virtual void SetScissor(i32 x, i32 y, u32 width, u32 height) = 0;
  virtual void SetBlendConstants(const float rgba[4]) = 0;

  virtual void Draw(u32 vertex_count,
                    u32 instance_count,
                    u32 first_vertex,
                    u32 first_instance) = 0;
  virtual void DrawIndexed(u32 index_count,
                           u32 instance_count,
                           u32 first_index,
                           i32 vertex_offset,
                           u32 first_instance) = 0;
  virtual void DrawMeshTasks(u32 x, u32 y, u32 z) = 0;
  virtual void Dispatch(u32 x, u32 y, u32 z) = 0;
  // Workgroup ids start at (bx, by, bz) (Caps::dispatch_base; the pipeline
  // needs ComputePipelineDesc::dispatch_base).
  virtual void DispatchBase(u32 bx, u32 by, u32 bz, u32 x, u32 y, u32 z) = 0;
  // Inside a pass: clear a rectangle of one bound attachment. `attachment`
  // indexes the colours; ~0u means the depth/stencil attachment.
  virtual void ClearAttachment(u32 attachment,
                               const ClearColor& color,
                               float depth,
                               u8 stencil,
                               u8 aspect,
                               i32 x,
                               i32 y,
                               u32 width,
                               u32 height) = 0;

  virtual void CopyBuffer(Buffer* dst,
                          u64 dst_offset,
                          Buffer* src,
                          u64 src_offset,
                          u64 bytes) = 0;
  virtual void CopyBufferToTexture(Texture* dst,
                                   Buffer* src,
                                   const BufferTextureCopy* regions,
                                   u32 count) = 0;
  virtual void CopyTextureToBuffer(Buffer* dst,
                                   Texture* src,
                                   const BufferTextureCopy* regions,
                                   u32 count) = 0;
  // Same-size texel copy; the formats need only match in texel size.
  virtual void CopyTexture(Texture* dst,
                           const TextureRegion& dst_region,
                           Texture* src,
                           const TextureRegion& src_region) = 0;
  // Scaled, format-converting copy (Caps::texture_blit). src must be in
  // kCopySrc and dst in kCopyDst.
  virtual void BlitTexture(Texture* dst,
                           const TextureRegion& dst_region,
                           Texture* src,
                           const TextureRegion& src_region,
                           Filter filter) = 0;
  // The texture must be in kCopyDst (or kGeneral).
  virtual void ClearTexture(Texture* texture,
                            TextureState state,
                            const TextureRange& range,
                            const ClearColor& color) = 0;
  virtual void ClearDepthStencil(Texture* texture,
                                 TextureState state,
                                 const TextureRange& range,
                                 float depth,
                                 u8 stencil) = 0;
  // Fill with a repeating 32-bit value; the buffer needs kBufferCopyDst.
  virtual void FillBuffer(Buffer* buffer,
                          u64 offset,
                          u64 bytes,
                          u32 value) = 0;
  // Small inline write (<= 64 KiB), ordered with the list's other work.
  virtual void UpdateBuffer(Buffer* buffer,
                            u64 offset,
                            u64 bytes,
                            const void* data) = 0;

  // A global memory dependency (src_access -> dst_access, either may be 0)
  // plus texture state transitions, as one barrier.
  virtual void Barrier(u32 src_access,
                       u32 dst_access,
                       const TextureBarrier* textures = nullptr,
                       u32 num_textures = 0) = 0;

  virtual void ResetTimestamps(TimestampPool* pool, u32 first, u32 count) = 0;
  // `start`: stamp when the GPU begins the following work, rather than when
  // everything recorded before it has finished.
  virtual void WriteTimestamp(TimestampPool* pool, u32 index, bool start) = 0;

  virtual void PushLabel(const char* label) = 0;
  virtual void PopLabel() = 0;
  virtual void InsertLabel(const char* label) = 0;
  // A progress marker the backend may use to locate a device loss.
  virtual void Checkpoint(u64 /*marker*/) {}
};

class Device {
 public:
  virtual ~Device() = default;

  virtual const Caps& caps() const = 0;  // NOLINT: accessor
  Backend backend() const { return caps().backend; }  // NOLINT: accessor

  virtual Buffer* CreateBuffer(const BufferDesc& desc) = 0;
  virtual Texture* CreateTexture(const TextureDesc& desc) = 0;
  virtual TextureView* CreateView(Texture* texture,
                                  const TextureViewDesc& desc) = 0;
  virtual Sampler* CreateSampler(const SamplerDesc& desc) = 0;
  virtual BindGroupLayout* CreateBindGroupLayout(
      const BindGroupLayoutDesc& desc) = 0;
  virtual BindGroup* CreateBindGroup(const BindGroupDesc& desc) = 0;
  // Rewrite bindings of an existing group. The group must not be in use by
  // GPU work that has not retired.
  virtual void UpdateBindGroup(BindGroup* group,
                               const BindingWrite* writes,
                               u32 count) = 0;
  virtual PipelineLayout* CreatePipelineLayout(
      const PipelineLayoutDesc& desc) = 0;
  // Null when the shaders or state cannot be served. Safe to call from any
  // thread, concurrently with recording.
  virtual Pipeline* CreateGraphicsPipeline(
      const GraphicsPipelineDesc& desc) = 0;
  virtual Pipeline* CreateComputePipeline(const ComputePipelineDesc& desc) = 0;
  virtual TimestampPool* CreateTimestampPool(u32 count) = 0;
  virtual CommandList* CreateCommandList() = 0;

  // Destroys any object created above. Null is ignored.
  virtual void Destroy(Object* object) = 0;
  // A name for capture tools. Ignored unless Caps::debug_labels.
  virtual void SetName(Object* /*object*/, const char* /*name*/) {}

  // True when `format` can be used with every TextureUsage bit in `usage`.
  virtual bool SupportsFormat(Format format, u32 usage) const = 0;

  // Queue the lists in order. The returned id retires once the GPU has
  // finished them; ids increase and retire in order.
  virtual u64 Submit(CommandList* const* lists, u32 count) = 0;
  u64 Submit(CommandList* list) { return Submit(&list, 1); }
  virtual bool IsComplete(u64 submission) = 0;
  // False on timeout or device loss.
  virtual bool Wait(u64 submission, u64 timeout_ns = ~0ull) = 0;
  virtual void WaitIdle() = 0;
  // The last id submitted.
  virtual u64 LastSubmission() const = 0;

  virtual bool ReadTimestamps(TimestampPool* pool,
                              u32 first,
                              u32 count,
                              u64* out) = 0;

  // Once a frame: lets the backend persist caches and trim pools.
  virtual void Maintain() {}
  // After a failed Wait: log what the backend knows about the device loss.
  virtual void ReportDeviceLoss() {}
};

}  // namespace gpu::rhi
