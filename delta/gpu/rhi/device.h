#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The device abstraction: what a graphics API has to provide for the renderer
 * to run on it. Everything above this seam (render targets keyed by guest
 * address, the texture cache, format decode, the compute working set) is one
 * body of code shared by every backend; everything below it is one backend's
 * business.
 *
 * The surface is sized by what the renderer actually calls, not by what an API
 * offers. Its shape follows three rules:
 *
 *   Resource state is explicit but tracked. Every buffer and texture knows the
 *   state it is in at the point the recording has reached, and Transition() to
 *   the state it already holds is free. Command lists must therefore be
 *   submitted in the order they were recorded.
 *
 *   A render pass carries its load/store ops. Vulkan needs them and D3D12 can
 *   use or ignore them, so expressing a clear as loadOp rather than as a clear
 *   command keeps the fast path available on both.
 *
 *   Binding is by group. A group is a Vulkan descriptor set and a D3D12
 *   descriptor table, bound by index; buffer bindings may take a per-draw
 *   offset, which Vulkan serves with dynamic descriptors and D3D12 by writing
 *   a fresh descriptor into its ring heap.
 *
 * A backend is obtained from its own header (vulkan/vk_rhi.h, d3d12/d3d12_rhi.h,
 * rhi/null_device.h); this header names none of them.
 */

#include <memory>

#include "base/arch.h"
#include "gpu/rhi/types.h"

namespace gpu::rhi {

// Resources. Each is created by a Device, owned by the caller, and destroyed
// before the Device is. A resource may not outlive work still referencing it:
// wait for the submission first.

class Buffer {
 public:
  virtual ~Buffer() = default;

  const BufferDesc& desc() const { return desc_; }  // NOLINT: accessor
  // The state the recording has left this buffer in.
  ResourceState state() const { return state_; }  // NOLINT: accessor
  void set_state(ResourceState s) { state_ = s; }

  // Host pointer to the buffer's memory, stable for the buffer's life. Only an
  // Upload or Readback buffer has one; Device memory returns null.
  virtual void* Map() = 0;

 protected:
  BufferDesc desc_;
  ResourceState state_ = ResourceState::Common;
};

class Texture {
 public:
  virtual ~Texture() = default;

  const TextureDesc& desc() const { return desc_; }  // NOLINT: accessor
  ResourceState state() const { return state_; }     // NOLINT: accessor
  void set_state(ResourceState s) { state_ = s; }

 protected:
  TextureDesc desc_;
  ResourceState state_ = ResourceState::Undefined;
};

class Sampler {
 public:
  virtual ~Sampler() = default;
};

class ShaderModule {
 public:
  virtual ~ShaderModule() = default;
};

class BindGroupLayout {
 public:
  virtual ~BindGroupLayout() = default;

  const BindGroupLayoutDesc& desc() const { return desc_; }  // NOLINT

 protected:
  BindGroupLayoutDesc desc_;
};

// A group's descriptors are written once, when it is created, and every draw
// that binds it reuses them. A binding whose contents change per draw is
// therefore either a dynamic buffer (an offset supplied at bind time) or a
// separate group.
class BindGroup {
 public:
  virtual ~BindGroup() = default;
};

class Pipeline {
 public:
  virtual ~Pipeline() = default;
};

// One recorded sequence of GPU work. Record between Begin() and End(), then
// hand it to Device::Submit. A list may be re-recorded after the submission
// that used it has retired.
class CommandList {
 public:
  virtual ~CommandList() = default;

  virtual void Begin() = 0;
  virtual void End() = 0;

  // Rendering. Draw commands are legal only inside a render pass, copies and
  // dispatches only outside one.
  //
  // BeginRenderPass transitions its own attachments (a pass cannot run with
  // one in any other state, so requiring the caller to do it would add no
  // expressiveness, only a way to get it wrong). Every other transition is the
  // caller's.
  virtual void BeginRenderPass(const RenderPassDesc& pass) = 0;
  virtual void EndRenderPass() = 0;

  virtual void SetPipeline(Pipeline* pipeline) = 0;
  // `offsets` supplies one value per dynamic binding of the group, in binding
  // order. Passing none is legal only for a group with no dynamic bindings.
  virtual void SetBindGroup(u32 index,
                            BindGroup* group,
                            const u32* offsets = nullptr,
                            u32 num_offsets = 0) = 0;
  virtual void SetPushConstants(u32 offset, u32 bytes, const void* data) = 0;

  // Must follow the SetPipeline of the pipeline that will consume it: D3D12
  // carries the vertex stride in the buffer binding rather than in the
  // pipeline, so the binding is only complete once the strides are known.
  virtual void SetVertexBuffer(u32 slot, Buffer* buffer, u64 offset) = 0;
  virtual void SetIndexBuffer(Buffer* buffer, u64 offset, IndexType type) = 0;
  // `x`, `y` are the rectangle's top-left corner in pixels. Clip space has +Y
  // up and depth in [0, 1]: D3D12's convention, which the Vulkan backend
  // reaches with a negative-height viewport. Face winding is therefore the
  // same on both, and geometry needs no per-backend flip.
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
  virtual void Dispatch(u32 groups_x, u32 groups_y, u32 groups_z) = 0;

  // Transfers. Row pitches must satisfy DeviceCaps::texture_row_pitch_alignment
  // and buffer offsets DeviceCaps::texture_copy_offset_alignment; D3D12
  // rejects anything else outright.
  virtual void CopyBuffer(Buffer* dst,
                          u64 dst_offset,
                          Buffer* src,
                          u64 src_offset,
                          u64 bytes) = 0;
  virtual void CopyBufferToTexture(Texture* dst,
                                   const TextureRegion& region,
                                   Buffer* src,
                                   u64 src_offset,
                                   u32 src_row_pitch) = 0;
  virtual void CopyTextureToBuffer(Buffer* dst,
                                   u64 dst_offset,
                                   u32 dst_row_pitch,
                                   Texture* src,
                                   const TextureRegion& region) = 0;
  // Fills with a repeating 32-bit value. The buffer must carry both
  // kBufferStorage (D3D12 has no fill command and clears through an unordered
  // access view) and kBufferCopyDst (Vulkan fills through a transfer op).
  virtual void FillBuffer(Buffer* buffer,
                          u64 offset,
                          u64 bytes,
                          u32 value) = 0;

  // Make the resource usable as `state`. Asking for the state it already holds
  // does nothing, except for UnorderedAccess, where it is the request to make
  // earlier writes through the same view visible to the next access.
  virtual void Transition(Buffer* buffer, ResourceState state) = 0;
  virtual void Transition(Texture* texture, ResourceState state) = 0;

  virtual void PushDebugLabel(const char* label) = 0;
  virtual void PopDebugLabel() = 0;
};

class Device {
 public:
  virtual ~Device() = default;

  // Static name of the API this device speaks ("vulkan", "d3d12", "null").
  virtual const char* backend_name() const = 0;  // NOLINT: accessor
  virtual const DeviceCaps& caps() const = 0;    // NOLINT: accessor

  virtual std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) = 0;
  virtual std::unique_ptr<Texture> CreateTexture(const TextureDesc& desc) = 0;
  virtual std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) = 0;
  virtual std::unique_ptr<ShaderModule> CreateShader(
      const ShaderDesc& desc) = 0;
  virtual std::unique_ptr<BindGroupLayout> CreateBindGroupLayout(
      const BindGroupLayoutDesc& desc) = 0;
  virtual std::unique_ptr<BindGroup> CreateBindGroup(
      const BindGroupDesc& desc) = 0;
  virtual std::unique_ptr<Pipeline> CreateGraphicsPipeline(
      const GraphicsPipelineDesc& desc) = 0;
  virtual std::unique_ptr<Pipeline> CreateComputePipeline(
      const ComputePipelineDesc& desc) = 0;
  virtual std::unique_ptr<CommandList> CreateCommandList() = 0;

  // Queue the list. The returned id retires once the GPU has finished it;
  // ids increase, and waiting on one waits for every earlier submission too.
  virtual u64 Submit(CommandList* list) = 0;
  virtual void Wait(u64 submission) = 0;
  virtual void WaitIdle() = 0;
};

}  // namespace gpu::rhi
