#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <type_traits>
#include "base/arch.h"
#include "gpu/rhi/types.h"

#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"

namespace gpu::rhi {

class Object {
 public:
  virtual ~Object() = default;

 private:
  friend class Device;
  size_t tracked_slot_ = ~size_t(0);
};

class Buffer : public Object {
 public:
  const BufferDesc& desc() const { return desc_; }  // NOLINT: accessor
  u8* mapped() const { return mapped_; }            // NOLINT: accessor
  u64 address() const { return address_; }          // NOLINT: accessor

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

inline u64 g_list_gpu_ns[4];
inline u64 g_list_gpu_n[4];
constexpr u32 kProfCats = 17;
inline u64 g_prof_cat_ns[kProfCats];

class CommandList : public Object {
 public:
  virtual void Begin() = 0;
  virtual void End() = 0;
  void SetProfileTag(u8 tag) { profile_tag_ = tag; }
  virtual void ProfileBegin(u32) {}
  virtual void ProfileEnd() {}

 protected:
  u8 profile_tag_ = 0;

 public:
  virtual void BeginRenderPass(const RenderPassDesc& pass) = 0;
  virtual void EndRenderPass() = 0;

  virtual void SetPipeline(Pipeline* pipeline) = 0;
  virtual void SetBindGroup(u32 index,
                            BindGroup* group,
                            const u32* dynamic_offsets = nullptr,
                            u32 num_offsets = 0) = 0;
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
  virtual void DrawMeshTasksIndirect(Buffer*, u64) {}
  virtual void Dispatch(u32 x, u32 y, u32 z) = 0;
  virtual void DispatchBase(u32 bx, u32 by, u32 bz, u32 x, u32 y, u32 z) = 0;
  virtual void DispatchIndirect(Buffer*, u64) {}
  virtual void DrawIndexedIndirect(Buffer*, u64) {}
  virtual void DrawIndirect(Buffer*, u64) {}
  virtual void BeginConditional(Buffer*, u64, bool) {}
  virtual void EndConditional() {}
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
  virtual void CopyTexture(Texture* dst,
                           const TextureRegion& dst_region,
                           Texture* src,
                           const TextureRegion& src_region) = 0;
  virtual void BlitTexture(Texture* dst,
                           const TextureRegion& dst_region,
                           Texture* src,
                           const TextureRegion& src_region,
                           Filter filter) = 0;
  virtual void ClearTexture(Texture* texture,
                            TextureState state,
                            const TextureRange& range,
                            const ClearColor& color) = 0;
  virtual void ClearDepthStencil(Texture* texture,
                                 TextureState state,
                                 const TextureRange& range,
                                 float depth,
                                 u8 stencil) = 0;
  virtual void FillBuffer(Buffer* buffer, u64 offset, u64 bytes, u32 value) = 0;
  virtual void UpdateBuffer(Buffer* buffer,
                            u64 offset,
                            u64 bytes,
                            const void* data) = 0;

  virtual void Barrier(u32 src_access,
                       u32 dst_access,
                       const TextureBarrier* textures = nullptr,
                       u32 num_textures = 0) = 0;

  virtual void ResetTimestamps(TimestampPool* pool, u32 first, u32 count) = 0;
  virtual void WriteTimestamp(TimestampPool* pool, u32 index, bool start) = 0;

  virtual void PushLabel(const char* label) = 0;
  virtual void PopLabel() = 0;
  virtual void InsertLabel(const char* label) = 0;
  virtual void Checkpoint(u64) {}
};

class Device {
 public:
  void Track(Object* object) {
    base::LockGuard lock(objects_mutex_);
    object->tracked_slot_ = objects_.size();
    objects_.push_back(object);
  }
  void Forget(Object* object) {
    base::LockGuard lock(objects_mutex_);
    if (!object || object->tracked_slot_ >= objects_.size() ||
        objects_[object->tracked_slot_] != object)
      return;
    Object* last = objects_.back();
    objects_[object->tracked_slot_] = last;
    last->tracked_slot_ = object->tracked_slot_;
    objects_.pop_back();
    object->tracked_slot_ = ~size_t(0);
  }
  void ReleaseObjects() {
    for (;;) {
      Object* object;
      {
        base::LockGuard lock(objects_mutex_);
        if (objects_.empty())
          break;
        object = objects_.back();
      }
      Destroy(object);
    }
  }
  size_t LiveObjects() const {
    base::LockGuard lock(objects_mutex_);
    return objects_.size();
  }

  virtual ~Device() = default;

  virtual const Caps& caps() const = 0;               // NOLINT: accessor
  Backend backend() const { return caps().backend; }  // NOLINT: accessor

  virtual Buffer* CreateBuffer(const BufferDesc& desc) = 0;
  virtual Texture* CreateTexture(const TextureDesc& desc) = 0;
  virtual TextureView* CreateView(Texture* texture,
                                  const TextureViewDesc& desc) = 0;
  virtual Sampler* CreateSampler(const SamplerDesc& desc) = 0;
  virtual BindGroupLayout* CreateBindGroupLayout(
      const BindGroupLayoutDesc& desc) = 0;
  virtual BindGroup* CreateBindGroup(const BindGroupDesc& desc) = 0;
  virtual void UpdateBindGroup(BindGroup* group,
                               const BindingWrite* writes,
                               u32 count) = 0;
  virtual PipelineLayout* CreatePipelineLayout(
      const PipelineLayoutDesc& desc) = 0;
  virtual Pipeline* CreateGraphicsPipeline(
      const GraphicsPipelineDesc& desc) = 0;
  virtual Pipeline* CreateComputePipeline(const ComputePipelineDesc& desc) = 0;
  virtual TimestampPool* CreateTimestampPool(u32 count) = 0;
  virtual CommandList* CreateCommandList() = 0;

  virtual void Destroy(Object* object) = 0;
  virtual bool CommitSparse(Buffer*, u64, u64) { return false; }
  virtual bool CopyAfter(Buffer*, u64, Buffer*, u64, u64, u64) { return false; }
  virtual int ExportFd(Texture*, u64*) { return -1; }
  virtual void SetName(Object*, const char*) {}

  virtual bool SupportsFormat(Format format, u32 usage) const = 0;
  virtual bool SupportsBlit(Format, Format) const {
    return caps().texture_blit;
  }
  virtual void QueryMemoryBudget(u64* used, u64* budget) const {
    *used = *budget = 0;
  }

  virtual u64 Submit(CommandList* const* lists, u32 count) = 0;
  u64 Submit(CommandList* list) { return Submit(&list, 1); }
  virtual bool IsComplete(u64 submission) = 0;
  virtual bool Wait(u64 submission, u64 timeout_ns = ~0ull) = 0;
  virtual void WaitIdle() = 0;
  virtual void ReleaseSessionObjects() {
    WaitIdle();
    ReleaseObjects();
  }
  virtual u64 LastSubmission() const = 0;

  virtual bool ReadTimestamps(TimestampPool* pool,
                              u32 first,
                              u32 count,
                              u64* out) = 0;

  virtual void Maintain() {}
  virtual void* CaptureHandle() const { return nullptr; }
  virtual void ReportDeviceLoss() {}

 private:
  mutable base::Mutex objects_mutex_;
  base::Vector<Object*> objects_;
};

template <typename T>
T* Release(base::UniquePointer<T>& object, Device* owner = nullptr) {
  T* raw = object.Get_UseOnlyIfYouKnowWhatYouareDoing();
  object.ResetUnchecked_UseOnlyIfYouKnowWhatYouareDoing();
  if constexpr (std::is_base_of_v<Object, T>)
    if (raw && owner)
      owner->Track(raw);
  return raw;
}

}  // namespace gpu::rhi
