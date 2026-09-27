/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstring>

#include "gpu/opengl/gl_rhi_internal.h"
#include <base/containers/array.h>
#include <base/algorithm.h>
#include <base/containers/vector.h>
#include <base/math/value_bounds.h>

namespace gpu::opengl {

namespace {

ClearKind ClearKindOf(rhi::Format format) {
  const GlFormat& f = ToGl(format);
  if (!rhi::GetFormatInfo(format).is_integer)
    return ClearKind::kFloat;
  return f.type == GL_BYTE || f.type == GL_SHORT || f.type == GL_INT
             ? ClearKind::kInt
             : ClearKind::kUint;
}

// Barrier bits that make incoherent (shader) writes visible to `access`.
GLbitfield AccessBits(u32 access) {
  GLbitfield bits = 0;
  if (access & rhi::kAccessVertexRead)
    bits |= GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
  if (access & rhi::kAccessIndexRead)
    bits |= GL_ELEMENT_ARRAY_BARRIER_BIT;
  if (access & rhi::kAccessUniformRead)
    bits |= GL_UNIFORM_BARRIER_BIT;
  if (access & rhi::kAccessIndirectRead)
    bits |= GL_COMMAND_BARRIER_BIT;
  if (access & (rhi::kAccessShaderRead | rhi::kAccessShaderWrite))
    bits |= GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
            GL_TEXTURE_FETCH_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT;
  if (access & (rhi::kAccessCopyRead | rhi::kAccessCopyWrite))
    bits |= GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT |
            GL_TEXTURE_UPDATE_BARRIER_BIT;
  if (access & (rhi::kAccessHostRead | rhi::kAccessHostWrite))
    bits |= GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT;
  if (access & (rhi::kAccessColorWrite | rhi::kAccessDepthWrite))
    bits |= GL_FRAMEBUFFER_BARRIER_BIT;
  return bits;
}

GLbitfield StateBits(rhi::TextureState state) {
  switch (state) {
    case rhi::TextureState::kUndefined:
      return 0;
    case rhi::TextureState::kGeneral:
      return GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
             GL_TEXTURE_UPDATE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT |
             GL_PIXEL_BUFFER_BARRIER_BIT;
    case rhi::TextureState::kColorTarget:
    case rhi::TextureState::kDepthTarget:
      return GL_FRAMEBUFFER_BARRIER_BIT;
    case rhi::TextureState::kDepthRead:
    case rhi::TextureState::kShaderRead:
      return GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
             GL_FRAMEBUFFER_BARRIER_BIT;
    case rhi::TextureState::kCopySrc:
    case rhi::TextureState::kCopyDst:
      return GL_TEXTURE_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT;
  }
  return GL_ALL_BARRIER_BITS;
}

GLuint Name(rhi::Buffer* b) {
  return b ? static_cast<GlBuffer*>(b)->name : 0;
}

}  // namespace

GlCommandList::GlCommandList(GlDevice& device) : device_(device) {
  for (u32 k = 0; k < kSlotKinds; k++)
    slot_state_[k].resize(device.slot_limit(static_cast<SlotKind>(k)));
}

void GlCommandList::Begin() {
  stream_.Clear();
  pipeline_ = nullptr;
  flushed_slots_ = nullptr;
  flushed_push_ = nullptr;
  for (BoundGroup& g : groups_)
    g = {};
  push_groups_used_ = 0;
  push_dirty_ = true;
  vertex_dirty_ = 0;
  flushed_input_ = nullptr;
  index_dirty_ = true;
  for (auto& slots : slot_state_)
    base::Fill(slots.begin(), slots.end(), SlotState{});
  pointers_dirty_ = true;
  flushed_pointers_ = nullptr;
  base::Fill(pass_colors_, pass_colors_ + base::ArraySize(pass_colors_), nullptr);
}

void GlCommandList::BeginRenderPass(const rhi::RenderPassDesc& pass) {
  auto* c = stream_.Add<CmdBeginPass>();
  c->color_count = static_cast<u8>(base::Min(pass.color_count, 8u));
  u8 store_discard = 0;
  for (u32 i = 0; i < c->color_count; i++) {
    const rhi::ColorAttachment& a = pass.colors[i];
    const auto* v = static_cast<const GlView*>(a.view);
    pass_colors_[i] = v;
    c->key.colors[i] = v;
    if (!v)
      continue;
    c->kinds[i] = ClearKindOf(v->format);
    c->clear[i] = a.clear;
    if (a.load == rhi::LoadOp::kClear)
      c->clear_mask |= 1u << i;
    else if (a.load == rhi::LoadOp::kDontCare)
      c->discard_mask |= 1u << i;
    if (a.store == rhi::StoreOp::kDontCare)
      store_discard |= 1u << i;
  }
  for (u32 i = c->color_count; i < 8; i++)
    pass_colors_[i] = nullptr;
  const rhi::DepthAttachment& d = pass.depth;
  if (d.view) {
    const rhi::FormatInfo& fi =
        rhi::GetFormatInfo(static_cast<const GlView*>(d.view)->format);
    c->key.depth = static_cast<const GlView*>(d.view);
    if (d.depth && fi.is_depth)
      c->key.depth_planes |= rhi::kAspectDepth;
    if (d.stencil && fi.is_stencil)
      c->key.depth_planes |= rhi::kAspectStencil;
    auto load = [&](u8 plane, rhi::LoadOp op) {
      if (!(c->key.depth_planes & plane))
        return;
      if (op == rhi::LoadOp::kClear)
        c->depth_clear |= plane;
      else if (op == rhi::LoadOp::kDontCare)
        c->depth_discard |= plane;
    };
    load(rhi::kAspectDepth, d.depth_load);
    load(rhi::kAspectStencil, d.stencil_load);
    if (d.read_only)
      c->depth_read_only |= rhi::kAspectDepth;
    if (d.stencil_read_only)
      c->depth_read_only |= rhi::kAspectStencil;
    c->clear_depth = d.clear_depth;
    c->clear_stencil = d.clear_stencil;
  }
  c->x = pass.x;
  c->y = pass.y;
  c->width = pass.width;
  c->height = pass.height;
  pass_end_ = {};
  pass_end_.discard_mask = store_discard;
  pass_end_.x = pass.x;
  pass_end_.y = pass.y;
  pass_end_.width = pass.width;
  pass_end_.height = pass.height;
}

void GlCommandList::EndRenderPass() {
  auto* c = stream_.Add<CmdEndPass>();
  const Cmd h = c->h;
  *c = pass_end_;
  c->h = h;
}

void GlCommandList::SetPipeline(rhi::Pipeline* pipeline) {
  pipeline_ = static_cast<GlPipeline*>(pipeline);
  stream_.Add<CmdPipeline>()->pipeline = pipeline_;
}

void GlCommandList::SetBindGroup(u32 index,
                                 rhi::BindGroup* group,
                                 const u32* dynamic_offsets,
                                 u32 num_offsets) {
  if (index >= kMaxGroups || !group)
    return;
  auto* g = static_cast<GlBindGroup*>(group);
  BoundGroup& b = groups_[index];
  b.layout = g->layout;
  b.entries = g->entries.data();
  std::memcpy(b.offsets, dynamic_offsets,
              base::Min(num_offsets, kMaxDynamicOffsets) * sizeof(u32));
  b.dirty = true;
}

void GlCommandList::PushBindGroup(u32 index,
                                  rhi::BindGroupLayout* layout,
                                  const rhi::BindingWrite* writes,
                                  u32 num_writes) {
  if (index >= kMaxGroups)
    return;
  auto* l = static_cast<GlBindGroupLayout*>(layout);
  if (push_groups_used_ == push_groups_.size())
    push_groups_.emplace_back();
  base::Vector<BoundResource>& entries = push_groups_[push_groups_used_++];
  entries.assign(l->desc().bindings.size(), BoundResource{});
  for (u32 i = 0; i < num_writes; i++)
    device_.Resolve(*l, writes[i], entries.data());
  // Pushed dynamic bindings take their whole offset from the write.
  for (BoundResource& r : entries)
    r.dynamic = kNotDynamic;
  BoundGroup& b = groups_[index];
  b.layout = l;
  b.entries = entries.data();
  b.dirty = true;
}

void GlCommandList::SetPushConstants(u32 offset, u32 bytes, const void* data) {
  if (offset >= kMaxPushBytes)
    return;
  std::memcpy(push_data_ + offset, data, base::Min(bytes, kMaxPushBytes - offset));
  push_dirty_ = true;
}

void GlCommandList::SetVertexBuffers(u32 first,
                                     u32 count,
                                     rhi::Buffer* const* buffers,
                                     const u64* offsets) {
  for (u32 i = 0; i < count && first + i < kMaxVertexBuffers; i++) {
    vertex_buffers_[first + i] = Name(buffers[i]);
    vertex_offsets_[first + i] = offsets[i];
    vertex_dirty_ |= 1u << (first + i);
  }
}

void GlCommandList::SetIndexBuffer(rhi::Buffer* buffer,
                                   u64 offset,
                                   rhi::IndexType type) {
  const GLuint name = Name(buffer);
  if (name != index_buffer_)
    index_dirty_ = true;
  index_buffer_ = name;
  index_offset_ = offset;
  index_type_ = type == rhi::IndexType::kUint32 ? GL_UNSIGNED_INT
                                                : GL_UNSIGNED_SHORT;
}

void GlCommandList::SetViewport(float x,
                                float y,
                                float width,
                                float height,
                                float min_depth,
                                float max_depth) {
  auto* c = stream_.Add<CmdViewport>();
  c->x = x;
  c->y = y;
  c->width = width;
  c->height = height;
  c->min_depth = min_depth;
  c->max_depth = max_depth;
}

void GlCommandList::SetScissor(i32 x, i32 y, u32 width, u32 height) {
  auto* c = stream_.Add<CmdScissor>();
  c->x = x;
  c->y = y;
  c->width = width;
  c->height = height;
}

void GlCommandList::SetBlendConstants(const float rgba[4]) {
  std::memcpy(stream_.Add<CmdBlendConstants>()->rgba, rgba, 4 * sizeof(float));
}

void GlCommandList::BindBuffer(SlotKind kind,
                               u32 slot,
                               const BoundResource& r,
                               u64 offset) {
  SlotState& s = slot_state_[static_cast<u32>(kind)][slot];
  u64 size = r.size;
  if (r.buffer_size && offset + size > r.buffer_size)
    size = offset < r.buffer_size ? r.buffer_size - offset : 0;
  if (s.name == r.name && s.offset == offset && s.size == size)
    return;
  s = {r.name, 0, offset, size};
  auto* c = stream_.Add<CmdBindBuffer>();
  c->target = kind == SlotKind::kUniformBuffer ? GL_UNIFORM_BUFFER
                                               : GL_SHADER_STORAGE_BUFFER;
  c->slot = slot;
  c->name = size ? r.name : 0;
  c->offset = offset;
  c->size = size;
}

void GlCommandList::FlushGroup(u32 index, const SlotMap::Group& map) {
  const BoundGroup& b = groups_[index];
  for (const SlotMap::Slot& slot : map.slots) {
    const i32 pos = b.layout == map.layout ? slot.position
                                           : b.layout->Position(slot.binding);
    if (pos < 0)
      continue;
    const BoundResource& r = b.entries[pos];
    switch (slot.kind) {
      case SlotKind::kUniformBuffer:
      case SlotKind::kStorageBuffer:
        BindBuffer(slot.kind, slot.slot, r,
                   r.offset + (r.dynamic != kNotDynamic ? b.offsets[r.dynamic]
                                                        : 0));
        break;
      case SlotKind::kTexture: {
        SlotState& s = slot_state_[static_cast<u32>(slot.kind)][slot.slot];
        if (s.name == r.name && s.sampler == r.sampler)
          break;
        s.name = r.name;
        s.sampler = r.sampler;
        auto* c = stream_.Add<CmdBindTexture>();
        c->unit = slot.slot;
        c->name = r.name;
        c->sampler = r.sampler;
        break;
      }
      case SlotKind::kPointer: {
        const u64 offset =
            r.offset + (r.dynamic != kNotDynamic ? b.offsets[r.dynamic] : 0);
        u64 size = r.size;
        if (r.buffer_size && offset + size > r.buffer_size)
          size = offset < r.buffer_size ? r.buffer_size - offset : 0;
        const u64 address = r.address && size ? r.address + offset : 0;
        const u32 entry[4] = {static_cast<u32>(address),
                              static_cast<u32>(address >> 32),
                              static_cast<u32>(address ? size : 0), 0};
        if (std::memcmp(pointers_[slot.slot], entry, sizeof(entry))) {
          std::memcpy(pointers_[slot.slot], entry, sizeof(entry));
          pointers_dirty_ = true;
        }
        break;
      }
      case SlotKind::kImage: {
        SlotState& s = slot_state_[static_cast<u32>(slot.kind)][slot.slot];
        const u64 where = (u64(r.level) << 32) | u32(r.layer) |
                          (u64(r.layered) << 63);
        if (s.name == r.name && s.offset == where && s.size == r.format)
          break;
        s = {r.name, 0, where, r.format};
        auto* c = stream_.Add<CmdBindImage>();
        c->unit = slot.slot;
        c->name = r.name;
        c->level = r.level;
        c->layer = r.layer;
        c->layered = r.layered;
        c->format = r.format ? r.format : GL_R32UI;
        break;
      }
    }
  }
}

void GlCommandList::FlushBindings() {
  const SlotMap* map = pipeline_->slots;
  if (map != flushed_slots_) {
    for (BoundGroup& g : groups_)
      g.dirty = true;
    flushed_slots_ = map;
  }
  for (u32 i = 0; i < map->groups.size() && i < kMaxGroups; i++) {
    BoundGroup& g = groups_[i];
    if (!g.dirty || !g.entries || map->groups[i].slots.empty())
      continue;
    FlushGroup(i, map->groups[i]);
    g.dirty = false;
  }
  // The pointer table and push constants are uniforms of the program, so a
  // new program needs them again even when the data did not change.
  if (pipeline_->pointer_count &&
      (pointers_dirty_ || flushed_pointers_ != pipeline_)) {
    const u32 count = base::Min(pipeline_->pointer_count, kMaxPointers);
    auto* c = stream_.Add<CmdPointers>(count * sizeof(pointers_[0]));
    c->count = count;
    std::memcpy(c + 1, pointers_, count * sizeof(pointers_[0]));
    pointers_dirty_ = false;
    flushed_pointers_ = pipeline_;
  }
  if ((pipeline_->push_count || pipeline_->push_ubo) &&
      (push_dirty_ || flushed_push_ != pipeline_)) {
    u32 bytes = pipeline_->push_bytes;
    for (u32 i = 0; i < pipeline_->push_count; i++)
      bytes = base::Max(bytes, pipeline_->push[i].vec4_count * 16);
    bytes = base::Min(bytes, kMaxPushBytes);
    auto* c = stream_.Add<CmdPush>(bytes);
    c->bytes = bytes;
    std::memcpy(c + 1, push_data_, bytes);
    push_dirty_ = false;
    flushed_push_ = pipeline_;
  }
}

void GlCommandList::FlushVertexInput(bool indexed) {
  VertexInput* input = pipeline_->vertex;
  const u32 count = static_cast<u32>(input->per_instance.size());
  if (input != flushed_input_ ||
      std::memcmp(flushed_strides_, pipeline_->strides,
                  sizeof(flushed_strides_))) {
    flushed_input_ = input;
    std::memcpy(flushed_strides_, pipeline_->strides,
                sizeof(flushed_strides_));
    vertex_dirty_ = (1u << count) - 1;
    index_dirty_ = true;
  }
  vertex_dirty_ &= (1u << count) - 1;
  if (vertex_dirty_) {
    auto* c = stream_.Add<CmdVertexBuffers>();
    c->input = input;
    c->first = 0;
    c->count = count;
    for (u32 i = 0; i < count; i++) {
      c->names[i] = vertex_buffers_[i];
      c->offsets[i] = static_cast<GLintptr>(vertex_offsets_[i]);
      c->strides[i] = static_cast<GLsizei>(pipeline_->strides[i]);
    }
    vertex_dirty_ = 0;
  }
  if (indexed && index_dirty_) {
    auto* c = stream_.Add<CmdIndexBuffer>();
    c->input = input;
    c->name = index_buffer_;
    index_dirty_ = false;
  }
}

void GlCommandList::Draw(u32 vertex_count,
                         u32 instance_count,
                         u32 first_vertex,
                         u32 first_instance) {
  if (!pipeline_ || pipeline_->compute)
    return;
  FlushBindings();
  FlushVertexInput(false);
  auto* c = stream_.Add<CmdDraw>();
  c->vertex_count = vertex_count;
  c->instance_count = instance_count;
  c->first_vertex = first_vertex;
  c->first_instance = first_instance;
}

void GlCommandList::DrawIndexed(u32 index_count,
                                u32 instance_count,
                                u32 first_index,
                                i32 vertex_offset,
                                u32 first_instance) {
  if (!pipeline_ || pipeline_->compute)
    return;
  FlushBindings();
  FlushVertexInput(true);
  auto* c = stream_.Add<CmdDrawIndexed>();
  c->index_count = index_count;
  c->instance_count = instance_count;
  c->vertex_offset = vertex_offset;
  c->first_instance = first_instance;
  c->type = index_type_;
  c->offset =
      index_offset_ + u64(first_index) * (index_type_ == GL_UNSIGNED_INT ? 4 : 2);
}

void GlCommandList::DrawMeshTasks(u32 /*x*/, u32 /*y*/, u32 /*z*/) {}

void GlCommandList::Dispatch(u32 x, u32 y, u32 z) {
  DispatchBase(0, 0, 0, x, y, z);
}

void GlCommandList::DispatchBase(u32 bx,
                                 u32 by,
                                 u32 bz,
                                 u32 x,
                                 u32 y,
                                 u32 z) {
  if (!pipeline_ || !pipeline_->compute)
    return;
  FlushBindings();
  auto* c = stream_.Add<CmdDispatch>();
  c->base[0] = bx;
  c->base[1] = by;
  c->base[2] = bz;
  c->x = x;
  c->y = y;
  c->z = z;
}

void GlCommandList::ClearAttachment(u32 attachment,
                                    const rhi::ClearColor& color,
                                    float depth,
                                    u8 stencil,
                                    u8 aspect,
                                    i32 x,
                                    i32 y,
                                    u32 width,
                                    u32 height) {
  auto* c = stream_.Add<CmdClearAttachment>();
  c->attachment = attachment;
  if (attachment < 8 && pass_colors_[attachment])
    c->kind = ClearKindOf(pass_colors_[attachment]->format);
  c->aspect = aspect;
  c->color = color;
  c->depth = depth;
  c->stencil = stencil;
  c->x = x;
  c->y = y;
  c->width = width;
  c->height = height;
}

void GlCommandList::CopyBuffer(rhi::Buffer* dst,
                               u64 dst_offset,
                               rhi::Buffer* src,
                               u64 src_offset,
                               u64 bytes) {
  auto* c = stream_.Add<CmdCopyBuffer>();
  c->dst = Name(dst);
  c->src = Name(src);
  c->dst_offset = dst_offset;
  c->src_offset = src_offset;
  c->bytes = bytes;
}

void GlCommandList::CopyBufferToTexture(rhi::Texture* dst,
                                        rhi::Buffer* src,
                                        const rhi::BufferTextureCopy* regions,
                                        u32 count) {
  const size_t bytes = count * sizeof(rhi::BufferTextureCopy);
  auto* c = stream_.Add<CmdBufferToTexture>(bytes);
  c->texture = static_cast<GlTexture*>(dst);
  c->buffer = static_cast<GlBuffer*>(src);
  c->count = count;
  std::memcpy(c + 1, regions, bytes);
}

void GlCommandList::CopyTextureToBuffer(rhi::Buffer* dst,
                                        rhi::Texture* src,
                                        const rhi::BufferTextureCopy* regions,
                                        u32 count) {
  const size_t bytes = count * sizeof(rhi::BufferTextureCopy);
  auto* c = stream_.Add<CmdTextureToBuffer>(bytes);
  c->texture = static_cast<GlTexture*>(src);
  c->buffer = static_cast<GlBuffer*>(dst);
  c->count = count;
  std::memcpy(c + 1, regions, bytes);
}

void GlCommandList::CopyTexture(rhi::Texture* dst,
                                const rhi::TextureRegion& dst_region,
                                rhi::Texture* src,
                                const rhi::TextureRegion& src_region) {
  auto* c = stream_.Add<CmdCopyTexture>();
  c->dst = static_cast<GlTexture*>(dst);
  c->src = static_cast<GlTexture*>(src);
  c->dst_region = dst_region;
  c->src_region = src_region;
}

void GlCommandList::BlitTexture(rhi::Texture* dst,
                                const rhi::TextureRegion& dst_region,
                                rhi::Texture* src,
                                const rhi::TextureRegion& src_region,
                                rhi::Filter filter) {
  auto* c = stream_.Add<CmdBlit>();
  c->dst = static_cast<GlTexture*>(dst);
  c->src = static_cast<GlTexture*>(src);
  c->dst_region = dst_region;
  c->src_region = src_region;
  c->filter = filter == rhi::Filter::kLinear ? GL_LINEAR : GL_NEAREST;
}

void GlCommandList::ClearTexture(rhi::Texture* texture,
                                 rhi::TextureState /*state*/,
                                 const rhi::TextureRange& range,
                                 const rhi::ClearColor& color) {
  auto* c = stream_.Add<CmdClearTexture>();
  c->texture = static_cast<GlTexture*>(texture);
  c->range = range;
  c->color = color;
}

void GlCommandList::ClearDepthStencil(rhi::Texture* texture,
                                      rhi::TextureState /*state*/,
                                      const rhi::TextureRange& range,
                                      float depth,
                                      u8 stencil) {
  auto* c = stream_.Add<CmdClearDepthStencil>();
  c->texture = static_cast<GlTexture*>(texture);
  c->range = range;
  c->depth = depth;
  c->stencil = stencil;
}

void GlCommandList::FillBuffer(rhi::Buffer* buffer,
                               u64 offset,
                               u64 bytes,
                               u32 value) {
  auto* c = stream_.Add<CmdFillBuffer>();
  c->name = Name(buffer);
  c->offset = offset;
  c->bytes = bytes == ~0ull ? buffer->desc().size - offset : bytes;
  c->value = value;
}

void GlCommandList::UpdateBuffer(rhi::Buffer* buffer,
                                 u64 offset,
                                 u64 bytes,
                                 const void* data) {
  auto* c = stream_.Add<CmdUpdateBuffer>(bytes);
  c->name = Name(buffer);
  c->offset = offset;
  c->bytes = bytes;
  std::memcpy(c + 1, data, bytes);
}

void GlCommandList::Barrier(u32 src_access,
                            u32 dst_access,
                            const rhi::TextureBarrier* textures,
                            u32 num_textures) {
  // GL orders and makes visible everything but shader stores and client
  // mappings on its own.
  GLbitfield bits = 0;
  if (src_access & rhi::kAccessShaderWrite)
    bits |= AccessBits(dst_access);
  if (dst_access & rhi::kAccessHostRead)
    bits |= GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT;
  for (u32 i = 0; i < num_textures; i++)
    if (textures[i].before == rhi::TextureState::kGeneral)
      bits |= StateBits(textures[i].after);
  if (bits)
    stream_.Add<CmdBarrier>()->bits = bits;
}

void GlCommandList::ResetTimestamps(rhi::TimestampPool* pool,
                                    u32 first,
                                    u32 count) {
  auto* c = stream_.Add<CmdResetTimestamps>();
  c->pool = static_cast<GlTimestampPool*>(pool);
  c->first = first;
  c->count = count;
}

// glQueryCounter stamps once earlier commands are done, which serves both.
void GlCommandList::WriteTimestamp(rhi::TimestampPool* pool,
                                   u32 index,
                                   bool /*start*/) {
  auto* c = stream_.Add<CmdWriteTimestamp>();
  c->pool = static_cast<GlTimestampPool*>(pool);
  c->first = index;
  c->count = 1;
}

void GlCommandList::Label(Op op, const char* text) {
  if (!device_.caps().debug_labels)
    return;
  const size_t n = text ? std::strlen(text) + 1 : 1;
  CmdLabel* c = nullptr;
  switch (op) {
    case Op::kPushLabel:
      c = stream_.Add<CmdPushLabel>(n);
      break;
    case Op::kPopLabel:
      c = stream_.Add<CmdPopLabel>(n);
      break;
    default:
      c = stream_.Add<CmdInsertLabel>(n);
      break;
  }
  std::memcpy(c + 1, text ? text : "", n);
}

void GlCommandList::PushLabel(const char* label) {
  Label(Op::kPushLabel, label);
}

void GlCommandList::PopLabel() {
  Label(Op::kPopLabel, nullptr);
}

void GlCommandList::InsertLabel(const char* label) {
  Label(Op::kInsertLabel, label);
}

}  // namespace gpu::opengl
