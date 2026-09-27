/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstring>

#include "base/logging.h"

#include "base/algorithm.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "gpu/d3d12/d3d12_internal.h"

namespace gpu::d3d12::impl {

namespace {

constexpr u64 kUploadBlock = 4ull << 20;
constexpr u32 kCpuScratch = 64;
const D3D12_RESOURCE_STATES kWriteStates =
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_COPY_DEST |
    D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_DEPTH_WRITE |
    D3D12_RESOURCE_STATE_STREAM_OUT;

constexpr u64 AlignUp(u64 v, u64 a) {
  return (v + a - 1) / a * a;
}

bool IsRead(D3D12_RESOURCE_STATES s) {
  return s != D3D12_RESOURCE_STATE_COMMON && !(s & kWriteStates);
}

// A colour for ClearRenderTargetView, which converts floats to the format.
void ClearFloats(rhi::Format format, const rhi::ClearColor& c, float out[4]) {
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(format);
  const bool sint =
      fi.is_integer &&
      (format == rhi::Format::kR8Sint || format == rhi::Format::kRG8Sint ||
       format == rhi::Format::kRGBA8Sint || format == rhi::Format::kR16Sint ||
       format == rhi::Format::kRG16Sint || format == rhi::Format::kRGBA16Sint ||
       format == rhi::Format::kR32Sint || format == rhi::Format::kRG32Sint ||
       format == rhi::Format::kRGBA32Sint);
  for (u32 i = 0; i < 4; i++)
    out[i] = !fi.is_integer ? c.f[i]
             : sint         ? static_cast<float>(c.i[i])
                            : static_cast<float>(c.u[i]);
}

// Bytes per texel (or block) of one plane of a format.
u32 PlaneBytes(rhi::Format format, u8 aspect) {
  if (format == rhi::Format::kD32FloatS8Uint)
    return aspect == rhi::kAspectStencil ? 1 : 4;
  return rhi::GetFormatInfo(format).bytes;
}

u32 PlaneOf(const D3D12Texture* t, u8 aspect) {
  return t->planes > 1 && aspect == rhi::kAspectStencil ? 1 : 0;
}

}  // namespace

D3D12CommandList::~D3D12CommandList() {
  for (const RingChunk& c : chunks_)
    device_.ReleaseChunk(c);
  for (UploadBlock& b : upload_)
    SafeRelease(b.resource);
  for (UploadBlock& b : scratch_)
    SafeRelease(b.resource);
  SafeRelease(cpu_scratch_);
  SafeRelease(cmd);
  SafeRelease(allocator);
}

bool D3D12CommandList::Init() {
  ID3D12Device* dev = device_.device;
  if (FAILED(dev->CreateCommandAllocator(
          D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator,
          reinterpret_cast<void**>(&allocator))) ||
      FAILED(dev->CreateCommandList(
          0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
          IID_ID3D12GraphicsCommandList, reinterpret_cast<void**>(&cmd))))
    return false;
  cmd->Close();
  D3D12_DESCRIPTOR_HEAP_DESC hd{};
  hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  hd.NumDescriptors = kCpuScratch;
  return SUCCEEDED(dev->CreateDescriptorHeap(
      &hd, IID_ID3D12DescriptorHeap, reinterpret_cast<void**>(&cpu_scratch_)));
}

void D3D12CommandList::ReleaseTransient() {
  while (chunks_.size() > 1) {
    device_.ReleaseChunk(chunks_.back());
    chunks_.pop_back();
  }
  for (RingChunk& c : chunks_)
    c.used = 0;
  for (UploadBlock& b : upload_)
    b.used = 0;
  upload_block_ = 0;
  // Keep the largest scratch buffer for the next recording.
  base::Sort(scratch_.begin(), scratch_.end(),
             [](const UploadBlock& a, const UploadBlock& b) {
               return a.size > b.size;
             });
  while (scratch_.size() > 1) {
    SafeRelease(scratch_.back().resource);
    scratch_.pop_back();
  }
  for (UploadBlock& b : scratch_)
    b.used = 0;
}

void D3D12CommandList::Begin() {
  ReleaseTransient();
  allocator->Reset();
  cmd->Reset(allocator, nullptr);
  ID3D12DescriptorHeap* heaps[] = {device_.ring_heap, device_.sampler_heap};
  cmd->SetDescriptorHeaps(2, heaps);
  barriers_.clear();
  buffer_states_.clear();
  scratch_states_.clear();
  transition_epoch_ = 0;
  pipeline_ = nullptr;
  root_ = nullptr;
  pso_ = nullptr;
  state_lost_ = false;
  for (BoundGroup& g : groups_)
    g = {};
  push_dirty_ = false;
  for (VertexBinding& v : vertex_)
    v = {};
  vertex_count_ = 0;
  vertex_dirty_ = false;
  index_buffer_ = nullptr;
  index_dirty_ = false;
  y_sign_ = 1.0f;
  pass_color_count_ = 0;
  pass_has_dsv_ = false;
  recording_ = true;
}

void D3D12CommandList::End() {
  FlushBarriers();
  cmd->Close();
  recording_ = false;
}

bool D3D12CommandList::AllocateDescriptors(u32 count,
                                           D3D12_CPU_DESCRIPTOR_HANDLE* cpu,
                                           D3D12_GPU_DESCRIPTOR_HANDLE* gpu) {
  if (count > kChunkSize)
    return false;
  if (chunks_.empty() || chunks_.back().used + count > kChunkSize) {
    RingChunk chunk;
    if (!device_.AcquireChunk(&chunk)) {
      BASE_LOGI("gpud3d12", "descriptor ring exhausted");
      return false;
    }
    chunks_.push_back(chunk);
  }
  RingChunk& c = chunks_.back();
  *cpu = device_.RingCpu(c.first + c.used);
  *gpu = device_.RingGpu(c.first + c.used);
  c.used += count;
  return true;
}

UploadAlloc D3D12CommandList::AllocateUpload(u64 bytes, u64 alignment) {
  for (; upload_block_ < upload_.size(); upload_block_++) {
    UploadBlock& b = upload_[upload_block_];
    const u64 offset = AlignUp(b.used, alignment);
    if (offset + bytes <= b.size) {
      b.used = offset + bytes;
      return {b.resource, offset, b.mapped + offset, b.va + offset};
    }
  }
  UploadBlock b;
  b.size = base::Max(kUploadBlock, AlignUp(bytes, 65536));
  b.resource = device_.CreateBufferResource(b.size, D3D12_HEAP_TYPE_UPLOAD,
                                            D3D12_RESOURCE_FLAG_NONE,
                                            D3D12_RESOURCE_STATE_GENERIC_READ);
  void* p = nullptr;
  if (!b.resource || FAILED(b.resource->Map(0, nullptr, &p))) {
    SafeRelease(b.resource);
    return {};
  }
  b.mapped = static_cast<u8*>(p);
  b.va = b.resource->GetGPUVirtualAddress();
  b.used = bytes;
  upload_.push_back(b);
  upload_block_ = upload_.size() - 1;
  return {b.resource, 0, b.mapped, b.va};
}

UploadAlloc D3D12CommandList::AllocateScratch(u64 bytes) {
  for (UploadBlock& b : scratch_) {
    const u64 offset = AlignUp(b.used, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    if (offset + bytes <= b.size) {
      b.used = offset + bytes;
      return {b.resource, offset, nullptr, b.va + offset};
    }
  }
  UploadBlock b;
  b.size = base::Max<u64>(kUploadBlock, AlignUp(bytes, 65536));
  b.resource = device_.CreateBufferResource(b.size, D3D12_HEAP_TYPE_DEFAULT,
                                            D3D12_RESOURCE_FLAG_NONE,
                                            D3D12_RESOURCE_STATE_COMMON);
  if (!b.resource)
    return {};
  b.va = b.resource->GetGPUVirtualAddress();
  b.used = bytes;
  scratch_.push_back(b);
  return {b.resource, 0, nullptr, b.va};
}

void D3D12CommandList::Transition(ID3D12Resource* resource,
                                  u32 subresource,
                                  D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
  if (before == after)
    return;
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition = {resource, subresource, before, after};
  barriers_.push_back(b);
}

void D3D12CommandList::FlushBarriers() {
  if (barriers_.empty())
    return;
  cmd->ResourceBarrier(static_cast<UINT>(barriers_.size()), barriers_.data());
  barriers_.clear();
}

void D3D12CommandList::Require(D3D12Buffer* buffer,
                               D3D12_RESOURCE_STATES state) {
  if (!buffer || buffer->fixed_state)
    return;
  auto it =
      buffer_states_.try_emplace(buffer, D3D12_RESOURCE_STATE_COMMON).first;
  D3D12_RESOURCE_STATES& cur = it->second;
  if (cur == state)
    return;
  D3D12_RESOURCE_STATES next = state;
  if (IsRead(cur) && IsRead(state)) {
    if ((cur & state) == state)
      return;
    next = cur | state;
  }
  Transition(buffer->resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, cur,
             next);
  cur = next;
  transition_epoch_++;
}

void D3D12CommandList::RequireScratch(ID3D12Resource* scratch,
                                      D3D12_RESOURCE_STATES state) {
  auto it =
      scratch_states_.try_emplace(scratch, D3D12_RESOURCE_STATE_COMMON).first;
  Transition(scratch, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, it->second,
             state);
  it->second = state;
}

void D3D12CommandList::TransitionTexture(D3D12Texture* texture,
                                         const rhi::TextureRange& range,
                                         D3D12_RESOURCE_STATES after,
                                         const D3D12_RESOURCE_STATES* before) {
  u32 first_plane = 0, last_plane = 0;
  if (texture->planes > 1) {
    const u8 a = range.aspect & (rhi::kAspectDepth | rhi::kAspectStencil);
    first_plane = a == rhi::kAspectStencil ? 1 : 0;
    last_plane = a == rhi::kAspectDepth ? 0 : 1;
  }
  bool uav_barrier = false;
  for (u32 plane = first_plane; plane <= last_plane; plane++)
    for (u32 layer = 0; layer < range.layers; layer++)
      for (u32 mip = 0; mip < range.mips; mip++) {
        const u32 sub = texture->Subresource(range.base_mip + mip,
                                             range.base_layer + layer, plane);
        if (sub >= texture->states.size())
          continue;
        D3D12_RESOURCE_STATES& cur = texture->states[sub];
        const D3D12_RESOURCE_STATES from = before ? *before : cur;
        if (from == after && after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
          uav_barrier = true;
        Transition(texture->resource, sub, from, after);
        cur = after;
      }
  if (uav_barrier) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = texture->resource;
    barriers_.push_back(b);
  }
}

void D3D12CommandList::Barrier(u32 src_access,
                               u32 dst_access,
                               const rhi::TextureBarrier* textures,
                               u32 num_textures) {
  (void)dst_access;
  // Buffers change state where they are used; a write followed by a read in
  // the same (UAV) state still needs this.
  if (src_access & (rhi::kAccessShaderWrite | rhi::kAccessComputeWrite)) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barriers_.push_back(b);
  }
  for (u32 i = 0; i < num_textures; i++) {
    const rhi::TextureBarrier& t = textures[i];
    D3D12Texture* tex = Tex(t.texture);
    const D3D12_RESOURCE_STATES after = device_.TextureState(tex, t.after);
    if (t.before == rhi::TextureState::kUndefined) {
      TransitionTexture(tex, t.range, after);
    } else {
      const D3D12_RESOURCE_STATES before = device_.TextureState(tex, t.before);
      TransitionTexture(tex, t.range, after, &before);
    }
  }
}

void D3D12CommandList::BeginRenderPass(const rhi::RenderPassDesc& pass) {
  FlushBarriers();
  pass_color_count_ = base::Min(pass.color_count, 8u);
  const D3D12_RECT area{pass.x, pass.y, pass.x + static_cast<LONG>(pass.width),
                        pass.y + static_cast<LONG>(pass.height)};
  for (u32 i = 0; i < pass_color_count_; i++)
    pass_rtvs_[i] = View(pass.colors[i].view)->rtv.cpu;
  const rhi::DepthAttachment& d = pass.depth;
  pass_has_dsv_ = d.view != nullptr;
  if (pass_has_dsv_) {
    D3D12View* v = View(d.view);
    pass_dsv_ =
        device_.Dsv(v, (d.read_only ? 1 : 0) | (d.stencil_read_only ? 2 : 0));
    pass_dsv_clear_ = device_.Dsv(v, 0);
  }
  cmd->OMSetRenderTargets(pass_color_count_, pass_rtvs_, FALSE,
                          pass_has_dsv_ ? &pass_dsv_ : nullptr);
  for (u32 i = 0; i < pass_color_count_; i++) {
    const rhi::ColorAttachment& c = pass.colors[i];
    if (c.load != rhi::LoadOp::kClear)
      continue;
    float color[4];
    ClearFloats(c.view->desc().format == rhi::Format::kUndefined
                    ? c.view->texture()->desc().format
                    : c.view->desc().format,
                c.clear, color);
    cmd->ClearRenderTargetView(pass_rtvs_[i], color, 1, &area);
  }
  if (pass_has_dsv_) {
    const u32 flags =
        (d.depth && d.depth_load == rhi::LoadOp::kClear ? D3D12_CLEAR_FLAG_DEPTH
                                                        : 0) |
        (d.stencil && d.stencil_load == rhi::LoadOp::kClear
             ? D3D12_CLEAR_FLAG_STENCIL
             : 0);
    if (flags)
      cmd->ClearDepthStencilView(pass_dsv_clear_,
                                 static_cast<D3D12_CLEAR_FLAGS>(flags),
                                 d.clear_depth, d.clear_stencil, 1, &area);
  }
  viewport_ = {float(pass.x),      float(pass.y), float(pass.width),
               float(pass.height), 0.0f,          1.0f};
  scissor_ = area;
  cmd->RSSetViewports(1, &viewport_);
  cmd->RSSetScissorRects(1, &scissor_);
}

void D3D12CommandList::EndRenderPass() {
  pass_color_count_ = 0;
  pass_has_dsv_ = false;
}

void D3D12CommandList::SetPipeline(rhi::Pipeline* pipeline) {
  D3D12Pipeline* p = static_cast<D3D12Pipeline*>(pipeline);
  if (!p->compute &&
      (!pipeline_ || pipeline_->compute ||
       std::memcmp(p->strides, pipeline_->strides, sizeof(p->strides))))
    vertex_dirty_ = true;
  pipeline_ = p;
  if (!p->compute) {
    cmd->IASetPrimitiveTopology(p->topology);
    cmd->OMSetStencilRef(p->stencil_ref);
  }
}

void D3D12CommandList::BindTable(
    u32 index,
    D3D12BindGroupLayout* layout,
    D3D12_GPU_DESCRIPTOR_HANDLE table,
    D3D12_GPU_DESCRIPTOR_HANDLE samplers,
    base::Vector<D3D12_GPU_VIRTUAL_ADDRESS> root_cbv,
    base::Vector<BufferUse> uses) {
  if (index >= 8)
    return;
  BoundGroup& g = groups_[index];
  g.layout = layout;
  g.table = table;
  g.samplers = samplers;
  g.root_cbv = base::move(root_cbv);
  g.uses = base::move(uses);
  g.dirty = true;
  g.validated_epoch = ~0ull;
}

void D3D12CommandList::WriteBufferView(D3D12_CPU_DESCRIPTOR_HANDLE dst,
                                       const GroupEntry& e,
                                       D3D12Buffer* buffer,
                                       u64 offset,
                                       u64 range) {
  switch (e.type) {
    case rhi::BindingType::kUniformBuffer:
    case rhi::BindingType::kUniformBufferDynamic:
      device_.WriteCbv(dst, buffer, offset, range);
      break;
    default:
      device_.WriteRawView(dst, buffer, offset, range, !e.read_only);
      break;
  }
}

void D3D12CommandList::SetBindGroup(u32 index,
                                    rhi::BindGroup* bind_group,
                                    const u32* dynamic_offsets,
                                    u32 num_offsets) {
  auto* group = static_cast<D3D12BindGroup*>(bind_group);
  D3D12BindGroupLayout* layout = group->layout;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
  D3D12_GPU_DESCRIPTOR_HANDLE table{};
  if (layout->table_size) {
    if (!AllocateDescriptors(layout->table_size, &cpu, &table))
      return;
    device_.device->CopyDescriptorsSimple(
        layout->table_size, cpu, group->table.cpu,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  }
  base::Vector<D3D12_GPU_VIRTUAL_ADDRESS> root_cbv(
      layout->dynamic_entries.size());
  for (u32 d = 0; d < layout->dynamic_entries.size(); d++) {
    const GroupEntry& e = layout->entries[layout->dynamic_entries[d]];
    const rhi::BindingWrite& w = group->dynamic[d];
    D3D12Buffer* buffer = Buf(w.buffer);
    const u64 offset = w.offset + (d < num_offsets ? dynamic_offsets[d] : 0);
    if (buffer)
      root_cbv[d] = buffer->va + offset;
    // Also in the table, for a pipeline layout that could not afford it a
    // root descriptor.
    WriteBufferView({cpu.ptr + size_t(e.slot) * device_.view_increment}, e,
                    buffer, offset, w.range);
  }
  D3D12_GPU_DESCRIPTOR_HANDLE samplers{};
  if (layout->sampler_count)
    samplers = device_.SamplerTable(group->samplers, &group->sampler_table,
                                    &group->sampler_generation);
  base::Vector<BufferUse> uses;
  for (const BufferUse& u : group->uses)
    if (u.buffer)
      uses.push_back(u);
  BindTable(index, layout, table, samplers, base::move(root_cbv),
            base::move(uses));
}

void D3D12CommandList::PushBindGroup(u32 index,
                                     rhi::BindGroupLayout* bind_layout,
                                     const rhi::BindingWrite* writes,
                                     u32 num_writes) {
  auto* layout = static_cast<D3D12BindGroupLayout*>(bind_layout);
  D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
  D3D12_GPU_DESCRIPTOR_HANDLE table{};
  if (layout->table_size &&
      !AllocateDescriptors(layout->table_size, &cpu, &table))
    return;
  base::Vector<bool> written(layout->entries.size());
  base::Vector<u32> sampler_ids(layout->sampler_count, 0);
  base::Vector<D3D12_GPU_VIRTUAL_ADDRESS> root_cbv(
      layout->dynamic_entries.size());
  base::Vector<BufferUse> uses;
  for (u32 i = 0; i < num_writes; i++) {
    const rhi::BindingWrite& w = writes[i];
    const GroupEntry* e = layout->Find(w.binding);
    if (!e)
      continue;
    written[e - layout->entries.data()] = true;
    const D3D12_CPU_DESCRIPTOR_HANDLE dst{cpu.ptr + size_t(e->slot) *
                                                        device_.view_increment};
    D3D12Buffer* buffer = Buf(w.buffer);
    switch (e->type) {
      case rhi::BindingType::kSampledTexture:
        if (w.view && View(w.view)->srv.valid())
          device_.device->CopyDescriptorsSimple(
              1, dst, View(w.view)->srv.cpu,
              D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        else
          device_.WriteNull(dst, e->type, true);
        sampler_ids[e->sampler_slot] =
            w.sampler ? static_cast<D3D12Sampler*>(w.sampler)->id : 0;
        break;
      case rhi::BindingType::kStorageTexture:
        if (w.view && View(w.view)->uav.valid())
          device_.device->CopyDescriptorsSimple(
              1, dst, View(w.view)->uav.cpu,
              D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        else
          device_.WriteNull(dst, e->type, false);
        break;
      default: {
        WriteBufferView(dst, *e, buffer, w.offset, w.range);
        if (e->dynamic_index != ~0u && buffer)
          root_cbv[e->dynamic_index] = buffer->va + w.offset;
        const bool ubo = e->type == rhi::BindingType::kUniformBuffer ||
                         e->type == rhi::BindingType::kUniformBufferDynamic;
        if (buffer)
          uses.push_back(
              {buffer, ubo ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                       : e->read_only
                           ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                 D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                           : D3D12_RESOURCE_STATE_UNORDERED_ACCESS});
        break;
      }
    }
  }
  for (size_t i = 0; i < layout->entries.size(); i++)
    if (!written[i])
      device_.WriteNull(
          {cpu.ptr + size_t(layout->entries[i].slot) * device_.view_increment},
          layout->entries[i].type, layout->entries[i].read_only);
  D3D12_GPU_DESCRIPTOR_HANDLE samplers{};
  if (layout->sampler_count)
    samplers = device_.SamplerTable(sampler_ids, nullptr, nullptr);
  BindTable(index, layout, table, samplers, base::move(root_cbv),
            base::move(uses));
}

void D3D12CommandList::SetPushConstants(u32 offset,
                                        u32 bytes,
                                        const void* data) {
  if (offset + bytes > sizeof(push_))
    return;
  std::memcpy(reinterpret_cast<u8*>(push_) + offset, data, bytes);
  push_dirty_ = true;
}

void D3D12CommandList::SetVertexBuffers(u32 first,
                                        u32 count,
                                        rhi::Buffer* const* buffers,
                                        const u64* offsets) {
  for (u32 i = 0; i < count && first + i < 16; i++)
    vertex_[first + i] = {Buf(buffers[i]), offsets[i]};
  vertex_count_ = base::Max(vertex_count_, base::Min(first + count, 16u));
  vertex_dirty_ = true;
}

void D3D12CommandList::SetIndexBuffer(rhi::Buffer* buffer,
                                      u64 offset,
                                      rhi::IndexType type) {
  index_buffer_ = Buf(buffer);
  index_offset_ = offset;
  index_type_ = type;
  index_dirty_ = true;
}

void D3D12CommandList::SetViewport(float x,
                                   float y,
                                   float width,
                                   float height,
                                   float min_depth,
                                   float max_depth) {
  // D3D12's viewport is Vulkan's y-up one: a negative Vulkan height needs no
  // flip, a positive one flips clip-space y in the last vertex stage.
  if (height < 0.0f) {
    viewport_ = {x, y + height, width, -height, min_depth, max_depth};
    y_sign_ = 1.0f;
  } else {
    viewport_ = {x, y, width, height, min_depth, max_depth};
    y_sign_ = -1.0f;
  }
  viewport_.MinDepth = base::Clamp(viewport_.MinDepth, 0.0f, 1.0f);
  viewport_.MaxDepth = base::Clamp(viewport_.MaxDepth, 0.0f, 1.0f);
  cmd->RSSetViewports(1, &viewport_);
}

void D3D12CommandList::SetScissor(i32 x, i32 y, u32 width, u32 height) {
  scissor_ = {x, y, x + static_cast<LONG>(width),
              y + static_cast<LONG>(height)};
  cmd->RSSetScissorRects(1, &scissor_);
}

void D3D12CommandList::SetBlendConstants(const float rgba[4]) {
  std::memcpy(blend_, rgba, sizeof(blend_));
  cmd->OMSetBlendFactor(blend_);
}

void D3D12CommandList::ApplyVertexBuffers() {
  D3D12_VERTEX_BUFFER_VIEW views[16]{};
  for (u32 i = 0; i < vertex_count_; i++) {
    const VertexBinding& v = vertex_[i];
    if (!v.buffer)
      continue;
    views[i].BufferLocation = v.buffer->va + v.offset;
    views[i].SizeInBytes = static_cast<UINT>(
        v.buffer->desc().size - base::Min(v.offset, v.buffer->desc().size));
    views[i].StrideInBytes = pipeline_->strides[i];
  }
  cmd->IASetVertexBuffers(0, vertex_count_, views);
}

void D3D12CommandList::FlushState(bool compute) {
  D3D12PipelineLayout* layout = pipeline_->layout;
  const bool new_root =
      state_lost_ || root_ != layout->root || root_compute_ != compute;
  if (new_root) {
    if (compute)
      cmd->SetComputeRootSignature(layout->root);
    else
      cmd->SetGraphicsRootSignature(layout->root);
    root_ = layout->root;
    root_compute_ = compute;
    for (BoundGroup& g : groups_)
      g.dirty = g.layout != nullptr;
    push_dirty_ = true;
  }
  ID3D12PipelineState* pso = pipeline_->pso;
  if (!compute && pipeline_->pso_restart32 &&
      index_type_ == rhi::IndexType::kUint32)
    pso = pipeline_->pso_restart32;
  if (pso != pso_ || state_lost_) {
    cmd->SetPipelineState(pso);
    pso_ = pso;
  }
  if (state_lost_ && !compute) {
    cmd->IASetPrimitiveTopology(pipeline_->topology);
    cmd->OMSetStencilRef(pipeline_->stencil_ref);
    cmd->RSSetViewports(1, &viewport_);
    cmd->RSSetScissorRects(1, &scissor_);
    cmd->OMSetBlendFactor(blend_);
    cmd->OMSetRenderTargets(pass_color_count_, pass_rtvs_, FALSE,
                            pass_has_dsv_ ? &pass_dsv_ : nullptr);
    vertex_dirty_ = index_dirty_ = true;
  }
  state_lost_ = false;

  for (u32 i = 0; i < layout->groups.size() && i < 8; i++) {
    BoundGroup& g = groups_[i];
    if (!g.layout)
      continue;
    if (g.validated_epoch != transition_epoch_) {
      for (const BufferUse& u : g.uses)
        Require(u.buffer, u.state);
      g.validated_epoch = transition_epoch_;
    }
    if (!g.dirty)
      continue;
    g.dirty = false;
    const RootGroup& rg = layout->groups[i];
    if (compute) {
      if (rg.table >= 0)
        cmd->SetComputeRootDescriptorTable(rg.table, g.table);
      if (rg.sampler_table >= 0)
        cmd->SetComputeRootDescriptorTable(rg.sampler_table, g.samplers);
      for (size_t d = 0; d < rg.root_cbv.size() && d < g.root_cbv.size(); d++)
        if (rg.root_cbv[d] >= 0)
          cmd->SetComputeRootConstantBufferView(rg.root_cbv[d], g.root_cbv[d]);
    } else {
      if (rg.table >= 0)
        cmd->SetGraphicsRootDescriptorTable(rg.table, g.table);
      if (rg.sampler_table >= 0)
        cmd->SetGraphicsRootDescriptorTable(rg.sampler_table, g.samplers);
      for (size_t d = 0; d < rg.root_cbv.size() && d < g.root_cbv.size(); d++)
        if (rg.root_cbv[d] >= 0)
          cmd->SetGraphicsRootConstantBufferView(rg.root_cbv[d], g.root_cbv[d]);
    }
  }

  if (!compute) {
    for (u32 i = 0; i < vertex_count_; i++)
      Require(vertex_[i].buffer,
              D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    if (index_buffer_)
      Require(index_buffer_, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    if (vertex_dirty_ && vertex_count_) {
      ApplyVertexBuffers();
      vertex_dirty_ = false;
    }
    if (index_dirty_ && index_buffer_) {
      D3D12_INDEX_BUFFER_VIEW ib{};
      ib.BufferLocation = index_buffer_->va + index_offset_;
      ib.SizeInBytes = static_cast<UINT>(
          index_buffer_->desc().size -
          base::Min(index_offset_, index_buffer_->desc().size));
      ib.Format = index_type_ == rhi::IndexType::kUint32 ? DXGI_FORMAT_R32_UINT
                                                         : DXGI_FORMAT_R16_UINT;
      cmd->IASetIndexBuffer(&ib);
      index_dirty_ = false;
    }
  }
  FlushBarriers();

  if (push_dirty_ && layout->push_dwords) {
    if (layout->push_constants >= 0) {
      if (compute)
        cmd->SetComputeRoot32BitConstants(layout->push_constants,
                                          layout->push_dwords, push_, 0);
      else
        cmd->SetGraphicsRoot32BitConstants(layout->push_constants,
                                           layout->push_dwords, push_, 0);
    } else if (layout->push_cbv >= 0) {
      const UploadAlloc a = AllocateUpload(layout->push_dwords * 4, 256);
      if (a.cpu) {
        std::memcpy(a.cpu, push_, layout->push_dwords * 4);
        if (compute)
          cmd->SetComputeRootConstantBufferView(layout->push_cbv, a.va);
        else
          cmd->SetGraphicsRootConstantBufferView(layout->push_cbv, a.va);
      }
    }
  }
  push_dirty_ = false;
  if (!compute) {
    u32 sign;
    std::memcpy(&sign, &y_sign_, 4);
    cmd->SetGraphicsRoot32BitConstant(layout->raster, sign, 0);
  }
}

void D3D12CommandList::Draw(u32 vertex_count,
                            u32 instance_count,
                            u32 first_vertex,
                            u32 first_instance) {
  FlushState(false);
  D3D12PipelineLayout* layout = pipeline_->layout;
  if (pipeline_->fan) {
    if (vertex_count < 3)
      return;
    const u32 n = (vertex_count - 2) * 3;
    const UploadAlloc a = AllocateUpload(n * 4, 4);
    if (!a.cpu)
      return;
    u32* idx = reinterpret_cast<u32*>(a.cpu);
    for (u32 t = 0; t < vertex_count - 2; t++) {
      idx[t * 3] = 0;
      idx[t * 3 + 1] = t + 1;
      idx[t * 3 + 2] = t + 2;
    }
    const D3D12_INDEX_BUFFER_VIEW ib{a.va, n * 4, DXGI_FORMAT_R32_UINT};
    cmd->IASetIndexBuffer(&ib);
    index_dirty_ = true;
    const i32 params[2] = {static_cast<i32>(first_vertex),
                           static_cast<i32>(first_instance)};
    cmd->SetGraphicsRoot32BitConstants(layout->draw, 2, params, 0);
    cmd->DrawIndexedInstanced(n, instance_count, 0,
                              static_cast<INT>(first_vertex), first_instance);
    return;
  }
  // SV_VertexID and SV_InstanceID count from zero; gl_VertexIndex and
  // gl_InstanceIndex count from the first vertex and instance.
  const i32 params[2] = {static_cast<i32>(first_vertex),
                         static_cast<i32>(first_instance)};
  cmd->SetGraphicsRoot32BitConstants(layout->draw, 2, params, 0);
  cmd->DrawInstanced(vertex_count, instance_count, first_vertex,
                     first_instance);
}

void D3D12CommandList::DrawIndexed(u32 index_count,
                                   u32 instance_count,
                                   u32 first_index,
                                   i32 vertex_offset,
                                   u32 first_instance) {
  if (pipeline_->fan) {
    static bool logged = false;
    if (!logged)
      BASE_LOGI("gpud3d12", "indexed triangle fans are not supported");
    logged = true;
    return;
  }
  FlushState(false);
  const i32 params[2] = {vertex_offset, static_cast<i32>(first_instance)};
  cmd->SetGraphicsRoot32BitConstants(pipeline_->layout->draw, 2, params, 0);
  cmd->DrawIndexedInstanced(index_count, instance_count, first_index,
                            vertex_offset, first_instance);
}

void D3D12CommandList::DrawMeshTasks(u32, u32, u32) {}

void D3D12CommandList::Dispatch(u32 x, u32 y, u32 z) {
  DispatchBase(0, 0, 0, x, y, z);
}

void D3D12CommandList::DispatchBase(u32 bx,
                                    u32 by,
                                    u32 bz,
                                    u32 x,
                                    u32 y,
                                    u32 z) {
  FlushState(true);
  const u32 groups[3] = {x, y, z};
  cmd->SetComputeRoot32BitConstants(pipeline_->layout->dispatch, 3, groups, 0);
  if (pipeline_->dispatch_base) {
    const u32 base[3] = {bx, by, bz};
    cmd->SetComputeRoot32BitConstants(pipeline_->layout->group_base, 3, base,
                                      0);
  }
  cmd->Dispatch(x, y, z);
}

void D3D12CommandList::ClearAttachment(u32 attachment,
                                       const rhi::ClearColor& color,
                                       float depth,
                                       u8 stencil,
                                       u8 aspect,
                                       i32 x,
                                       i32 y,
                                       u32 width,
                                       u32 height) {
  const D3D12_RECT rect{x, y, x + static_cast<LONG>(width),
                        y + static_cast<LONG>(height)};
  FlushBarriers();
  if (attachment != ~0u) {
    if (attachment < pass_color_count_)
      cmd->ClearRenderTargetView(pass_rtvs_[attachment], color.f, 1, &rect);
    return;
  }
  const u32 flags =
      ((aspect & rhi::kAspectDepth) ? D3D12_CLEAR_FLAG_DEPTH : 0) |
      ((aspect & rhi::kAspectStencil) ? D3D12_CLEAR_FLAG_STENCIL : 0);
  if (pass_has_dsv_ && flags)
    cmd->ClearDepthStencilView(pass_dsv_clear_,
                               static_cast<D3D12_CLEAR_FLAGS>(flags), depth,
                               stencil, 1, &rect);
}

void D3D12CommandList::CopyBuffer(rhi::Buffer* dst,
                                  u64 dst_offset,
                                  rhi::Buffer* src,
                                  u64 src_offset,
                                  u64 bytes) {
  Require(Buf(src), D3D12_RESOURCE_STATE_COPY_SOURCE);
  Require(Buf(dst), D3D12_RESOURCE_STATE_COPY_DEST);
  FlushBarriers();
  cmd->CopyBufferRegion(Buf(dst)->resource, dst_offset, Buf(src)->resource,
                        src_offset, bytes);
}

void D3D12CommandList::CopyBufferTexture(D3D12Texture* texture,
                                         D3D12Buffer* buffer,
                                         const rhi::BufferTextureCopy& c,
                                         bool to_texture) {
  const rhi::TextureRegion& r = c.region;
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(texture->desc().format);
  const u32 block = fi.compressed ? 4 : 1;
  const u32 bpb = PlaneBytes(texture->desc().format, r.aspect);
  const u32 plane = PlaneOf(texture, r.aspect);
  const u32 row_length = c.row_length ? c.row_length : r.width;
  const u32 image_height = c.image_height ? c.image_height : r.height;
  const u64 row_pitch = u64((row_length + block - 1) / block) * bpb;
  const u32 rows = (image_height + block - 1) / block;
  const u32 copy_rows = (r.height + block - 1) / block;
  const u64 row_bytes = u64((r.width + block - 1) / block) * bpb;
  const u64 slice_pitch = row_pitch * rows;
  const u64 layer_pitch = slice_pitch * r.depth;
  const bool depth_format = fi.is_depth || fi.is_stencil;

  const D3D12_RESOURCE_DESC rd = texture->resource->GetDesc();
  for (u32 layer = 0; layer < r.layers; layer++) {
    const u32 sub = texture->Subresource(r.mip, r.base_layer + layer, plane);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    device_.device->GetCopyableFootprints(&rd, sub, 1, 0, &fp, nullptr, nullptr,
                                          nullptr);
    const DXGI_FORMAT format = fp.Footprint.Format;
    D3D12_TEXTURE_COPY_LOCATION tex{};
    tex.pResource = texture->resource;
    tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    tex.SubresourceIndex = sub;
    const u64 offset = c.buffer_offset + layer * layer_pitch;
    const D3D12_BOX box{0, 0, 0, r.width, r.height, r.depth};
    const D3D12_BOX tex_box{UINT(r.x),
                            UINT(r.y),
                            UINT(r.z),
                            UINT(r.x) + r.width,
                            UINT(r.y) + r.height,
                            UINT(r.z) + r.depth};
    // Depth/stencil copies are whole subresources in D3D12.
    const D3D12_BOX* src_box = depth_format ? nullptr : &box;
    const D3D12_BOX* tex_src_box = depth_format ? nullptr : &tex_box;

    D3D12_TEXTURE_COPY_LOCATION buf{};
    buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    buf.PlacedFootprint.Footprint.Format = format;
    buf.PlacedFootprint.Footprint.Depth = r.depth;
    const bool pitch_ok = row_pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0;
    if (pitch_ok && offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT == 0) {
      buf.pResource = buffer->resource;
      buf.PlacedFootprint.Offset = offset;
      buf.PlacedFootprint.Footprint.Width = row_length;
      buf.PlacedFootprint.Footprint.Height =
          r.depth > 1 ? rows * block : copy_rows * block;
      buf.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(row_pitch);
      if (to_texture) {
        Require(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        FlushBarriers();
        cmd->CopyTextureRegion(&tex, r.x, r.y, r.z, &buf, src_box);
      } else {
        Require(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        FlushBarriers();
        cmd->CopyTextureRegion(&buf, 0, 0, 0, &tex, tex_src_box);
      }
      continue;
    }

    // Repack through scratch memory with D3D12's pitch and placement.
    const u64 pitch =
        pitch_ok ? row_pitch
                 : AlignUp(row_bytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const u64 scratch_slice = pitch * copy_rows;
    const UploadAlloc s = AllocateScratch(scratch_slice * r.depth);
    if (!s.resource)
      return;
    buf.pResource = s.resource;
    buf.PlacedFootprint.Offset = s.offset;
    buf.PlacedFootprint.Footprint.Width = (r.width + block - 1) / block * block;
    buf.PlacedFootprint.Footprint.Height = copy_rows * block;
    buf.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(pitch);
    auto rows_copy = [&](bool into_scratch) {
      if (pitch == row_pitch && copy_rows == rows) {
        const u64 bytes = slice_pitch * (r.depth - 1) + row_pitch * copy_rows;
        if (into_scratch)
          cmd->CopyBufferRegion(s.resource, s.offset, buffer->resource, offset,
                                bytes);
        else
          cmd->CopyBufferRegion(buffer->resource, offset, s.resource, s.offset,
                                bytes);
        return;
      }
      for (u32 z = 0; z < r.depth; z++)
        for (u32 row = 0; row < copy_rows; row++) {
          const u64 so = s.offset + z * scratch_slice + row * pitch;
          const u64 bo = offset + z * slice_pitch + row * row_pitch;
          if (into_scratch)
            cmd->CopyBufferRegion(s.resource, so, buffer->resource, bo,
                                  row_bytes);
          else
            cmd->CopyBufferRegion(buffer->resource, bo, s.resource, so,
                                  row_bytes);
        }
    };
    if (to_texture) {
      RequireScratch(s.resource, D3D12_RESOURCE_STATE_COPY_DEST);
      Require(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
      FlushBarriers();
      rows_copy(true);
      RequireScratch(s.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
      FlushBarriers();
      cmd->CopyTextureRegion(&tex, r.x, r.y, r.z, &buf, src_box);
    } else {
      RequireScratch(s.resource, D3D12_RESOURCE_STATE_COPY_DEST);
      FlushBarriers();
      cmd->CopyTextureRegion(&buf, 0, 0, 0, &tex, tex_src_box);
      RequireScratch(s.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
      Require(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
      FlushBarriers();
      rows_copy(false);
    }
  }
}

void D3D12CommandList::CopyBufferToTexture(
    rhi::Texture* dst,
    rhi::Buffer* src,
    const rhi::BufferTextureCopy* regions,
    u32 count) {
  for (u32 i = 0; i < count; i++)
    CopyBufferTexture(Tex(dst), Buf(src), regions[i], true);
}

void D3D12CommandList::CopyTextureToBuffer(
    rhi::Buffer* dst,
    rhi::Texture* src,
    const rhi::BufferTextureCopy* regions,
    u32 count) {
  for (u32 i = 0; i < count; i++)
    CopyBufferTexture(Tex(src), Buf(dst), regions[i], false);
}

void D3D12CommandList::CopyTexture(rhi::Texture* dst,
                                   const rhi::TextureRegion& dst_region,
                                   rhi::Texture* src,
                                   const rhi::TextureRegion& src_region) {
  D3D12Texture* d = Tex(dst);
  D3D12Texture* s = Tex(src);
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(s->desc().format);
  FlushBarriers();
  for (u32 layer = 0; layer < src_region.layers; layer++) {
    D3D12_TEXTURE_COPY_LOCATION to{};
    to.pResource = d->resource;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.SubresourceIndex =
        d->Subresource(dst_region.mip, dst_region.base_layer + layer,
                       PlaneOf(d, dst_region.aspect));
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = s->resource;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.SubresourceIndex =
        s->Subresource(src_region.mip, src_region.base_layer + layer,
                       PlaneOf(s, src_region.aspect));
    const D3D12_BOX box{UINT(src_region.x),
                        UINT(src_region.y),
                        UINT(src_region.z),
                        UINT(src_region.x) + src_region.width,
                        UINT(src_region.y) + src_region.height,
                        UINT(src_region.z) + src_region.depth};
    cmd->CopyTextureRegion(&to, dst_region.x, dst_region.y, dst_region.z, &from,
                           fi.is_depth || fi.is_stencil ? nullptr : &box);
  }
}

CpuRange D3D12CommandList::AttachmentView(D3D12Texture* texture,
                                          u32 mip,
                                          u32 layer,
                                          bool depth) {
  base::LockGuard<base::Mutex> lock(texture->view_mutex);
  auto& cache = depth ? texture->dsvs : texture->rtvs;
  const u32 key = mip | (layer << 8);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  CpuRange range;
  const DXGI_FORMAT format = Dxgi(texture->desc().format).format;
  const bool is3d = texture->desc().dim == rhi::TextureDim::k3D;
  if (depth) {
    if (!device_.dsvs.Allocate(1, &range))
      return {};
    D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = format;
    dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    dv.Texture2DArray = {mip, layer, 1};
    device_.device->CreateDepthStencilView(texture->resource, &dv, range.cpu);
  } else {
    if (!device_.rtvs.Allocate(1, &range))
      return {};
    D3D12_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = format;
    if (is3d) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
      rv.Texture3D = {mip, 0, ~0u};
    } else if (texture->desc().dim == rhi::TextureDim::k1D) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
      rv.Texture1DArray = {mip, layer, 1};
    } else {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
      rv.Texture2DArray = {mip, layer, 1, 0};
    }
    device_.device->CreateRenderTargetView(texture->resource, &rv, range.cpu);
  }
  cache[key] = range;
  return range;
}

void D3D12CommandList::BlitTexture(rhi::Texture* dst,
                                   const rhi::TextureRegion& dst_region,
                                   rhi::Texture* src,
                                   const rhi::TextureRegion& src_region,
                                   rhi::Filter filter) {
  D3D12Texture* d = Tex(dst);
  D3D12Texture* s = Tex(src);
  const DXGI_FORMAT format = Dxgi(d->desc().format).format;
  ID3D12PipelineState* pso = device_.BlitPipeline(format);
  if (!pso || !(d->flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) ||
      rhi::GetFormatInfo(s->desc().format).is_integer) {
    BASE_LOGI("gpud3d12", "blit {} -> {} is not supported",
              rhi::FormatName(s->desc().format),
              rhi::FormatName(d->desc().format));
    return;
  }
  const u32 src_w = base::Max(1u, s->desc().width >> src_region.mip);
  const u32 src_h = base::Max(1u, s->desc().height >> src_region.mip);
  const D3D12_RESOURCE_STATES copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
  const D3D12_RESOURCE_STATES copy_dst = D3D12_RESOURCE_STATE_COPY_DEST;
  const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  for (u32 layer = 0; layer < src_region.layers; layer++) {
    const rhi::TextureRange sr{rhi::kAspectColor, src_region.mip, 1,
                               src_region.base_layer + layer, 1};
    const rhi::TextureRange dr{rhi::kAspectColor, dst_region.mip, 1,
                               dst_region.base_layer + layer, 1};
    TransitionTexture(s, sr, read, &copy_src);
    TransitionTexture(d, dr, D3D12_RESOURCE_STATE_RENDER_TARGET, &copy_dst);
    FlushBarriers();

    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
    if (!AllocateDescriptors(1, &cpu, &gpu))
      return;
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = Dxgi(s->desc().format).format;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2DArray = {
        src_region.mip, 1, src_region.base_layer + layer, 1, 0, 0.0f};
    device_.device->CreateShaderResourceView(s->resource, &sv, cpu);
    const CpuRange rtv =
        AttachmentView(d, dst_region.mip, dst_region.base_layer + layer, false);

    cmd->SetGraphicsRootSignature(device_.blit_root());
    cmd->SetPipelineState(pso);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->OMSetRenderTargets(1, &rtv.cpu, FALSE, nullptr);
    const D3D12_VIEWPORT vp{float(dst_region.x),
                            float(dst_region.y),
                            float(dst_region.width),
                            float(dst_region.height),
                            0.0f,
                            1.0f};
    const D3D12_RECT rect{dst_region.x, dst_region.y,
                          dst_region.x + LONG(dst_region.width),
                          dst_region.y + LONG(dst_region.height)};
    cmd->RSSetViewports(1, &vp);
    cmd->RSSetScissorRects(1, &rect);
    const float consts[4] = {float(src_region.x) / float(src_w),
                             float(src_region.y) / float(src_h),
                             float(src_region.width) / float(src_w),
                             float(src_region.height) / float(src_h)};
    cmd->SetGraphicsRoot32BitConstants(0, 4, consts, 0);
    cmd->SetGraphicsRoot32BitConstant(0, filter == rhi::Filter::kLinear, 4);
    cmd->SetGraphicsRootDescriptorTable(1, gpu);
    cmd->DrawInstanced(3, 1, 0, 0);

    TransitionTexture(s, sr, copy_src, &read);
    const D3D12_RESOURCE_STATES rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
    TransitionTexture(d, dr, copy_dst, &rt);
  }
  state_lost_ = true;
  root_ = nullptr;
}

void D3D12CommandList::ClearTexture(rhi::Texture* texture,
                                    rhi::TextureState state,
                                    const rhi::TextureRange& range,
                                    const rhi::ClearColor& color) {
  D3D12Texture* t = Tex(texture);
  const D3D12_RESOURCE_STATES cur = device_.TextureState(t, state);
  const bool is3d = t->desc().dim == rhi::TextureDim::k3D;
  const u32 layers = is3d ? 1 : range.layers;
  if (t->flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
    const D3D12_RESOURCE_STATES rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
    TransitionTexture(t, range, rt, &cur);
    FlushBarriers();
    float c[4];
    ClearFloats(t->desc().format, color, c);
    for (u32 mip = 0; mip < range.mips; mip++)
      for (u32 layer = 0; layer < layers; layer++) {
        const CpuRange rtv = AttachmentView(t, range.base_mip + mip,
                                            range.base_layer + layer, false);
        if (rtv.valid())
          cmd->ClearRenderTargetView(rtv.cpu, c, 0, nullptr);
      }
    TransitionTexture(t, range, cur, &rt);
    return;
  }
  if (t->flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
    const D3D12_RESOURCE_STATES uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    TransitionTexture(t, range, uav, &cur);
    FlushBarriers();
    const bool integer = rhi::GetFormatInfo(t->desc().format).is_integer;
    for (u32 mip = 0; mip < range.mips; mip++) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
      uv.Format = Dxgi(t->desc().format).format;
      if (is3d) {
        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uv.Texture3D = {range.base_mip + mip, 0, ~0u};
      } else {
        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uv.Texture2DArray = {range.base_mip + mip, range.base_layer,
                             range.layers, 0};
      }
      D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
      D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
      if (!AllocateDescriptors(1, &cpu, &gpu))
        return;
      const D3D12_CPU_DESCRIPTOR_HANDLE local{
          cpu_scratch_->GetCPUDescriptorHandleForHeapStart().ptr +
          size_t(cpu_scratch_next_++ % kCpuScratch) * device_.view_increment};
      device_.device->CreateUnorderedAccessView(t->resource, nullptr, &uv, cpu);
      device_.device->CreateUnorderedAccessView(t->resource, nullptr, &uv,
                                                local);
      if (integer)
        cmd->ClearUnorderedAccessViewUint(gpu, local, t->resource, color.u, 0,
                                          nullptr);
      else
        cmd->ClearUnorderedAccessViewFloat(gpu, local, t->resource, color.f, 0,
                                           nullptr);
    }
    TransitionTexture(t, range, cur, &uav);
    return;
  }
  ClearByCopy(t, range, color);
}

void D3D12CommandList::ClearByCopy(D3D12Texture* texture,
                                   const rhi::TextureRange& range,
                                   const rhi::ClearColor& color) {
  // Formats that are neither renderable nor storable. Only 32-bit channels
  // are packed here; the rest have no such texture on D3D12 hardware.
  const rhi::Format format = texture->desc().format;
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(format);
  if (fi.compressed || fi.bytes != fi.channels * 4u) {
    BASE_LOGI("gpud3d12", "cannot clear a {} texture", rhi::FormatName(format));
    return;
  }
  for (u32 mip = 0; mip < range.mips; mip++) {
    const rhi::TextureDesc& d = texture->desc();
    const u32 w = base::Max(1u, d.width >> (range.base_mip + mip));
    const u32 h = base::Max(1u, d.height >> (range.base_mip + mip));
    const u32 depth = d.dim == rhi::TextureDim::k3D
                          ? base::Max(1u, d.depth >> (range.base_mip + mip))
                          : 1u;
    const u64 pitch =
        AlignUp(u64(w) * fi.bytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const UploadAlloc a = AllocateUpload(
        pitch * h * depth, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    if (!a.cpu)
      return;
    for (u64 row = 0; row < u64(h) * depth; row++)
      for (u32 x = 0; x < w; x++)
        std::memcpy(a.cpu + row * pitch + x * fi.bytes, color.u, fi.bytes);
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = a.resource;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = {a.offset,
                            {Dxgi(format).format, w, h, depth, UINT(pitch)}};
    FlushBarriers();
    for (u32 layer = 0; layer < range.layers; layer++) {
      D3D12_TEXTURE_COPY_LOCATION to{};
      to.pResource = texture->resource;
      to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      to.SubresourceIndex = texture->Subresource(range.base_mip + mip,
                                                 range.base_layer + layer, 0);
      cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
  }
}

void D3D12CommandList::ClearDepthStencil(rhi::Texture* texture,
                                         rhi::TextureState state,
                                         const rhi::TextureRange& range,
                                         float depth,
                                         u8 stencil) {
  D3D12Texture* t = Tex(texture);
  const D3D12_RESOURCE_STATES cur = device_.TextureState(t, state);
  const D3D12_RESOURCE_STATES write = D3D12_RESOURCE_STATE_DEPTH_WRITE;
  rhi::TextureRange all = range;
  all.aspect = rhi::kAspectDepth | rhi::kAspectStencil;
  TransitionTexture(t, all, write, &cur);
  FlushBarriers();
  const u32 flags =
      ((range.aspect & rhi::kAspectDepth) ? D3D12_CLEAR_FLAG_DEPTH : 0) |
      ((range.aspect & rhi::kAspectStencil) && t->planes > 1
           ? D3D12_CLEAR_FLAG_STENCIL
           : 0);
  for (u32 mip = 0; mip < range.mips; mip++)
    for (u32 layer = 0; layer < range.layers; layer++) {
      const CpuRange dsv = AttachmentView(t, range.base_mip + mip,
                                          range.base_layer + layer, true);
      if (dsv.valid() && flags)
        cmd->ClearDepthStencilView(dsv.cpu,
                                   static_cast<D3D12_CLEAR_FLAGS>(flags), depth,
                                   stencil, 0, nullptr);
    }
  TransitionTexture(t, all, cur, &write);
}

void D3D12CommandList::FillBuffer(rhi::Buffer* buffer,
                                  u64 offset,
                                  u64 bytes,
                                  u32 value) {
  D3D12Buffer* b = Buf(buffer);
  if (bytes == ~0ull || offset + bytes > b->desc().size)
    bytes = (b->desc().size - base::Min(offset, b->desc().size)) & ~3ull;
  if (!bytes)
    return;
  if (b->fixed_state) {
    if (b->desc().memory != rhi::MemoryKind::kReadback) {
      BASE_LOGI("gpud3d12", "FillBuffer on an upload buffer");
      return;
    }
    const UploadAlloc a = AllocateUpload(bytes, 4);
    if (!a.cpu)
      return;
    for (u64 i = 0; i < bytes / 4; i++)
      std::memcpy(a.cpu + i * 4, &value, 4);
    FlushBarriers();
    cmd->CopyBufferRegion(b->resource, offset, a.resource, a.offset, bytes);
    return;
  }
  Require(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  FlushBarriers();
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
  uv.Format = DXGI_FORMAT_R32_UINT;
  uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uv.Buffer.FirstElement = offset / 4;
  uv.Buffer.NumElements = static_cast<UINT>(bytes / 4);
  D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
  D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
  if (!AllocateDescriptors(1, &cpu, &gpu))
    return;
  const D3D12_CPU_DESCRIPTOR_HANDLE local{
      cpu_scratch_->GetCPUDescriptorHandleForHeapStart().ptr +
      size_t(cpu_scratch_next_++ % kCpuScratch) * device_.view_increment};
  device_.device->CreateUnorderedAccessView(b->resource, nullptr, &uv, cpu);
  device_.device->CreateUnorderedAccessView(b->resource, nullptr, &uv, local);
  const UINT values[4] = {value, value, value, value};
  cmd->ClearUnorderedAccessViewUint(gpu, local, b->resource, values, 0,
                                    nullptr);
}

void D3D12CommandList::UpdateBuffer(rhi::Buffer* buffer,
                                    u64 offset,
                                    u64 bytes,
                                    const void* data) {
  const UploadAlloc a = AllocateUpload(bytes, 16);
  if (!a.cpu)
    return;
  std::memcpy(a.cpu, data, bytes);
  Require(Buf(buffer), D3D12_RESOURCE_STATE_COPY_DEST);
  FlushBarriers();
  cmd->CopyBufferRegion(Buf(buffer)->resource, offset, a.resource, a.offset,
                        bytes);
}

void D3D12CommandList::ResetTimestamps(rhi::TimestampPool*, u32, u32) {}

// D3D12 has one kind of timestamp, taken when the preceding work is done;
// `start` changes nothing.
void D3D12CommandList::WriteTimestamp(rhi::TimestampPool* pool,
                                      u32 index,
                                      bool /*start*/) {
  auto* p = static_cast<D3D12TimestampPool*>(pool);
  FlushBarriers();
  cmd->EndQuery(p->heap, D3D12_QUERY_TYPE_TIMESTAMP, index);
  cmd->ResolveQueryData(p->heap, D3D12_QUERY_TYPE_TIMESTAMP, index, 1,
                        p->readback, u64(index) * 8);
}

void D3D12CommandList::PushLabel(const char* label) {
  if (device_.debug_labels)
    cmd->BeginEvent(1, label, static_cast<UINT>(std::strlen(label) + 1));
}

void D3D12CommandList::PopLabel() {
  if (device_.debug_labels)
    cmd->EndEvent();
}

void D3D12CommandList::InsertLabel(const char* label) {
  if (device_.debug_labels)
    cmd->SetMarker(1, label, static_cast<UINT>(std::strlen(label) + 1));
}

}  // namespace gpu::d3d12::impl
