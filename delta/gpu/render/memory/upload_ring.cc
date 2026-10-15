/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/memory/upload_ring.h"
#include "base/arch.h"

#include "gpu/gpu_check.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/render_target.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "options/options.h"

namespace {
DELTA_OPTION(u32, kLdsDump, "DELTA_GPU_LDSDUMP", 0);
DELTA_OPTION(u32, kUboRingMb, "DELTA_GPU_UBORING_MB", 256);
}  // namespace

namespace gpu::render {

u64 UboRingBytes() {
  if (g_ring.ubo_bytes)
    return g_ring.ubo_bytes;
  const u64 requested =
      u64(base::Clamp(kUboRingMb.get(), 16u, 2048u)) * 1024 * 1024;
  const u64 range = Device().caps().max_storage_buffer_range;
  return range ? base::Min(requested, u64(range) * 2) : requested;
}

u64 VbRingBytes() {
  static const u64 kBytes = [] {
    const char* e = std::getenv("DELTA_GPU_VBRING_MB");
    const long mb = e ? std::atol(e) : 0;
    if (mb <= 0)
      return kVbRing;
    return static_cast<u64>(mb) * 1024 * 1024;
  }();
  return kBytes;
}

namespace {

rhi::Buffer* HostBuffer(u64 bytes, u32 usage, const char* name, u8** map) {
  rhi::BufferDesc desc;
  desc.size = bytes;
  desc.usage = usage;
  desc.memory = rhi::MemoryKind::kUpload;
  desc.name = name;
  rhi::Buffer* buffer = Device().CreateBuffer(desc);
  *map = buffer ? buffer->mapped() : nullptr;
  if (buffer && !*map) {
    Device().Destroy(buffer);
    return nullptr;
  }
  return buffer;
}

u32 GraphicsStages() {
  const rhi::Caps& caps = Device().caps();
  return rhi::kStageVertex | rhi::kStageFragment |
         (caps.geometry_shader ? rhi::kStageGeometry : 0u) |
         (caps.mesh_shader ? rhi::kStageMesh : 0u);
}

}  // namespace

bool CreateUploadRings() {
  const rhi::Caps& caps = Device().caps();
  g_ring.ubo_align = base::Max(caps.uniform_offset_alignment, 1u);
  g_ring.ubo_stride =
      (kCbufWindow + g_ring.ubo_align - 1) & ~(u64)(g_ring.ubo_align - 1);
  g_ring.empty_layout = Device().CreateBindGroupLayout({});
  if (!g_ring.empty_layout)
    return false;
  if (caps.max_storage_buffer_range >= UboRingBytes() / 2) {
    rhi::BindGroupLayoutDesc indirect;
    indirect.bindings.push_back(
        {0, rhi::BindingType::kStorageBuffer, GraphicsStages(), true});
    indirect.bindings.push_back(
        {1, rhi::BindingType::kUniformBufferDynamic, GraphicsStages()});
    g_ring.indirect_cbuf_layout = Device().CreateBindGroupLayout(indirect);
  }

  g_ring.sbo_align = base::Max(caps.storage_offset_alignment, 4u);
  g_ring.sbo_count =
      base::Min<u32>(caps.max_dynamic_storage_buffers, kRawBufBindings);
  if (g_ring.sbo_count < gpu::gcn::kMinGfxBuffers) {
    BASE_LOGI("gpuvk",
              "only {} dynamic storage buffers available, below "
              "the floor of {}: shaders reading raw buffers will "
              "decline",
              g_ring.sbo_count, gpu::gcn::kMinGfxBuffers);
    g_ring.sbo_count = gpu::gcn::kMinGfxBuffers;
  }
  gpu::gcn::SetMaxGfxBuffers(g_ring.sbo_count);
  gpu::gcn::SetPushBudget(caps.max_push_constant_bytes);
  gpu::gcn::SetHostSubgroupSize(caps.subgroup_size);
  BASE_LOGI("gpuvk", "subgroup: {} lanes{}", caps.subgroup_size,
            gpu::gcn::WaveSplitsAcrossSubgroups()
                ? " (a GCN wave spans several)"
                : "");
  g_ring.sbo_stride =
      (kRawBufWindow + g_ring.sbo_align - 1) & ~(u64)(g_ring.sbo_align - 1);
  return true;
}

bool EnsureVertexRing() {
  if (!g_ring.vb)
    g_ring.vb =
        HostBuffer(VbRingBytes(), rhi::kBufferVertex | rhi::kBufferCopyDst,
                   "vertex ring", &g_ring.vb_map);
  return g_ring.vb_map != nullptr;
}

bool EnsureIndexRing() {
  if (!g_ring.ib)
    g_ring.ib =
        HostBuffer(kIbRing, rhi::kBufferIndex, "index ring", &g_ring.ib_map);
  return g_ring.ib_map != nullptr;
}

bool EnsureCbufRing() {
  if (g_ring.ubo_buf)
    return g_ring.ubo_map != nullptr;
  g_ring.ubo_bytes = UboRingBytes();
  g_ring.ubo_buf = HostBuffer(
      UboRingBytes(),
      rhi::kBufferUniform | rhi::kBufferStorage | rhi::kBufferCopyDst,
      "cbuffer ring", &g_ring.ubo_map);
  if (!g_ring.ubo_buf)
    return false;
  std::memset(g_ring.ubo_map, 0, UboRingBytes());
  g_ring.ubo_stride =
      (kCbufWindow + g_ring.ubo_align - 1) & ~(u64)(g_ring.ubo_align - 1);

  if (g_ring.indirect_cbuf_layout) {
    for (u32 slot = 0; slot < 2; ++slot) {
      rhi::BindGroupDesc indirect;
      indirect.layout = g_ring.indirect_cbuf_layout;
      rhi::BindingWrite table;
      table.binding = 0;
      table.buffer = g_ring.ubo_buf;
      table.offset = slot * (UboRingBytes() / 2);
      table.range = UboRingBytes() / 2;
      rhi::BindingWrite offsets;
      offsets.binding = 1;
      offsets.buffer = g_ring.ubo_buf;
      offsets.range = gpu::gcn::kIndirectDrawDwords * sizeof(u32);
      indirect.writes = {table, offsets};
      g_ring.indirect_cbuf_sets[slot] = Device().CreateBindGroup(indirect);
    }
  }
  return true;
}

bool EnsureLdsScratch() {
  if (g_ring.lds_set)
    return true;
  rhi::BindGroupLayoutDesc layout;
  layout.bindings.push_back({0, rhi::BindingType::kStorageBuffer,
                             rhi::kStageVertex | rhi::kStageFragment});
  g_ring.lds_layout = Device().CreateBindGroupLayout(layout);
  if (!g_ring.lds_layout)
    return false;
  rhi::BufferDesc desc;
  desc.size = kLdsScratch;
  desc.usage = rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc;
  desc.memory =
      kLdsDump ? rhi::MemoryKind::kReadback : rhi::MemoryKind::kDevice;
  desc.name = "shared lds";
  g_ring.lds_buf = Device().CreateBuffer(desc);
  if (!g_ring.lds_buf)
    return false;
  if (kLdsDump)
    g_ring.lds_map = g_ring.lds_buf->mapped();
  rhi::BindGroupDesc set;
  set.layout = g_ring.lds_layout;
  rhi::BindingWrite w;
  w.buffer = g_ring.lds_buf;
  w.range = kLdsScratch;
  set.writes.push_back(w);
  g_ring.lds_set = Device().CreateBindGroup(set);
  if (!g_ring.lds_set)
    return false;
  BASE_LOGI("gpuvk", "shared LDS scratch: {} MB ({} waves)",
            (unsigned long long)(kLdsScratch >> 20), gpu::gcn::kLdsWaves);
  return true;
}

rhi::BindGroupLayout* GetCbufLayout(u32 count) {
  if (!count || count > kCbufBindings)
    return nullptr;
  auto& groups = g_ring.cbuf_bindings[count];
  if (!groups.layout) {
    const u32 limit = Device().caps().max_dynamic_uniform_buffers;
    if (count > limit)
      BASE_LOGI("gpuvk", "shader needs {} dynamic UBOs, device supports {}",
                count, limit);
    rhi::BindGroupLayoutDesc desc;
    for (u32 i = 0; i < count; ++i)
      desc.bindings.push_back(
          {i, rhi::BindingType::kUniformBufferDynamic, GraphicsStages()});
    groups.layout = Device().CreateBindGroupLayout(desc);
  }
  return groups.layout;
}

rhi::BindGroup* GetCbufBindGroup(u32 count) {
  auto* layout = GetCbufLayout(count);
  if (!layout)
    return nullptr;
  auto& groups = g_ring.cbuf_bindings[count];
  auto*& set = g_ring.ubo_buf ? groups.upload : groups.zero;
  if (set)
    return set;
  rhi::Buffer* buffer = g_ring.ubo_buf ? g_ring.ubo_buf : g_ring.ubo_zero;
  if (!buffer)
    return nullptr;
  rhi::BindGroupDesc desc;
  desc.layout = layout;
  for (u32 i = 0; i < count; ++i) {
    rhi::BindingWrite write;
    write.binding = i;
    write.buffer = buffer;
    write.range = kCbufWindow;
    desc.writes.push_back(write);
  }
  set = Device().CreateBindGroup(desc);
  return set;
}

bool EnsureCbufZero() {
  if (g_ring.ubo_zero)
    return true;
  if (!g_frame.recording)
    return false;
  rhi::BufferDesc desc;
  desc.size = kCbufWindow;
  desc.usage = rhi::kBufferUniform | rhi::kBufferStorage | rhi::kBufferCopyDst;
  desc.name = "zero constant buffer";
  g_ring.ubo_zero = Device().CreateBuffer(desc);
  if (!g_ring.ubo_zero)
    return false;
  EndRegion();
  g_frame.list->FillBuffer(g_ring.ubo_zero, 0, kCbufWindow, 0);
  g_frame.list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessUniformRead);
  return true;
}

bool EnsureRawBufferZero() {
  if (g_ring.sbo_zero)
    return true;
  if (!g_frame.recording)
    return false;
  rhi::BufferDesc desc;
  desc.size = kRawBufWindow;
  desc.usage = rhi::kBufferStorage | rhi::kBufferCopyDst;
  desc.name = "zero raw buffer";
  g_ring.sbo_zero = Device().CreateBuffer(desc);
  if (!g_ring.sbo_zero)
    return false;
  EndRegion();
  g_frame.list->FillBuffer(g_ring.sbo_zero, 0, kRawBufWindow, 0);
  g_frame.list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessShaderRead);
  return true;
}

bool EnsureRawBufferRing() {
  if (g_ring.sbo_map)
    return true;
  g_ring.sbo_buf = HostBuffer(
      kSboRing, rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc,
      "raw buffer ring", &g_ring.sbo_map);
  if (!g_ring.sbo_buf)
    return false;
  std::memset(g_ring.sbo_map, 0, kSboRing);
  g_ring.sbo_written.assign(
      static_cast<size_t>((kSboRing + g_ring.sbo_stride - 1) /
                          g_ring.sbo_stride),
      0);
  BASE_LOGI("gpuvk", "raw-buffer ring: {} MB, {} KB windows",
            (unsigned long long)(kSboRing >> 20), kRawBufWindow >> 10);
  return true;
}

rhi::BindGroupLayout* GetRawBufferLayout(u32 count) {
  if (!count || count > g_ring.sbo_count)
    return nullptr;
  auto& groups = g_ring.raw_bindings[count];
  if (!groups.layout) {
    const auto& caps = Device().caps();
    rhi::BindGroupLayoutDesc desc;
    for (u32 i = 0; i < count; ++i)
      desc.bindings.push_back({i, rhi::BindingType::kStorageBufferDynamic,
                               rhi::kStageVertex | rhi::kStageFragment |
                                   (caps.mesh_shader ? rhi::kStageMesh : 0u),
                               true});
    groups.layout = Device().CreateBindGroupLayout(desc);
  }
  return groups.layout;
}

rhi::BindGroup* GetRawBufferBindGroup(u32 count) {
  auto* layout = GetRawBufferLayout(count);
  if (!layout)
    return nullptr;
  auto& groups = g_ring.raw_bindings[count];
  auto*& set = g_ring.sbo_buf ? groups.upload : groups.zero;
  if (set)
    return set;
  rhi::Buffer* buffer = g_ring.sbo_buf ? g_ring.sbo_buf : g_ring.sbo_zero;
  if (!buffer)
    return nullptr;
  rhi::BindGroupDesc desc;
  desc.layout = layout;
  for (u32 i = 0; i < count; ++i) {
    rhi::BindingWrite write;
    write.binding = i;
    write.buffer = buffer;
    write.range = kRawBufWindow;
    desc.writes.push_back(write);
  }
  set = Device().CreateBindGroup(desc);
  return set;
}

bool AllocateTextureUpload(u32 slot,
                           u64 bytes,
                           u64 alignment,
                           TextureUploadSlice& slice) {
  GPU_BUGCHECK(slot < 2, "slot %u is not a frame-ring slot", slot);
  if (!bytes)
    return false;
  alignment = base::Max<u64>(alignment, 4);
  GPU_BUGCHECK((alignment & (alignment - 1)) == 0,
               "alignment %llu is not a power of two",
               (unsigned long long)alignment);
  auto& blocks = g_ring.texture_uploads[slot];
  for (auto& block : blocks) {
    const u64 aligned = (block.offset + alignment - 1) & ~(alignment - 1);
    if (aligned <= block.capacity && bytes <= block.capacity - aligned) {
      slice = {block.buffer, aligned, block.map + aligned};
      block.offset = aligned + bytes;
      return true;
    }
  }

  constexpr u64 kInitialCapacity = 16ull * 1024 * 1024;
  u64 capacity = kInitialCapacity;
  while (capacity < bytes) {
    if (capacity > std::numeric_limits<u64>::max() / 2)
      return false;
    capacity *= 2;
  }
  TextureUploadBlock block;
  block.buffer =
      HostBuffer(capacity, rhi::kBufferCopySrc, "texture upload", &block.map);
  if (!block.buffer)
    return false;
  block.capacity = capacity;
  block.offset = bytes;
  slice = {block.buffer, 0, block.map};
  blocks.push_back(block);
  return true;
}

void ResetTextureUploads(u32 slot) {
  GPU_BUGCHECK(slot < 2, "slot %u is not a frame-ring slot", slot);
  constexpr u64 kIdleResetsBeforeTrim = 300;
  auto& blocks = g_ring.texture_uploads[slot];
  for (size_t i = 0; i < blocks.size();) {
    TextureUploadBlock& block = blocks[i];
    block.idle_resets = block.offset ? 0 : block.idle_resets + 1;
    block.offset = 0;
    if (block.idle_resets < kIdleResetsBeforeTrim) {
      ++i;
      continue;
    }
    Device().Destroy(block.buffer);
    blocks.erase(blocks.begin() + i);
  }
}

}  // namespace gpu::render
