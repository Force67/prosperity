/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/upload_ring.h"
#include "base/arch.h"

#include "gpu/gpu_check.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <base/logging.h>
#include <options/options.h>
#include <base/algorithm.h>
#include <base/math/value_bounds.h>

namespace {
// DELTA_GPU_LDSDUMP=<dwords>: host-visible shared-LDS scratch, dumped after a
// frame.
DELTA_OPTION(u32, kLdsDump, "DELTA_GPU_LDSDUMP", 0);
DELTA_OPTION(u32, kUboRingMb, "DELTA_GPU_UBORING_MB", 256);
}  // namespace

namespace gpu::render {

u64 UboRingBytes() {
  // Once allocated, the physical buffer determines the capacity. A later
  // option update must not make offsets escape the mapped allocation.
  if (g_ring.ubo_bytes)
    return g_ring.ubo_bytes;
  const u64 requested =
      u64(base::Clamp(kUboRingMb.get(), 16u, 2048u)) * 1024 * 1024;
  const u64 range = Device().caps().max_storage_buffer_range;
  return range ? base::Min(requested, u64(range) * 2) : requested;
}

u64 VbRingBytes() {
  static const u64 bytes = [] {
    const char* e = std::getenv("DELTA_GPU_VBRING_MB");
    const long mb = e ? std::atol(e) : 0;
    if (mb <= 0)
      return kVbRing;
    return static_cast<u64>(mb) * 1024 * 1024;
  }();
  return bytes;
}

namespace {

// A persistently mapped host buffer, or null.
rhi::Buffer* HostBuffer(u64 bytes, u32 usage, const char* name,
                        u8** map) {
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
  g_ring.vb = HostBuffer(VbRingBytes(),
                         rhi::kBufferVertex | rhi::kBufferCopyDst,
                         "vertex ring",
                         &g_ring.vb_map);
  g_ring.ib = HostBuffer(kIbRing, rhi::kBufferIndex, "index ring",
                         &g_ring.ib_map);
  if (!g_ring.vb || !g_ring.ib)
    return false;
  // Recomp cbuffer ring + dynamic-UBO descriptors (set 1) + empty set-0 layout.
  g_ring.ubo_align = base::Max(caps.uniform_offset_alignment, 1u);
  if (caps.max_dynamic_uniform_buffers < kCbufBindings)
    BASE_LOGI("gpuvk", "only {}/{} dynamic UBOs available; set 1 is an "
                       "out-of-spec layout on this device and a cbuffer may "
                       "silently read zero",
              caps.max_dynamic_uniform_buffers, kCbufBindings);
  g_ring.ubo_stride = (kCbufWindow + g_ring.ubo_align - 1) &
                      ~(u64)(g_ring.ubo_align - 1);
  // kMaxCbufBindings, not 8: a shader pair whose constant buffers exceed the
  // cap is planned only up to it, and every s_buffer_load from a dropped base
  // emits nothing, leaving its destination SGPRs zero. Skyrim's UI shaders
  // sit right at 8 cbufs, so their transform matrix read back as an all-zero
  // matrix and collapsed every vertex position.
  rhi::BindGroupLayoutDesc ubo;
  for (u32 i = 0; i < kCbufBindings; i++)
    ubo.bindings.push_back(
        {i, rhi::BindingType::kUniformBufferDynamic, GraphicsStages()});
  g_ring.ubo_layout = Device().CreateBindGroupLayout(ubo);
  g_ring.empty_layout = Device().CreateBindGroupLayout({});
  if (!g_ring.ubo_layout || !g_ring.empty_layout)
    return false;
  // Mesh stages can need more windows than the dynamic UBO limit allows.
  // Bind the current frame's half of the existing ring as one read-only
  // storage buffer, plus one dynamic UBO containing the window offsets.
  if (caps.max_storage_buffer_range >= UboRingBytes() / 2) {
    rhi::BindGroupLayoutDesc indirect;
    indirect.bindings.push_back(
        {0, rhi::BindingType::kStorageBuffer, GraphicsStages(), true});
    indirect.bindings.push_back(
        {1, rhi::BindingType::kUniformBufferDynamic, GraphicsStages()});
    g_ring.indirect_cbuf_layout = Device().CreateBindGroupLayout(indirect);
  }

  // Raw-buffer set layout (set 2). Every recompiled pipeline layout that has a
  // shader reading buffers by hand names it, so it exists from the start; the
  // ring behind it is allocated only if such a shader actually appears.
  g_ring.sbo_align = base::Max(caps.storage_offset_alignment, 4u);
  // These are DYNAMIC storage buffers, whose device limit has a floor of 4
  // while real desktop parts report 16+. Take what the device offers (up to
  // our compile-time ceiling) and tell the recompiler, so a shader
  // referencing more raw buffers than 4 is planned rather than declined
  // wherever the hardware can carry it.
  g_ring.sbo_count =
      base::Min<u32>(caps.max_dynamic_storage_buffers, kRawBufBindings);
  if (g_ring.sbo_count < gpu::gcn::kMinGfxBuffers) {
    BASE_LOGI("gpuvk", "only {} dynamic storage buffers available, below "
                       "the floor of {}: shaders reading raw buffers will "
                       "decline",
              g_ring.sbo_count, gpu::gcn::kMinGfxBuffers);
    g_ring.sbo_count = gpu::gcn::kMinGfxBuffers;
  }
  gpu::gcn::SetMaxGfxBuffers(g_ring.sbo_count);
  // Whether the push range can also carry each stage's code address (2x8
  // bytes after the 128 bytes of user data): with it, no graphics module
  // ever bakes its own address and the shader cache keys by content.
  gpu::gcn::SetPushBudget(caps.max_push_constant_bytes);
  // A wave64 shader may rely on lockstep the host only gives us within one
  // subgroup, so the recompiler has to know how wide this device's is.
  gpu::gcn::SetHostSubgroupSize(caps.subgroup_size);
  BASE_LOGI("gpuvk", "subgroup: {} lanes{}", caps.subgroup_size,
            gpu::gcn::WaveSplitsAcrossSubgroups()
                ? " (a GCN wave spans several)"
                : "");
  g_ring.sbo_stride = (kRawBufWindow + g_ring.sbo_align - 1) &
                      ~(u64)(g_ring.sbo_align - 1);
  rhi::BindGroupLayoutDesc sbo;
  for (u32 i = 0; i < g_ring.sbo_count; i++)
    sbo.bindings.push_back(
        {i, rhi::BindingType::kStorageBufferDynamic,
         rhi::kStageVertex | rhi::kStageFragment |
             (caps.mesh_shader ? rhi::kStageMesh : 0u)});
  g_ring.sbo_layout = Device().CreateBindGroupLayout(sbo);
  return g_ring.sbo_layout != nullptr;
}

// The device is initialized before guest mappings exist, but the title profile
// is loaded later. Allocate constants at the first frame so its budget applies
// to both the allocation and the offsets used to address it.
bool EnsureCbufRing() {
  if (g_ring.ubo_buf)
    return g_ring.ubo_map != nullptr;
  g_ring.ubo_bytes = UboRingBytes();
  g_ring.ubo_buf =
      HostBuffer(UboRingBytes(),
                 rhi::kBufferUniform | rhi::kBufferStorage |
                     rhi::kBufferCopyDst,
                 "cbuffer ring", &g_ring.ubo_map);
  if (!g_ring.ubo_buf)
    return false;
  std::memset(g_ring.ubo_map, 0, UboRingBytes());
  g_ring.ubo_stride = (kCbufWindow + g_ring.ubo_align - 1) &
                      ~(u64)(g_ring.ubo_align - 1);

  rhi::BindGroupDesc set;
  set.layout = g_ring.ubo_layout;
  for (u32 i = 0; i < kCbufBindings; i++) {
    rhi::BindingWrite w;
    w.binding = i;
    w.buffer = g_ring.ubo_buf;
    w.range = kCbufWindow;
    set.writes.push_back(w);
  }
  g_ring.ubo_set = Device().CreateBindGroup(set);
  if (!g_ring.ubo_set)
    return false;
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

// One storage buffer at set 3 binding 0, big enough for kLdsWaves blocks. It is
// never uploaded to and never read back: the shader writes its wave's block and
// reads it again within the same draw.
// DELTA_GPU_LDSDUMP=<dwords> makes it host visible and mapped instead, so what
// an NGG shader staged through LDS can be read after the frame, the only way
// to tell "the write never happened" from "the read was at another address".
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
  desc.usage = rhi::kBufferStorage | rhi::kBufferCopyDst;
  desc.memory = kLdsDump ? rhi::MemoryKind::kReadback : rhi::MemoryKind::kDevice;
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

bool EnsureRawBufferRing() {
  if (g_ring.sbo_map)
    return true;
  if (!g_ring.sbo_layout)
    return false;
  g_ring.sbo_buf = HostBuffer(kSboRing,
                              rhi::kBufferStorage | rhi::kBufferCopyDst,
                              "raw buffer ring",
                              &g_ring.sbo_map);
  if (!g_ring.sbo_buf)
    return false;
  std::memset(g_ring.sbo_map, 0, kSboRing);
  g_ring.sbo_written.assign(
      static_cast<size_t>((kSboRing + g_ring.sbo_stride - 1) /
                          g_ring.sbo_stride),
      0);
  rhi::BindGroupDesc set;
  set.layout = g_ring.sbo_layout;
  for (u32 i = 0; i < g_ring.sbo_count; i++) {
    rhi::BindingWrite w;
    w.binding = i;
    w.buffer = g_ring.sbo_buf;
    w.range = kRawBufWindow;
    set.writes.push_back(w);
  }
  g_ring.sbo_set = Device().CreateBindGroup(set);
  if (!g_ring.sbo_set)
    return false;
  BASE_LOGI("gpuvk", "raw-buffer ring: {} MB, {} KB windows, {} bindings",
            (unsigned long long)(kSboRing >> 20), kRawBufWindow >> 10,
            g_ring.sbo_count);
  return true;
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
    const u64 aligned =
        (block.offset + alignment - 1) & ~(alignment - 1);
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
  block.buffer = HostBuffer(capacity, rhi::kBufferCopySrc, "texture upload",
                            &block.map);
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
  // Reset runs in BeginFrame after this slot's submission retired, so no
  // in-flight transfer references these blocks, which is the one point where
  // destroying one is safe. Blocks grow on demand (a loading burst can leave
  // hundreds of idle megabytes behind), so drop any block that has sat unused
  // for ~10s of resets; the next burst simply recreates it.
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
