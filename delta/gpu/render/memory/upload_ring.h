/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include "base/containers/vector.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct TextureUploadBlock {
  rhi::Buffer* buffer = nullptr;
  u8* map = nullptr;
  u64 capacity = 0;
  u64 offset = 0;
  u64 idle_resets = 0;
};

struct TextureUploadSlice {
  rhi::Buffer* buffer = nullptr;
  u64 offset = 0;
  u8* map = nullptr;
};

constexpr u64 kVbRing = 1024ull * 1024 * 1024;
u64 VbRingBytes();
constexpr u64 kIbRing = 128ull * 1024 * 1024;
constexpr u64 kUboRing = 256ull * 1024 * 1024;
u64 UboRingBytes();
constexpr u32 kCbufWindow = gpu::gcn::kCbufDwords * 4;
constexpr u32 kCbufBindings = gpu::gcn::kMaxCbufBindings;
constexpr u64 kSboRing = 512ull * 1024 * 1024;
constexpr u32 kRawBufWindow = gpu::gcn::kGfxBufferDwords * 4;
constexpr u32 kRawBufBindings = gpu::gcn::kMaxGfxBuffers;
constexpr u64 kLdsScratch =
    u64(gpu::gcn::kLdsWaves) * gpu::gcn::kLdsMaxDwords * 4;

struct BufferBindGroups {
  rhi::BindGroupLayout* layout = nullptr;
  rhi::BindGroup* upload = nullptr;
  rhi::BindGroup* zero = nullptr;
};

struct UploadRings {
  rhi::Buffer* vb = nullptr;
  u8* vb_map = nullptr;
  u64 vb_offset = 0, vb_end = kVbRing;

  rhi::Buffer* ib = nullptr;
  rhi::Buffer* converted_ib = nullptr;
  rhi::Buffer* index_convert_src = nullptr;
  u8* ib_map = nullptr;
  u64 ib_offset = 0, ib_end = kIbRing;

  rhi::Buffer* ubo_buf = nullptr;
  u8* ubo_map = nullptr;
  u64 ubo_bytes = 0;
  u64 ubo_offset = 0, ubo_end = kUboRing;
  u32 ubo_align = 256;
  u64 ubo_stride = kCbufWindow;
  BufferBindGroups cbuf_bindings[kCbufBindings + 1]{};
  rhi::BindGroupLayout* empty_layout = nullptr;
  rhi::Buffer* ubo_zero = nullptr;
  rhi::BindGroupLayout* indirect_cbuf_layout = nullptr;
  rhi::BindGroup* indirect_cbuf_sets[2]{};

  rhi::Buffer* sbo_buf = nullptr;
  u8* sbo_map = nullptr;
  u64 sbo_offset = 0, sbo_end = kSboRing;
  u32 sbo_count = gpu::gcn::kMinGfxBuffers;
  u32 sbo_align = 256;
  u64 sbo_stride = kRawBufWindow;
  base::Vector<u32> sbo_written;
  BufferBindGroups raw_bindings[kRawBufBindings + 1]{};
  rhi::Buffer* sbo_zero = nullptr;

  rhi::Buffer* lds_buf = nullptr;
  rhi::BindGroupLayout* lds_layout = nullptr;
  rhi::BindGroup* lds_set = nullptr;
  u8* lds_map = nullptr;

  base::Vector<TextureUploadBlock> texture_uploads[2];
};

extern UploadRings& g_ring;

bool CreateUploadRings();
bool EnsureVertexRing();
bool EnsureIndexRing();
bool EnsureCbufRing();
bool EnsureCbufZero();
rhi::BindGroupLayout* GetCbufLayout(u32 count);
rhi::BindGroup* GetCbufBindGroup(u32 count);
bool EnsureRawBufferRing();
bool EnsureRawBufferZero();
rhi::BindGroupLayout* GetRawBufferLayout(u32 count);
rhi::BindGroup* GetRawBufferBindGroup(u32 count);
bool EnsureLdsScratch();
bool AllocateTextureUpload(u32 slot,
                           u64 bytes,
                           u64 alignment,
                           TextureUploadSlice& slice);
void ResetTextureUploads(u32 slot);

}  // namespace gpu::render
