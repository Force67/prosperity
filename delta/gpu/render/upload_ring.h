/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Per-frame host-visible upload rings. Every draw's vertices, indices and
// constant-buffer windows are copied into these mapped rings and bound from
// there; each frame slot owns one half, so the in-flight frame's data is never
// overwritten (see vk_frame).

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
  // Consecutive ResetTextureUploads calls with no allocation from this block;
  // long-idle blocks are destroyed at reset (see ResetTextureUploads).
  u64 idle_resets = 0;
};

struct TextureUploadSlice {
  rhi::Buffer* buffer = nullptr;
  u64 offset = 0;
  u8* map = nullptr;
};

// Per-frame vertex ring. A 1080p title that re-draws its whole scene for a
// depth prepass and again for the G-buffer needs far more than a token
// allocation: SotC declined 83 draws a frame (its entire world) against a
// 16 MiB ring, and 1 against 384 MiB. Sized for that, since the failure mode is
// silent deletion of geometry rather than a stall. 512 MiB carried SotC's
// title screen; its attract demo's heaviest cuts issue 11.6k draws a frame and
// still hit the 256 MiB half with the per-frame dedupe active (RINGHWM peak ==
// cap, true need unknown). Doubled so the demo's worst frame fits.
constexpr u64 kVbRing = 1024ull * 1024 * 1024;
// DELTA_GPU_VBRING_MB=<n>: override the vertex ring size (default kVbRing).
// Halved per frame slot like every other ring, so the usable per-frame budget
// is n/2 MB. Exists because the ring is a hard per-frame draw budget, not a
// cache: a draw whose vertices do not fit is declined outright (kRing) and
// simply never appears, which is invisible in a draw count. Read once.
u64 VbRingBytes();
// IB and UBO sized from SotC's attract demo (RINGHWM): the IB half peaked at
// 19.7 of 32 MiB before the per-frame staging dedupe, and the UBO half was
// FULL at ~130 draws (16 KiB window x 16 bindings per draw) in frames that
// wanted 2500+. Dedupe removes the repeats; the raised caps buy headroom for
// what remains unique, at host-visible memory cost only.
constexpr u64 kIbRing =
    128ull * 1024 * 1024;  // per-frame index ring (32-bit), see kVbRing
constexpr u64 kUboRing = 256ull * 1024 * 1024;  // per-frame recomp cbuffer ring
// DELTA_GPU_UBORING_MB changes the total budget (half per frame slot).
// Heavy scenes can require more unique constant windows than the default.
// Latched on the first frame, after the title profile has loaded.
u64 UboRingBytes();
constexpr u32 kCbufWindow = gpu::gcn::kCbufDwords * 4;
constexpr u32 kCbufBindings = gpu::gcn::kMaxCbufBindings;  // set-1 UBO bindings
// Raw-buffer ring: the windows a recompiled shader's hand-written MUBUF loads
// read from, at set-2 bindings 0..kRawBufBindings-1; the device's dynamic
// storage buffer limit determines how many bindings are available.
// The window is a compromise the shader knows about: a MUBUF index has no
// static bound, so a resource larger than this is staged only up to here and
// the recompiled shader clamps, reading the window's last dword past it. Small
// resources reserve only their aligned payload, and repeated buffers share an
// upload. Each frame owns half the ring; its resources must fit within that
// half.
constexpr u64 kSboRing = 512ull * 1024 * 1024;
constexpr u32 kRawBufWindow = gpu::gcn::kGfxBufferDwords * 4;
constexpr u32 kRawBufBindings = gpu::gcn::kMaxGfxBuffers;
// Shared-LDS scratch: kLdsWaves blocks of the recompiler's LDS cap.
constexpr u64 kLdsScratch =
    u64(gpu::gcn::kLdsWaves) * gpu::gcn::kLdsMaxDwords * 4;

struct UploadRings {
  // Vertex ring: interleaved pos+colour+uv for the heuristic path, the raw
  // guest vertex records for the recompiled path.
  rhi::Buffer* vb = nullptr;
  u8* vb_map = nullptr;
  u64 vb_offset = 0, vb_end = kVbRing;

  // Index ring: 32-bit indices (16-bit guest indices are widened on upload).
  rhi::Buffer* ib = nullptr;
  u8* ib_map = nullptr;
  u64 ib_offset = 0, ib_end = kIbRing;

  // Recomp cbuffer ring: per-draw VS/PS constant buffers live at set 1 bindings
  // 0..kCbufBindings-1, each addressed by a dynamic offset into this ring.
  // empty_layout fills set 0 for untextured recomp draws.
  rhi::Buffer* ubo_buf = nullptr;
  u8* ubo_map = nullptr;
  u64 ubo_bytes = 0;  // latched when allocated after title settings load
  u64 ubo_offset = 0, ubo_end = kUboRing;
  u32 ubo_align = 256;
  u64 ubo_stride = kCbufWindow;
  bool zero_window_initialized = false;
  rhi::BindGroupLayout* ubo_layout = nullptr;
  rhi::BindGroupLayout* empty_layout = nullptr;
  rhi::BindGroup* ubo_set = nullptr;
  rhi::BindGroupLayout* indirect_cbuf_layout = nullptr;
  rhi::BindGroup* indirect_cbuf_sets[2]{};

  // Raw-buffer ring: same shape as the cbuffer ring (fixed windows selected by
  // a dynamic offset), but storage buffers, because a MUBUF address is a
  // per-lane index rather than a uniform offset. Window 0 stays zero and is
  // what an unresolved binding points at.
  rhi::Buffer* sbo_buf = nullptr;
  u8* sbo_map = nullptr;
  u64 sbo_offset = 0, sbo_end = kSboRing;
  // Bindings actually created, = min(device dynamic-SSBO limit,
  // kRawBufBindings). The recompiler is told this via gcn::SetMaxGfxBuffers so
  // it never plans a binding the layout does not have.
  u32 sbo_count = gpu::gcn::kMinGfxBuffers;
  u32 sbo_align = 256;
  u64 sbo_stride = kRawBufWindow;
  base::Vector<u32> sbo_written;
  rhi::BindGroupLayout* sbo_layout = nullptr;
  rhi::BindGroup* sbo_set = nullptr;

  // Shared LDS (set 3): one device-local block per wave for a graphics stage
  // that stages through LDS. Nothing reads it back on the CPU and nothing
  // survives a draw, so it is plain device memory written and read by the
  // shader alone.
  rhi::Buffer* lds_buf = nullptr;
  rhi::BindGroupLayout* lds_layout = nullptr;
  rhi::BindGroup* lds_set = nullptr;
  u8* lds_map = nullptr;  // only under DELTA_GPU_LDSDUMP

  // Texture uploads are recorded into the active frame command buffer. Each
  // frame slot owns its blocks so an in-flight transfer is never overwritten.
  base::Vector<TextureUploadBlock> texture_uploads[2];
};

extern UploadRings& g_ring;

bool CreateUploadRings();
bool EnsureCbufRing();
// Allocate the raw-buffer ring + its descriptor set. Deferred to the first
// draw that needs one: the ring is large, and a title whose vertex fetches the
// vertex-input state already covers never binds set 2 at all. The set layout
// itself is created up front, because pipeline layouts name it.
bool EnsureRawBufferRing();
// The shared-LDS scratch buffer and its set-3 descriptor, created on first use
// by a shader that declares LDS in a graphics stage.
bool EnsureLdsScratch();
bool AllocateTextureUpload(u32 slot,
                           u64 bytes,
                           u64 alignment,
                           TextureUploadSlice& slice);
void ResetTextureUploads(u32 slot);

}  // namespace gpu::render
