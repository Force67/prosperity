/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// What the rest of the renderer needs from the compute path.

#include "base/arch.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

// A draw is about to sample the surface at `base` (`w`x`h` texels, guest
// layout `layout`) through `img`, which is in `state`. If a dispatch's output
// for exactly that surface is sitting in a range buffer, copy it VRAM->image
// on the frame command list, leave the image in kShaderRead and return true;
// guest memory is then never involved. `seq` is the range revision the image
// already holds, and a match records nothing. A null `img` only asks whether
// the range could supply it.
bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     rhi::Texture* img,
                     rhi::TextureState state,
                     u64* seq);

// The same for a texture inside a raw range a dispatch wrote through a V#:
// the tiling shader detiles the range's bytes (`tiled`) into `linear` first.
bool CsSupplyTextureFromBuffer(u64 base,
                               const gcn::TextureLayout32& tiled,
                               const gcn::TextureLayout32& linear,
                               u32 w,
                               u32 h,
                               rhi::Texture* img,
                               rhi::TextureState state,
                               u64* seq);

// The buffer counterpart. When [base, base+bytes) lies inside one range a
// dispatch wrote and has not written back, CsBufferRevision is its write
// revision (0 otherwise) and CsSupplyBuffer copies it VRAM->`dst`+`dst_off`
// on the frame command list, leaving guest memory stale on purpose: a draw
// reading a dispatch's output then costs no readback and no wait.
u64 CsBufferRevision(u64 base, u64 bytes);
bool CsSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off);
// The same window bound in place as a vertex stream: `*buffer`+`*offset` is
// the range's own buffer, made visible to vertex fetch on the frame list.
// Skinning output read by the next draws then never goes through guest memory.
bool CsVertexBuffer(u64 base, u64 bytes, rhi::Buffer** buffer, u64* offset);
// Whether CsVertexBuffer can serve the window, without binding anything.
bool CsHoldsVertices(u64 base, u64 bytes);

// Destroy range buffers retired two frames ago; called once per BeginFrame.
void ReleaseRetiredCsBuffers();

// A draw is about to read the live render target at `base`, which a dispatch
// wrote through the range aliasing it: copy the range's pixels into the image
// on the frame command buffer. Returns false when no such range exists or the
// shapes disagree (the caller then falls back to the guest-memory flush).
bool CsRefreshRtFromTruth(u64 base);
// The frame went on recording into a fresh command list (SubmitFrameChunk);
// dispatches recorded into the frame list follow it.
void CsFrameListChanged();

// Something other than a dispatch rewrote guest memory at [base, base+bytes):
// ranges staged from it re-read it at their next use. Flush pending dispatch
// writes to it first, or they land over the new bytes.
void CsForgetGuestRange(u64 base, u64 bytes);

// Hands every guest CPU write the write tracker saw since the last call to
// whatever keeps a copy of guest memory: the draw path's staging caches and
// the compute ranges. Call before trusting either.
void CollectGuestWrites();
// The same, at most once a submission: a CPU write inside one cannot be
// ordered against its dispatches anyway.
void CollectGuestWritesForSubmission();

// The guest CPU wrote [first, end). Ranges staged from it re-read it, and a
// range still holding dispatch output keeps those bytes out of its writeback.
void CsNoteGuestWrites(u64 first, u64 end);

// DELTA_GPU_QCHECK: an empty submission through the same queue, waited. A
// failure names the queue work that ran BEFORE this point as the device loss.
bool QueueCheck(const char* where);
bool QueueCheckArmed();

}  // namespace gpu::render
