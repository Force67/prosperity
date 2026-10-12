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
// the range could supply it. For compressed images, layout uses blocks while
// w and h remain pixel dimensions. With supplied_mips, a matching prefix of
// the mip chain is allowed; the caller must supply the remaining levels.
bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     rhi::Texture* img,
                     rhi::TextureState state,
                     u64* seq,
                     bool compressed = false,
                     u32* supplied_mips = nullptr);

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

// A texture uploaded from guest memory: the GPU copies its bytes out of the
// guest pages, detiles them when tiled, and copies mips `first_mip` on into
// `img` (left in kShaderRead). A volume's slices are the layout's layers. False
// when a tiled layout has no GPU tiling table; the caller detiles on the CPU
// then.
bool DetileTextureOnGpu(u64 base,
                        const gcn::TextureLayout32& tiled,
                        u32 texel_w,
                        u32 texel_h,
                        rhi::Texture* img,
                        rhi::TextureState state,
                        bool is_3d = false,
                        u32 first_mip = 0);

// `texture`'s first mip tiled into guest memory at `base` on the GPU, behind
// the draws recorded so far. `state` is the texture's layout on the frame's
// list, kept. False when the layout has no GPU tiling table.
bool WriteTextureToGuest(rhi::Texture* texture,
                         rhi::TextureState& state,
                         u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear);

// The fast-clear state of the CMASK (`htile` false) or HTILE word at `address`
// when a dispatch wrote it and its output is still on the GPU: the GPU turns
// the word into a predicate for BeginConditional, then marks `expand` bytes
// from `address` expanded (0xFF), as the CPU does with a CMASK it read (0
// leaves the word alone). False
// when the word is not there or the device has no predicates.
bool CsClearPredicate(u64 address,
                      u64 expand,
                      bool htile,
                      rhi::Buffer** predicate,
                      u64* offset);

// The buffer counterpart. When [base, base+bytes) lies inside one range a
// dispatch wrote and has not written back, CsBufferRevision is its write
// revision (0 otherwise) and CsSupplyBuffer copies it VRAM->`dst`+`dst_off`
// on the frame command list, leaving guest memory stale on purpose: a draw
// reading a dispatch's output then costs no readback and no wait.
u64 CsBufferRevision(u64 base, u64 bytes);
bool CsSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off);
// The same window bound in place as a vertex or index stream:
// `*buffer`+`*offset` is the range's own buffer, made visible to geometry fetch
// on the frame list. Skinning output read by the next draws then never goes
// through guest memory.
bool CsGeometryBuffer(u64 base, u64 bytes, rhi::Buffer** buffer, u64* offset);
// Bind a raw shader window in place, with the same lifetime and ordering.
bool CsStorageBuffer(u64 base, u64 bytes, rhi::Buffer** buffer, u64* offset);
// Whether CsGeometryBuffer can serve the window, without binding anything.
bool CsHoldsGeometry(u64 base, u64 bytes);

// The frame's list while one records, else the compute batch, which goes on
// the queue ahead of the next frame. Null when neither can open.
rhi::CommandList* FrameOrBatchList();

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
// Put a compute batch opened between frames on the queue, ahead of the frame
// about to begin: it may read targets the frame renders into.
void CsSubmitBatchBeforeFrame();

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

// The buffer and offset the GPU reads GpuIndirectArgs' arguments at.
rhi::Buffer* IndirectArgsBuffer(u64 args, u64* offset);
// While set, the draw call itself runs only when the staged guest flag is
// non-zero (BeginGuestPredicate); null otherwise.
rhi::Buffer* GuestPredicate(u64* offset);

// Indirect arguments anywhere in guest memory, copied on `list` by the GPU
// into a buffer the draw or dispatch then reads at `offset`; behind any GPU
// write on its way to them. Null when they cannot be staged.
rhi::Buffer* StageIndirectArgs(rhi::CommandList* list,
                               u64 args,
                               u64 bytes,
                               u64* offset);

}  // namespace gpu::render
