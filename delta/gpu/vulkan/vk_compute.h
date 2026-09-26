/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// What the rest of the renderer needs from the compute path.

#include "base/arch.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/rhi/device.h"

namespace gpu::vk {

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

// Destroy range buffers retired two frames ago; called once per BeginFrame.
void ReleaseRetiredCsBuffers();

// A draw is about to read the live render target at `base`, which a dispatch
// wrote through the range aliasing it: copy the range's pixels into the image
// on the frame command buffer. Returns false when no such range exists or the
// shapes disagree (the caller then falls back to the guest-memory flush).
bool CsRefreshRtFromTruth(u64 base);

// Something other than a dispatch rewrote guest memory at [base, base+bytes):
// ranges staged from it re-read it at their next use. Flush pending dispatch
// writes to it first, or they land over the new bytes.
void CsForgetGuestRange(u64 base, u64 bytes);

// DELTA_GPU_QCHECK: an empty submission through the same queue, waited. A
// failure names the queue work that ran BEFORE this point as the device loss.
bool QueueCheck(const char* where);
bool QueueCheckArmed();

}  // namespace gpu::vk
