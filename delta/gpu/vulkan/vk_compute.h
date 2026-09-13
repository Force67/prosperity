/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// What the rest of the Vulkan backend needs from the compute path.

#include <vulkan/vulkan.h>
#include "base/arch.h"
#include "gpu/gcn/gcn_detile.h"

namespace gpu::vk {

// A draw is about to sample the surface at `base` (`w`x`h` texels, guest
// layout `layout`) through `img`. If a dispatch's output for exactly that
// surface is sitting in a range buffer, copy it VRAM->image on the frame
// command buffer and return true; guest memory is then never involved. `seq`
// is the range revision the image already holds, and a match records nothing.
// A null `img` only asks whether the range could supply it.
bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     VkImage img,
                     VkImageLayout old_layout,
                     u64* seq);

// Destroy range buffers retired two frames ago; called once per BeginFrame.
void ReleaseRetiredCsBuffers();

// A draw is about to read the live render target at `base`, which a dispatch
// wrote through the range aliasing it: copy the range's pixels into the image
// on the frame command buffer. Returns false when no such range exists or the
// shapes disagree (the caller then falls back to the guest-memory flush).
bool CsRefreshRtFromTruth(u64 base);

}  // namespace gpu::vk
