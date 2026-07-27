#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GPU command processor: consumes a PM4 draw command buffer (dcb), tracks the
 * Liverpool register state, and drives the Vulkan renderer on each draw. Entry
 * point for the Gnm HLE submit path.
 */

#include <cstdint>

namespace gpu {

// Process one PM4 draw command buffer (guest GPU address, identity-mapped to a
// host pointer; size in bytes). Walks the packet stream, updating register
// state and issuing draws. Safe to call from the guest's GPU submit thread.
void SubmitDcb(const void* dcb, uint32_t size_bytes);

// Process one PM4 constant command buffer (ccb). The Constant Engine fills its
// on-chip CE RAM (WRITE/LOAD_CONST_RAM) and dumps it to guest memory
// (DUMP_CONST_RAM) where shaders read it as constant buffers. The DE-only path
// ignored this; draws whose cbuffers come via the CE then read stale memory.
// Must run before the matching SubmitDcb (the CE runs ahead of the draw
// engine).
void SubmitCcb(const void* ccb, uint32_t size_bytes);

// End the current frame and present the render target at `scanout_base` (the
// videoout flip buffer). Called by the Gnm submit-and-flip HLE.
void EndFrame(uint64_t scanout_base);

}  // namespace gpu
