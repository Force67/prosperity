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

// Wall time spent walking command buffers (the whole PM4 walk, not just the
// parts the renderer times), and the number of submissions, since the last FPS
// report reset them. Separates "the guest is busy" from "we are busy running
// its command stream".
extern uint64_t g_ns_dcb;
extern uint32_t g_dcb_n;

// Select the PS4 shader hardware profile for subsequent submissions. The
// composition root calls this after combining configured Neo hardware with the
// title's param.sfo Neo-support bit.
void SetPs4NeoMode(bool enabled);

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
