#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The PS5 GPU command processor: consumes the AGC command buffers libSceAgc
 * submits (through the kernel's /dev/gc doorbell ring), tracks the gfx10.3
 * register state they program, and drives the renderer. This header is its
 * whole surface; the decode itself is split by decision across ps5/ (reg_state,
 * draw_state, compute_dispatch, shader_cache, cmd_trace), none of which is
 * reachable from outside gpu/.
 *
 * An AGC command buffer is a PM4 type-3 stream using the SAME IT_ opcode table
 * as the PS4, so the packet framing is shared (gpu/ps4/pm4.h) and the walk
 * looks like the PS4's. What differs is the gfx10.3 register file
 * (agc_regs.h), the RDNA2 shader ISA (ps5/rdna) and, above all, that AGC
 * programs most of its state out of guest memory rather than inline
 * (reg_state.h). The renderer (gpu/rhi) and the DrawInfo contract are shared
 * with the PS4 path.
 *
 * Submission is synchronous: a submit returns once the whole buffer has been
 * walked, so every fence label the packets ask the GPU to write is complete by
 * the time the walk passes it.
 */

#include "base/arch.h"

namespace gpu::ps5 {

// Process one AGC draw command buffer (guest GPU address, identity-mapped to a
// host pointer; size in bytes). Walks the packet stream, updating register
// state and issuing draws.
void SubmitDcb(const void* dcb, u32 size_bytes);
// A submit taken from a queue's ring: may stop early at an unsatisfied wait
// and returns the dwords consumed (see RingStall).
u32 SubmitDcbRing(const void* dcb, u32 size_bytes, u32 queue);

// Process one AGC constant command buffer. Framed identically, so it is the
// same walk; AGC has no separate constant engine to model.
void SubmitCcb(const void* ccb, u32 size_bytes);

// End the current frame and present the render target at `scanout_base` (the
// videoout flip buffer).
void EndFrame(u64 scanout_base);

}  // namespace gpu::ps5
