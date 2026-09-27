#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PS4 GPU command processor: consumes the PM4 command buffers the Gnm HLE
 * submits, tracks the Liverpool register state they program, and drives the
 * renderer. This header is its whole surface; the decode itself is split by
 * decision across ps4/ (draw_state, compute_dispatch, shader_cache, cmd_trace),
 * none of which is reachable from outside gpu/.
 *
 * A submit returns once the whole buffer has been walked. The draws it found
 * are recorded on the renderer thread (render_queue.h), and every fence label
 * is written there, in order, once the draws ahead of it have read their guest
 * memory.
 */

#include <cstddef>
#include <cstdint>
#include "base/arch.h"

namespace gpu::ps4 {

// Select the PS4 shader hardware profile for subsequent submissions. The
// composition root calls this after combining configured Neo hardware with the
// title's param.sfo Neo-support bit.
void SetPs4NeoMode(bool enabled);

// Register the kernel's resumable guest-memory write watch without making the
// GPU module depend on the kernel. Used by the DELTA_GPU_ROOT_WPROT_* probes to
// follow a dynamically allocated shader root when its command pool is reused.
using WriteWatchCallback = void (*)(uintptr_t, size_t, unsigned, bool, bool);
void SetWriteWatchCallback(WriteWatchCallback callback);

// Process one PM4 draw command buffer (guest GPU address, identity-mapped to a
// host pointer; size in bytes). Walks the packet stream, updating register
// state and issuing draws. Safe to call from the guest's GPU submit thread.
void SubmitDcb(const void* dcb, u32 size_bytes);

// Process one PM4 constant command buffer (ccb). The Constant Engine fills its
// on-chip CE RAM (WRITE/LOAD_CONST_RAM) and dumps it to guest memory
// (DUMP_CONST_RAM) where shaders read it as constant buffers. The DE-only path
// ignored this; draws whose cbuffers come via the CE then read stale memory.
// Must run before the matching SubmitDcb (the CE runs ahead of the draw
// engine).
void SubmitCcb(const void* ccb, u32 size_bytes);

// End the current frame and present the render target at `scanout_base` (the
// videoout flip buffer). Called by the Gnm submit-and-flip HLE.
void EndFrame(u64 scanout_base);

}  // namespace gpu::ps4
