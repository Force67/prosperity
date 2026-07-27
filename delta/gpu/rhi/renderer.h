#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The renderer as the command processors see it. This header and command.h are
 * the whole surface: a command processor decodes guest packets into a DrawInfo
 * or a ComputeInfo and calls the entry points below, and never names a graphics
 * API type. The backend lives entirely behind this seam (gpu/vulkan today).
 *
 * Rendering is offscreen: each draw renders into the image for its
 * DrawInfo::rt_base (a render target keyed by guest address), and EndFrame
 * reads back the target at the scanout address to present it (or dump it,
 * headless).
 */

#include <cstdint>

// Spelled from the delta root so this header resolves for callers outside the
// gpu module too (the gpu-internal include root is not exported).
#include "gpu/rhi/command.h"

namespace gpu::rhi {

// Bring the backend up. Returns false when no usable device exists (the
// renderer is then disabled and the emulator runs without graphics).
bool Init();
bool Available();

// Frame lifecycle. BeginFrame starts recording; EndFrame submits, reads back
// the render target at `scanout_base` (the flip buffer) and presents/dumps it.
void BeginFrame();
void Draw(const DrawInfo& d);
void EndFrame(uint64_t scanout_base);

// Run a compute dispatch on the GPU. Returns true if it executed, false if it
// could not be set up (the caller then skips the dispatch, as before).
bool Dispatch(const ComputeInfo& ci);

// Write every GPU-dirty compute range back to guest memory. Must run before
// anything reads guest memory that a dispatch may have written: draw
// recording, CP DMA, frame end. No-op when nothing is dirty.
void FlushCsWrites();
// Flush only dirty ranges overlapping [base, base+bytes).
void FlushCsWritesRange(uint64_t base, uint64_t bytes);

// A CP DMA immediate fill over guest memory. When the range covers a live
// render target that is how the title clears it -- there is no clear packet on
// this hardware -- so the target takes a pending clear with the filled value.
void NoteMemoryFill(uint64_t base, uint64_t bytes, uint32_t value);

}  // namespace gpu::rhi
