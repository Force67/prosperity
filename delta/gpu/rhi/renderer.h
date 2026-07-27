#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The renderer as the command processors see it. This header and command.h are
 * the whole surface: a command processor decodes guest packets into a DrawInfo
 * or a ComputeInfo and calls the operations below, and never names a graphics
 * API type. The backend lives entirely behind this seam (gpu/vulkan today).
 *
 * The renderer is a value the caller holds: a Renderer with a few cheap
 * queries, operated on by free functions. All backend state hangs off
 * Renderer::state (opaque here; defined by the backend), so a second backend
 * or a test double is a different BackendState behind the same operations.
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

struct BackendState;  // owned by the backend; opaque outside it

struct Renderer {
  BackendState* state = nullptr;

  // True once Init succeeded; every operation below is a no-op (or returns
  // false) on an unavailable renderer.
  // Chromium's cheap-accessor spelling. NOLINT: the naming check cannot
  // distinguish accessors from functions that do work.
  bool available() const { return state != nullptr; }  // NOLINT
};

// Bring the backend up. Returns false when no usable device exists (the
// renderer is then left unavailable and the emulator runs without graphics).
// Idempotent: calling again on an available renderer is a no-op success.
bool Init(Renderer& renderer);

// Frame lifecycle. BeginFrame starts recording; EndFrame submits, reads back
// the render target at `scanout_base` (the flip buffer) and presents/dumps it.
void BeginFrame(Renderer& renderer);
void Draw(Renderer& renderer, const DrawInfo& d);
void EndFrame(Renderer& renderer, uint64_t scanout_base);

// Run a compute dispatch on the GPU. Returns true if it executed, false if it
// could not be set up (the caller then skips the dispatch, as before).
bool Dispatch(Renderer& renderer, const ComputeInfo& ci);

// Write every GPU-dirty compute range back to guest memory. Must run before
// anything reads guest memory that a dispatch may have written: draw
// recording, CP DMA, frame end. No-op when nothing is dirty; returns false when
// the data could not be made current.
bool FlushCsWrites(Renderer& renderer);
// Flush only dirty ranges overlapping [base, base+bytes), with the same result
// contract as FlushCsWrites.
bool FlushCsWritesRange(Renderer& renderer, uint64_t base, uint64_t bytes);

// A CP DMA immediate fill over guest memory. When the range covers a live
// render target that is how the title clears it -- there is no clear packet on
// this hardware -- so the target takes a pending clear with the filled value.
void NoteMemoryFill(Renderer& renderer,
                    uint64_t base,
                    uint64_t bytes,
                    uint32_t value);

// The process-wide renderer instance the command processors drive. The
// composition root (main/dapi) Init()s it once; the HLE submit paths reach it
// through this accessor because the guest-called entry points cannot thread a
// handle. The single point of ambient state at this seam.
Renderer& DefaultRenderer();

}  // namespace gpu::rhi
