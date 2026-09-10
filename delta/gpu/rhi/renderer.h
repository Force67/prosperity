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

#include <cstddef>
#include "base/arch.h"

// Spelled from the delta root, the one include convention the layering check
// (tests/check_layering.py) accepts; all modules share that include root, so
// internal headers are kept private by the check, not by the build.
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
void EndFrame(Renderer& renderer, u64 scanout_base);

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
bool FlushCsWritesRange(Renderer& renderer, u64 base, u64 bytes);

// CP transfers address the same 64 KiB GDS allocation as shader DS operations.
// Offsets and lengths are bytes; transfers wait for preceding compute work.
bool ReadGds(Renderer& renderer, u32 offset, void* data, u32 bytes);
bool WriteGds(Renderer& renderer, u32 offset, const void* data, u32 bytes);
bool FillGds(Renderer& renderer, u32 offset, u32 bytes, u32 value);

// Monotonic count of compute-results-became-visible-in-guest-memory events
// (a range writeback, or an executed batch of dispatches that write guest
// memory directly). A cached copy of guest bytes taken at generation G is
// current iff the generation still reads G and no dirty range overlaps it.
u64 CsWritebackGeneration();
// Whether any GPU-dirty compute range overlaps [base, base+bytes). O(1) when
// nothing is dirty anywhere (the common case on the draw path).
bool CsRangeDirtyOverlapping(u64 base, u64 bytes);

// A CP DMA immediate fill over guest memory. When the range covers a live
// render target that is how the title clears it -- there is no clear packet on
// this hardware -- so the target takes a pending clear with the filled value.
void NoteMemoryFill(Renderer& renderer,
                    u64 base,
                    u64 bytes,
                    u32 value);

// Does `addr` fall inside a compute staging range -- guest memory the renderer
// snapshots and copies back? A guest fault on memory the guest alone should own
// wants that answered on the spot: the crash handler asks, so a corrupted heap
// word can be attributed to (or cleared of) the compute writeback without a
// second run. Returns false if no range covers it; otherwise fills `out` with
// the range's base, size and staging state.
bool DescribeCsRangeCovering(u64 addr, char* out, size_t out_size);

// The process-wide renderer instance the command processors drive. The
// composition root (main/dapi) Init()s it once; the HLE submit paths reach it
// through this accessor because the guest-called entry points cannot thread a
// handle. The single point of ambient state at this seam.
Renderer& DefaultRenderer();

}  // namespace gpu::rhi
