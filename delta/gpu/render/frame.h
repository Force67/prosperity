/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The frame ring. Two slots let frame N record (and the guest emulate) while
// frame N-1 still rasterizes; each slot owns a command list, a submission id, a
// readback buffer and half of each upload ring. Slot N-1's submission is waited
// – and its pixels presented, one frame late: at frame N's EndFrame.

#include "base/arch.h"
#include "base/containers/vector.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct FrameSlot {
  rhi::CommandList* list = nullptr;  // the chunk this slot is recording
  u64 submission = 0;  // the frame's last submission, waited at finish
  rhi::TimestampPool* timestamps = nullptr;
  // DELTA_GPU_PASSPROF: a timestamp pair per render region; mark i owns
  // queries 2i and 2i+1.
  struct PassMark {
    u64 target;
    u32 draws;
    bool traced;  // also a zone on the Tracy GPU timeline
    u16 query;    // its timeline query id
  };
  rhi::TimestampPool* pass_timestamps = nullptr;
  base::Vector<PassMark> pass_marks;
  rhi::Buffer* readback = nullptr;
  // Chunks of this frame submitted early (SubmitFrameChunk), recycled once
  // the slot's submission has retired.
  base::Vector<rhi::CommandList*> chunks;
  bool submitted = false;    // submitted and not yet waited
  bool presentable = false;  // the frame copied pixels into `readback`
  bool present_to_window = false;
  // Metadata of the recorded frame, consumed when it is presented.
  u32 w = 0, h = 0;
  rhi::Format fmt = rhi::Format::kUndefined;
  int frame_num = 0;
  u32 frame_draws = 0, frame_max_idx = 0;
  bool frame_had_room = false;
  u64 present_base = 0, scanout_base = 0;
};

struct FrameState {
  // The active slot's command list and readback buffer, aliased here so the
  // recording path does not thread the slot through every call.
  rhi::CommandList* list = nullptr;
  rhi::Buffer* readback = nullptr;
  void* readback_map = nullptr;
  u64 readback_size = 0;

  int num = 0;  // monotonic frame counter; the caches age against it
  u32 draws = 0;
  u32 heuristic = 0;  // draws that fell back to the heuristic quad path
  u32 max_idx = 0;    // largest index_count this frame (3D detector)
  bool recording = false;
  bool had_room = false;   // this frame sampled a room-sized (~832w) RT
  bool room_bake = false;  // this frame RENDERED into a room-sized RT

  // Which chunk of the frame `list` is recording: bumped by SubmitFrameChunk.
  // Work recorded against chunk N is on the queue once the sequence passes N.
  u64 chunk_seq = 1;
  u32 draws_at_chunk = 0;  // `draws` when the chunk was opened

  FrameSlot slots[2];
  u32 slot_idx = 0;
};

extern FrameState& g_frame;

bool CreateFrameSlots();

// The device the renderer runs on.
rhi::Device& Device();

// A command list for work the CPU needs finished before it goes on. Record
// into it, then EndImmediate submits, waits and recycles it; false when the
// submission failed or the device was lost.
rhi::CommandList* BeginImmediate();
bool EndImmediate(rhi::CommandList* list);
// Pipelined by default; DELTA_GPU_SYNC=1 restores the submit-and-wait frame.
bool FramePipelined();
// DELTA_GPU_PASSPROF: time the render region about to open, keyed by its
// first target; PassProfEnd closes it. No-ops when the knob is off.
void PassProfBegin(u64 target);
void PassProfEnd();
// Grow the active slot's readback buffer to hold one w*h image of `fmt`.
void EnsureReadback(u32 w, u32 h, rhi::Format fmt);

// Submit what the frame has recorded so far, without waiting, and go on
// recording into a fresh command buffer. Work submitted next (a compute batch,
// a bridge copy) then executes after those draws instead of a frame early.
bool SubmitFrameChunk();
// Stamp every target's submitted layout with its recorded one: what the GPU
// holds once the work submitted so far executes.
void StampSubmittedLayouts();

}  // namespace gpu::render
