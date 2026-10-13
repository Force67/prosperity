/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/containers/vector.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct FrameSlot {
  rhi::CommandList* list = nullptr;
  u64 submission = 0;
  rhi::TimestampPool* timestamps = nullptr;
  struct PassMark {
    u64 target;
    u32 draws;
    bool traced;
    u16 query;
  };
  rhi::TimestampPool* pass_timestamps = nullptr;
  base::Vector<PassMark> pass_marks;
  rhi::Buffer* readback = nullptr;
  base::Vector<rhi::CommandList*> chunks;
  bool submitted = false;
  bool presentable = false;
  bool present_to_window = false;
  rhi::Texture* shared = nullptr;
  int shared_fd = -1;
  u32 shared_id = 0;
  u64 shared_bytes = 0;
  bool shared_frame = false;
  u32 w = 0, h = 0;
  rhi::Format fmt = rhi::Format::kUndefined;
  int frame_num = 0;
  u32 frame_draws = 0, frame_max_idx = 0;
  bool frame_had_room = false;
  u64 present_base = 0, scanout_base = 0;
};

struct FrameState {
  rhi::CommandList* list = nullptr;
  rhi::Buffer* readback = nullptr;
  void* readback_map = nullptr;
  u64 readback_size = 0;

  int num = 0;
  u32 draws = 0;
  u32 max_idx = 0;
  bool recording = false;
  bool had_room = false;
  bool room_bake = false;

  u64 chunk_seq = 1;
  u32 draws_at_chunk = 0;

  FrameSlot slots[2];
  u32 slot_idx = 0;
};

extern FrameState& g_frame;

bool CreateFrameSlots();

rhi::Device& Device();

rhi::CommandList* BeginImmediate();
bool EndImmediate(rhi::CommandList* list, const char* why = "imm.other");
bool FramePipelined();
void PassProfBegin(u64 target);
void PassProfEnd();
void DrawProfBegin(u64 ps);
void DrawProfEnd();
void EnsureReadback(u32 w, u32 h, rhi::Format fmt);

bool SubmitFrameChunk();
void StampSubmittedLayouts();

}  // namespace gpu::render
