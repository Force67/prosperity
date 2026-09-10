#pragma once
#include <base.h>
#include "base/arch.h"

namespace runtime::video {
// Public Videodec2 ABI. Sizes and offsets checked against the PS5 firmware
// wrappers; the 0x50 configuration extends this 0x48 prefix.
struct Config {
  u64 size;
  u32 resource, codec, profile, level;
  i32 width, height, references;
  u32 pipeline;
  void* queue;
  u64 affinity;
  i32 priority;
  u8 progressive, check_memory, reserved[2];
  void* extra;
};
struct Memory {
  u64 size, cpu_bytes;
  void* cpu;
  u64 gpu_bytes;
  void* gpu;
  u64 shared_bytes;
  void* shared;
  u64 frame_bytes;
  u32 alignment, reserved;
};
struct Input { u64 size; const void* data; u64 bytes, pts, dts, attached; };
struct FrameBuffer { u64 size; void* data; u64 bytes; bool accepted; };
struct Output {
  u64 size;
  u8 valid, error, pictures, reserved;
  u32 codec, width, pitch, height;
  void* data;
  u64 bytes;
  u32 format, pitch_bytes;
};
struct ComputeMemory { u64 size, bytes; void* data; };
struct ComputeConfig { u64 size; u16 pipe, queue; u8 check_memory, reserved; u16 reserved2; };
static_assert(sizeof(Config) == 0x48 && sizeof(Memory) == 0x48);
static_assert(sizeof(Input) == 0x30 && sizeof(FrameBuffer) == 0x20);
static_assert(sizeof(Output) == 0x38 && sizeof(ComputeMemory) == 0x18);

i32 PS4ABI QueryCompute(ComputeMemory*);
i32 PS4ABI AllocateQueue(const ComputeConfig*, const ComputeMemory*, void**);
i32 PS4ABI ReleaseQueue(void*);
i32 PS4ABI QueryMemory(const Config*, Memory*);
i32 PS4ABI Create(const Config*, const Memory*, void**);
i32 PS4ABI Delete(void*);
i32 PS4ABI Decode(void*, const Input*, FrameBuffer*, Output*);
i32 PS4ABI Flush(void*, FrameBuffer*, Output*);
i32 PS4ABI Reset(void*);
i32 PS4ABI Picture(const Output*, void*, void*);
}
