/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
struct Renderer;
}

namespace gpu::render::trace {

extern bool g_recording;
inline bool Recording() {
  return g_recording;
}

void FrameBegin(int frame_num);
void FrameEnd(u64 scanout_base);

struct RegionInfo {
  const u64* mrt_base = nullptr;
  const u32* mrt_info = nullptr;
  u32 mrt_count = 0;
  u32 width = 0, height = 0;
  u64 depth_base = 0;
  u64 stencil_base = 0;
  u32 color_clear_mask = 0;
  bool depth_clear = false;
  float depth_clear_value = 1.0f;
};
void RegionBegin(const RegionInfo& region);
void RegionEnd();

struct DrawBindings {
  const u64* tex_color = nullptr;
  const u64* tex_feedback = nullptr;
  const u64* tex_depth = nullptr;
  const u64* tex_storage = nullptr;
  const void* const* tex_guest = nullptr;
  u32 tex_count = 0;
  u32 cbuf_mask = 0;
  u32 rawbuf_mask = 0;
};

void RecordDraw(const render::DrawInfo& draw,
                const char* path,
                const DrawBindings* bindings);
void RecordDecline(const char* reason);

void RecordDispatch(const render::ComputeInfo& dispatch);
bool CaptureDispatchInputs(Renderer& renderer,
                           const render::ComputeInfo& dispatch);
void CaptureDispatchOutputs(const render::ComputeInfo& dispatch);
void CaptureDispatchBindings(const render::ComputeInfo& dispatch,
                             const u32* push_constants,
                             u32 push_dwords,
                             const rhi::BindingWrite* bindings,
                             rhi::Buffer* const* snapshots);

void RecordBarrier(const char* aspect,
                   u64 image,
                   rhi::TextureState from,
                   rhi::TextureState to);

void RecordMemoryFill(u64 base, u64 bytes, u64 value, u32 pattern_bytes = 4);

void RecordVariantSwap(u64 base,
                       u64 from,
                       u32 from_w,
                       u32 from_h,
                       u64 to,
                       u32 to_w,
                       u32 to_h);

bool WantValidation();
bool WantSyncValidation();
void OnDeviceMessage(const char* level,
                     const char* id,
                     const char* labels,
                     const char* message);

const char* ObjectName(u64 handle);

}  // namespace gpu::render::trace
