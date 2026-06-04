#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Headless Vulkan renderer for the GPU command processor. Owns its own Vulkan
 * device (no surface): rendering is offscreen into a render-target image that
 * matches the guest scanout). The command processor calls beginFrame() once per
 * submitted frame, draw() per decoded PM4 draw, and endFrame() to finish and
 * read the result back (presented to the window when a display exists, or dumped
 * for headless verification).
 */

#include <cstdint>

namespace gpu::vk {

// Per-draw inputs extracted by the command processor (resource-tracked from the
// shader + register state). Addresses are guest (identity-mapped, host-readable).
struct DrawInfo {
  const void *vertexData = nullptr;  // base of attribute-0 (position) buffer
  uint32_t vertexCount = 0;
  uint32_t vertexStride = 0;   // bytes per vertex in the source buffer
  uint32_t posOffset = 0;      // byte offset of the float2 position
  uint32_t primType = 0;       // VGT_PRIMITIVE_TYPE
  uint32_t indexCount = 0;
  float mvp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  uint64_t rtBase = 0;         // CB_COLOR0 address; the draw's render target
  uint32_t rtW = 0, rtH = 0;   // render-target dimensions

  // Texturing (optional). If texBase != 0 the draw is textured: uvData holds the
  // float2 UVs (same vertexCount/stride as the position buffer's source).
  const void *uvData = nullptr;
  uint32_t uvStride = 0;
  uint32_t uvOffset = 0;       // byte offset of the float2 uv within the vertex
  uint32_t colorOffset = 0xFFFFFFFFu;  // byte offset of float3 color; ~0 = white
  uint64_t texBase = 0;
  uint32_t texW = 0, texH = 0;

  // Per-draw blend state, decoded from CB_BLEND0_CONTROL (raw dword) + whether
  // blending is enabled for color target 0. The renderer maps the GNM blend
  // factors/functions to a Vulkan pipeline (cached per unique state).
  uint32_t blendControl = 0;
  bool blendEnable = false;
};

// Bring up the headless Vulkan device. Returns false if Vulkan is unavailable
// (then the renderer is disabled and the emulator runs without graphics).
bool init();
bool available();

// Frame lifecycle. Each draw renders into the Vulkan image for its DrawInfo.rtBase
// (a render target keyed by guest address). beginFrame starts recording;
// endFrame submits, reads back the render target at `scanoutBase` (the flip
// buffer) and presents/dumps it.
void beginFrame();
void draw(const DrawInfo &d);
void endFrame(uint64_t scanoutBase);

}  // namespace gpu::vk
