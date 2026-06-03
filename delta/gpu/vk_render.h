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

  // Texturing (optional). If texBase != 0 the draw is textured: uvData holds the
  // float2 UVs (same vertexCount/stride as the position buffer's source).
  const void *uvData = nullptr;
  uint32_t uvStride = 0;
  uint32_t uvOffset = 0;       // byte offset of the float2 uv within the vertex
  uint32_t colorOffset = 0xFFFFFFFFu;  // byte offset of float3 color; ~0 = white
  uint64_t texBase = 0;
  uint32_t texW = 0, texH = 0;
};

// Bring up the headless Vulkan device. Returns false if Vulkan is unavailable
// (then the renderer is disabled and the emulator runs without graphics).
bool init();
bool available();

// Frame lifecycle. rtAddr/width/height/pitch/format describe the render target
// (the guest scanout). beginFrame ensures an RT image of that size exists and
// starts recording; endFrame submits, reads the RT back to a linear RGBA buffer
// and presents/dumps it.
void beginFrame(uint64_t rtAddr, uint32_t width, uint32_t height, uint32_t pitch,
                uint32_t cbInfo);
void draw(const DrawInfo &d);
void endFrame();

}  // namespace gpu::vk
