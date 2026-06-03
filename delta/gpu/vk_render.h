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

// Per-draw inputs extracted from the Liverpool register state by the command
// processor. Addresses are guest GPU addresses (identity-mapped, host-readable).
struct DrawInfo {
  uint64_t vsAddr = 0;   // VS GCN bytecode
  uint64_t psAddr = 0;   // PS GCN bytecode
  uint32_t primType = 0; // VGT_PRIMITIVE_TYPE
  uint32_t indexCount = 0;
  uint32_t indexType = 0;     // 0=16-bit, 1=32-bit
  uint64_t indexAddr = 0;     // index buffer (0 => auto)
  const uint32_t *vsUserData = nullptr;  // 16 SGPRs
  const uint32_t *psUserData = nullptr;
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
