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

namespace gpu::gcn { struct Recompiled; }

namespace gpu::vk {

// One vertex attribute for the recompiled-shader path: where the recompiled VS
// reads input `location` from within the (interleaved) vertex buffer.
struct VertexAttr {
  uint32_t location = 0;
  uint32_t offset = 0;    // byte offset within the vertex
  uint32_t numComps = 0;  // 1..4
  uint32_t dfmt = 0;      // GCN data format (selects the Vulkan format)
  uint32_t nfmt = 0;      // GCN number format
};

// Per-draw inputs extracted by the command processor (resource-tracked from the
// shader + register state). Addresses are guest (identity-mapped, host-readable).
struct DrawInfo {
  const void *vertexData = nullptr;  // base of attribute-0 (position) buffer
  uint32_t vertexCount = 0;
  uint32_t vertexStride = 0;   // bytes per vertex in the source buffer
  uint32_t posOffset = 0;      // byte offset of the float2 position
  uint32_t primType = 0;       // VGT_PRIMITIVE_TYPE (4 = triangle list)

  // Index buffer (DRAW_INDEX_2). When indexData != null the draw is indexed: the
  // indices select vertices out of the vertex buffer. indexType: 0 = 16-bit,
  // 1 = 32-bit. Without an index buffer the draw is sequential (DRAW_INDEX_AUTO).
  const void *indexData = nullptr;
  uint32_t indexCount = 0;
  uint32_t indexType = 0;
  uint32_t instanceCount = 1;  // from IT_NUM_INSTANCES (tilemaps draw instanced)
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
  uint32_t texTiling = 8;       // T# tiling_index (8/31 = linear; else tiled)
  uint32_t texPitch = 0;        // T# surface pitch in pixels (0 = use texW)

  // Per-draw blend state, decoded from CB_BLEND0_CONTROL (raw dword) + whether
  // blending is enabled for color target 0. The renderer maps the GNM blend
  // factors/functions to a Vulkan pipeline (cached per unique state).
  uint32_t blendControl = 0;
  bool blendEnable = false;
  // CB_TARGET_MASK (per-MRT channel write enable; MRT0 = bits[3:0]) and
  // CB_COLOR_CONTROL (MODE field [6:4]; 0 = disable color output). Honoured as the
  // Vulkan colorWriteMask so a draw the game masks off (e.g. a fullscreen "clear"
  // it expects to write nothing) does not overwrite the target.
  uint32_t targetMask = 0xF;
  uint32_t colorControl = 0;

  // Recompiled-shader path. When recomp != null and nvattrs > 0 the renderer runs
  // the game's actual VS/PS (recompiled to SPIR-V) instead of the heuristic quad:
  // vertexData/vertexStride is the raw interleaved vertex buffer, vattrs describe
  // the inputs, mvp holds the constant buffer (pushed), texBase the sampler.
  uint64_t vsAddr = 0, psAddr = 0;       // pipeline cache key
  const gcn::Recompiled *recomp = nullptr;
  VertexAttr vattrs[8];
  uint32_t nvattrs = 0;
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
