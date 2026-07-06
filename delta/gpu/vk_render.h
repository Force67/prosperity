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

namespace gpu::gcn { struct Recompiled; struct RecompiledCs; }

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
  // Recompiled VS cbuffer (transforms): the guest constant-buffer base+size resolved
  // from the VS's cbuffer V#. The renderer copies a window of it into a dynamic UBO
  // (set 1) the recompiled VS reads. mvp[] mirrors the first 64 bytes (heuristic-path
  // fallback). cbufBase==0 means unresolved (fall back to mvp).
  uint64_t cbufBase = 0;
  uint32_t cbufSize = 0;
  uint64_t rtBase = 0;         // CB_COLOR0 address; the draw's render target
  uint32_t rtW = 0, rtH = 0;   // render-target dimensions (shared by all MRT targets)

  // Multiple render targets (CB_COLOR0..7). mrtBase[0] mirrors rtBase. A target is
  // bound when its CB_TARGET_MASK nibble is non-zero and its base is a valid guest
  // address; mrtCount is highest-bound-index + 1 (1 for the common single-RT case,
  // so the renderer's single-attachment path is unchanged). All targets share rtW/rtH.
  uint64_t mrtBase[8] = {0};
  uint32_t mrtCount = 1;

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

  // Multi-texture: a PS can sample several textures (Doom64's 3D walls/floors use
  // a diffuse + lightmap + ... loaded from the EUD resource table). texs[0] mirrors
  // texBase. When nTexs > 1 the renderer binds an N-sampler descriptor set; texs[i]
  // maps to the recompiled PS's sampler binding i.
  struct DrawTex { uint64_t base = 0; uint32_t w = 0, h = 0, tiling = 8, pitch = 0; };
  DrawTex texs[8];
  uint32_t nTexs = 0;

  // Per-draw blend state, decoded from CB_BLEND0_CONTROL (raw dword) + whether
  // blending is enabled for color target 0. The renderer maps the GNM blend
  // factors/functions to a Vulkan pipeline (cached per unique state).
  uint32_t blendControl = 0;
  bool blendEnable = false;
  // Per-MRT blend: CB_BLENDn_CONTROL for each color target, with a per-target enable
  // bit in mrtBlendMask. mrtBlend[0]/mrtBlendMask bit0 mirror blendControl/blendEnable,
  // so the single-RT path is unchanged; an MRT draw (CB_COLOR1..7) gets each target's
  // own blend instead of target 0's blend applied to every attachment.
  uint32_t mrtBlend[8] = {0};
  uint32_t mrtBlendMask = 0;
  // CB_TARGET_MASK (per-MRT channel write enable; MRT0 = bits[3:0]) and
  // CB_COLOR_CONTROL (MODE field [6:4]; 0 = disable color output). Honoured as the
  // Vulkan colorWriteMask so a draw the game masks off (e.g. a fullscreen "clear"
  // it expects to write nothing) does not overwrite the target.
  uint32_t targetMask = 0xF;
  uint32_t colorControl = 0;

  // Depth/stencil (DB) state. When depthBase is a valid guest address and depthValid
  // is set the draw's region binds a Vulkan depth attachment keyed by depthBase, and
  // honours the DB_DEPTH_CONTROL test/write/func below. 2D titles leave depthBase 0
  // (DB_Z_INFO format invalid), so no depth attachment is bound (unchanged path).
  uint64_t depthBase = 0;
  bool depthValid = false;        // DB_Z_INFO format != 0
  bool depthTestEnable = false;   // DB_DEPTH_CONTROL Z_ENABLE
  bool depthWriteEnable = false;  // DB_DEPTH_CONTROL Z_WRITE_ENABLE
  uint32_t depthFunc = 7;         // DB_DEPTH_CONTROL ZFUNC (maps 1:1 to VkCompareOp)
  float depthClear = 1.0f;        // DB_DEPTH_CLEAR (fast-clear value)

  // Primitive-setup: raster topology + face culling, from VGT_PRIMITIVE_TYPE and
  // PA_SU_SC_MODE_CNTL. 2D titles draw triangle lists with no culling (unchanged).
  uint32_t cullMode = 0;          // PA_SU_SC_MODE_CNTL: CULL_FRONT[0] CULL_BACK[1]
  bool frontCCW = true;           // FACE[2] == 0

  // Recompiled-shader path. When recomp != null and nvattrs > 0 the renderer runs
  // the game's actual VS/PS (recompiled to SPIR-V) instead of the heuristic quad:
  // vertexData/vertexStride is the raw interleaved vertex buffer, vattrs describe
  // the inputs, mvp holds the constant buffer (pushed), texBase the sampler.
  uint64_t vsAddr = 0, psAddr = 0;       // pipeline cache key
  const gcn::Recompiled *recomp = nullptr;
  VertexAttr vattrs[8];
  uint32_t nvattrs = 0;
};

// A compute dispatch resolved by the command processor: the recompiled CS + the
// live guest memory ranges its descriptors point at (resolved from COMPUTE_USER_DATA)
// + the raw user data (pushed to the shader). The renderer stages each range into a
// storage buffer, runs the dispatch, and copies the written ranges back to guest
// memory (where the graphics texture path re-reads them).
struct ComputeInfo {
  uint64_t csAddr = 0;             // pipeline cache key
  uint32_t groups[3] = {1, 1, 1};  // workgroup counts (DISPATCH_DIRECT dims)
  const gcn::RecompiledCs *recomp = nullptr;
  uint32_t userData[16] = {};      // COMPUTE_USER_DATA_0..15 (push constants)
  struct Res {
    uint64_t base = 0;    // guest address the storage buffer aliases
    uint64_t size = 0;    // bytes staged
    uint32_t binding = 0;
    bool written = false;  // copy back to guest after the dispatch
  };
  Res res[8];
  uint32_t nres = 0;
};

// Bring up the headless Vulkan device. Returns false if Vulkan is unavailable
// (then the renderer is disabled and the emulator runs without graphics).
bool init();
bool available();

// Run a compute dispatch on the GPU. Returns true if it executed, false if it
// could not be set up (the caller then skips the dispatch, as before).
bool dispatch(const ComputeInfo &ci);

// Frame lifecycle. Each draw renders into the Vulkan image for its DrawInfo.rtBase
// (a render target keyed by guest address). beginFrame starts recording;
// endFrame submits, reads back the render target at `scanoutBase` (the flip
// buffer) and presents/dumps it.
void beginFrame();
void draw(const DrawInfo &d);
void endFrame(uint64_t scanoutBase);

}  // namespace gpu::vk
