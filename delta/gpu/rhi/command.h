#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The work a command processor hands the renderer: one decoded draw, or one
 * decoded compute dispatch. Backend-agnostic by construction -- nothing here
 * names a graphics API type, so the PS4 (PM4/GCN) and PS5 (AGC/RDNA2) command
 * processors compile without seeing the backend at all.
 *
 * Addresses are guest addresses (identity-mapped, host-readable).
 */

#include <cstdint>

namespace gpu::gcn { struct Recompiled; struct RecompiledCs; }

namespace gpu::rhi {

// One vertex attribute for the recompiled-shader path: where the recompiled VS
// reads input `location` from within a vertex buffer binding. `binding` indexes
// DrawInfo::vbufs -- multiple attributes that interleave in one buffer share a
// binding (distinct offsets); attributes fed from separate buffers each get
// their own binding (SotC streams position/normal/uv from distinct buffers).
struct VertexAttr {
  uint32_t location = 0;
  uint32_t binding = 0;   // index into DrawInfo::vbufs
  uint32_t offset = 0;    // byte offset within the binding's vertex record
  uint32_t num_comps = 0;  // 1..4
  uint32_t dfmt = 0;      // GCN data format (selects the backend format)
  uint32_t nfmt = 0;      // GCN number format
};

// One vertex buffer binding for the recompiled-shader path. Each distinct guest
// V# base + stride becomes one vertex binding; its records are uploaded into
// the renderer's vertex ring and bound for the draw.
struct VertexBinding {
  const void *data = nullptr;  // guest base of this binding's vertex data
  uint32_t stride = 0;         // bytes per record
  uint32_t numRecords = 0;     // records available in the source buffer
};

// Per-draw inputs extracted by the command processor (resource-tracked from the
// shader + register state).
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
  // The legacy transform buffer fields feed heuristic rendering and mirror the first
  // resolved VS cbuffer. Recompiled shaders use cbufs[] at set 1 bindings 0..7; each
  // entry is copied into a zero-padded dynamic UBO window. mvp[] remains binding 0's
  // fallback when the VS descriptor cannot be resolved.
  uint64_t cbufBase = 0;
  uint32_t cbufSize = 0;
  struct DrawCbuf { uint64_t base = 0; uint32_t size = 0; };
  DrawCbuf cbufs[16];
  uint32_t nCbufs = 0;
  uint64_t rtBase = 0;         // CB_COLOR0 address; the draw's render target
  uint32_t rtW = 0, rtH = 0;   // render-target dimensions (shared by all MRT targets)

  // Multiple render targets (CB_COLOR0..7). mrtBase[0] mirrors rtBase. A target is
  // bound when its CB_TARGET_MASK nibble and CB_COLORn_INFO format are non-zero and
  // its base is a valid guest address. mrtInfo preserves CB_COLORn_INFO so image and
  // pipeline attachment formats match the guest surface. mrtCount is zero for a
  // depth-only draw.
  uint64_t mrtBase[8] = {0};
  uint32_t mrtInfo[8] = {0};
  uint32_t mrtCount = 0;

  // Texturing (optional). texBase preserves every valid T# address so the renderer
  // can resolve non-RGBA guest formats to live render targets. Direct guest-memory
  // uploads remain limited to formats the upload path can decode.
  const void *uvData = nullptr;
  uint32_t uvStride = 0;
  uint32_t uvOffset = 0;       // byte offset of the float2 uv within the vertex
  uint32_t colorOffset = 0xFFFFFFFFu;  // byte offset of float3 color; ~0 = white
  uint64_t texBase = 0;
  uint32_t texW = 0, texH = 0;
  uint32_t texDfmt = 0, texNfmt = 0;
  uint32_t texTiling = 8;       // T# tiling_index (8/31 = linear; else tiled)
  uint32_t texPitch = 0;        // T# surface pitch in pixels (0 = use texW)
  uint32_t texLayers = 1;       // physical layers in the image allocation
  uint32_t texBaseArray = 0;    // first layer exposed by the image view
  uint32_t texViewLayers = 1;   // layers exposed by the image view
  uint32_t texMipLevels = 1;    // physical mip levels in the image allocation
  uint32_t texBaseMip = 0;      // first mip exposed by the image view
  uint32_t texViewMips = 1;     // mip levels exposed by the image view
  uint32_t texMinLod = 0;       // T# MIN_LOD clamp in U4.8 fixed-point
  uint32_t texSampler[4] = {};  // guest sampler descriptor for this MIMG binding
  bool texPow2Pad = false;      // physical mip dimensions/layers use POW2_PAD
  bool texSamplerValid = false;
  bool texArrayed = false;      // MIMG DA: shader consumes a layer coordinate
  bool texForceLodZero = false;
  bool texDepthCompare = false;
  uint32_t texSwizzle = 0;      // packed T# DST_SEL for the legacy single texture

  // Multi-texture: a PS can sample several textures (Doom64's 3D walls/floors use
  // a diffuse + lightmap + ... loaded from the EUD resource table). texs[0] mirrors
  // texBase. When nTexs > 1 the renderer binds an N-sampler descriptor set; texs[i]
  // maps to the recompiled PS's sampler binding i.
  struct DrawTex {
    uint64_t base = 0;
    uint32_t w = 0, h = 0, tiling = 8, pitch = 0;
    uint32_t dfmt = 0, nfmt = 0;
    uint32_t layers = 1, base_array = 0, view_layers = 1;
    uint32_t mip_levels = 1, base_mip = 0, view_mips = 1;
    uint32_t min_lod = 0;
    uint32_t sampler[4] = {};
    bool pow2_pad = false;
    bool sampler_valid = false;
    bool arrayed = false;
    bool force_lod_zero = false;
    bool depth_compare = false;
    bool storage = false;
    uint32_t swizzle = 0;  // packed T# DST_SEL_X/Y/Z/W (0 = identity)
  };
  DrawTex texs[16];
  uint32_t nTexs = 0;

  // Per-draw blend state, decoded from CB_BLEND0_CONTROL (raw dword) + whether
  // blending is enabled for color target 0. The renderer maps the GNM blend
  // factors/functions to a pipeline (cached per unique state).
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
  // colour write mask so a draw the game masks off (e.g. a fullscreen "clear"
  // it expects to write nothing) does not overwrite the target.
  uint32_t targetMask = 0xF;
  uint32_t colorControl = 0;

  // Depth/stencil (DB) state. When depthBase is a valid guest address and depthValid
  // is set the draw's region binds a depth attachment keyed by depthBase, and
  // honours the DB_DEPTH_CONTROL test/write/func below. 2D titles leave depthBase 0
  // (DB_Z_INFO format invalid), so no depth attachment is bound (unchanged path).
  uint64_t depthBase = 0;
  bool depthValid = false;        // DB_Z_INFO format != 0
  bool depthTestEnable = false;   // DB_DEPTH_CONTROL Z_ENABLE
  bool depthWriteEnable = false;  // DB_DEPTH_CONTROL Z_WRITE_ENABLE
  uint32_t depthFunc = 7;         // DB_DEPTH_CONTROL ZFUNC (maps 1:1 to the compare op)
  float depthClear = 1.0f;        // DB_DEPTH_CLEAR (fast-clear value)

  // Primitive-setup: raster topology + face culling, from VGT_PRIMITIVE_TYPE and
  // PA_SU_SC_MODE_CNTL. 2D titles draw triangle lists with no culling (unchanged).
  uint32_t cullMode = 0;          // PA_SU_SC_MODE_CNTL: CULL_FRONT[0] CULL_BACK[1]
  bool frontCCW = true;           // FACE[2] == 0

  // XY viewport transform from PA_CL_VPORT_0_*.
  float viewportXScale = 0, viewportXOffset = 0;
  float viewportYScale = 0, viewportYOffset = 0;

  // Recompiled-shader path. When recomp != null the renderer runs the game's actual
  // VS/PS instead of the heuristic quad; procedural VS programs may have no attributes.
  // vertexData/vertexStride is the raw interleaved vertex buffer, vattrs describe
  // the inputs, mvp holds the constant buffer (pushed), texBase the sampler.
  uint64_t vsAddr = 0, psAddr = 0;       // pipeline cache key
  uint32_t vsUserData[32] = {};
  uint32_t psUserData[32] = {};
  const gcn::Recompiled *recomp = nullptr;
  VertexAttr vattrs[8];
  uint32_t nvattrs = 0;
  // Vertex buffer bindings the attributes read from. vbufs[0] mirrors
  // vertexData/vertexStride so the single-binding fast path and the heuristic
  // fallback are unchanged; a multi-stream draw fills one entry per distinct V#.
  VertexBinding vbufs[8];
  uint32_t nvbufs = 0;
};

// A compute dispatch resolved by the command processor: the recompiled CS + the
// live guest memory ranges its descriptors point at (resolved from COMPUTE_USER_DATA)
// + the raw user data (pushed to the shader). The renderer stages each range into a
// storage buffer, runs the dispatch, and copies the written ranges back to guest
// memory (where the graphics texture path re-reads them).
struct ComputeInfo {
  static constexpr uint32_t kMaxResources = 32;

  uint64_t csAddr = 0;             // pipeline cache key
  uint32_t groups[3] = {1, 1, 1};  // workgroup counts (DISPATCH_DIRECT dims)
  const gcn::RecompiledCs *recomp = nullptr;
  uint32_t userData[16] = {};      // COMPUTE_USER_DATA_0..15 (push constants)
  struct Res {
    uint64_t base = 0;    // guest address the storage buffer aliases
    uint64_t size = 0;    // bytes staged in linear SSBO layout
    uint64_t guestSize = 0;  // physical guest bytes (same as size when linear)
    uint32_t binding = 0;
    bool written = false;  // copy back to guest after the dispatch
    bool zeroFill = false;  // inactive/null descriptor: bind zeroed dummy storage
    bool imageStaging = false;  // detile and/or expand compact texels
    uint32_t width = 0, height = 0, pitch = 0;
    uint32_t layers = 0, mipLevels = 0, tilingIdx = 0;
    uint32_t elemBytes = 4;
    uint32_t stageElemBytes = 4;
    uint32_t dfmt = 0;
    bool pow2Pad = false;
  };
  Res res[kMaxResources];
  uint32_t nres = 0;
};

}  // namespace gpu::rhi
