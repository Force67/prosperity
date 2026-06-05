/*
 * PS4Delta : PS4 emulation and research project
 *
 * GPU command processor. See cmd_processor.h.
 */

#include "cmd_processor.h"
#include "pm4.h"
#include "liverpool.h"
#include "vk_render.h"
#include "gcn/gcn_decode.h"
#include "gcn/gcn_resource.h"
#include "gcn/gcn_translate.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace gpu {
namespace {

std::mutex g_mtx;
Regs g_regs;  // persistent register state across submits (Gnm relies on this)
const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;
std::atomic<uint64_t> g_totalSubmits{0};
std::atomic<uint64_t> g_totalDraws{0};
bool g_vkTried = false;
bool g_frameActive = false;

// Index-buffer state, set by IT_INDEX_TYPE / IT_INDEX_BASE before a draw.
uint32_t g_indexType = 0;  // 0 = 16-bit, 1 = 32-bit (VGT_DMA_INDEX_TYPE bits[1:0])
uint64_t g_indexBase = 0;  // from IT_INDEX_BASE (DRAW_INDEX_2 carries its own base)
uint32_t g_numInstances = 1;  // from IT_NUM_INSTANCES; applies to the next draw(s)

// Current render-target / framebuffer geometry, derived from the screen scissor
// (CB regs don't carry an explicit width/height).
uint32_t fbWidth() {
  uint32_t br = g_regs[mmPA_SC_SCREEN_SCISSOR_BR];
  uint32_t w = br & 0xFFFF;
  return w ? w : 1920;
}
uint32_t fbHeight() {
  uint32_t br = g_regs[mmPA_SC_SCREEN_SCISSOR_BR];
  uint32_t h = br >> 16;
  return h ? h : 1080;
}

// Write a run of register values from a SET_*_REG packet body into the file.
void setRegs(uint32_t base, const uint32_t *body, uint32_t count) {
  // body[0] = reg offset (relative to base); body[1..] = values.
  uint32_t off = base + body[0];
  for (uint32_t i = 1; i < count; i++) {
    uint32_t idx = off + (i - 1);
    if (idx < kRegFileSize)
      g_regs[idx] = body[i];
  }
}

bool isDraw(uint32_t op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDEX_INDIRECT ||
         op == IT_DRAW_INDEX_MULTI_AUTO;
}

// Opcode histogram (DELTA_GPU_TRACE): shows what the dcb
// contains and whether the walker reaches a draw or desyncs.
uint32_t g_opHist[256] = {};
int g_dcbSeen = 0;
void dumpHist() {
  std::fprintf(stderr, "[gpu] dcb opcode histogram (after %d dcbs):\n", g_dcbSeen);
  for (int i = 0; i < 256; i++)
    if (g_opHist[i])
      std::fprintf(stderr, "[gpu]   op %#04x x%u\n", i, g_opHist[i]);
}

// Issue the current register state as a draw: begin the frame lazily on the
// first draw, then hand the draw to the Vulkan renderer.
void handleDraw(uint32_t op, const uint32_t *body, uint32_t count) {
  uint64_t vsA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_VS);
  uint64_t psA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_PS);
  // One-time: recompile the first VS/PS pair to GLSL and print it (DELTA_GPU_SHTRACE),
  // to validate the GCN->GLSL translator before wiring it into the pipeline.
  static bool g_shTried = false;
  if (std::getenv("DELTA_GPU_SHTRACE") && !g_shTried &&
      vsA >= 0x1000000000ull && vsA < 0x20000000000ull &&
      psA >= 0x1000000000ull && psA < 0x20000000000ull) {
    g_shTried = true;
    gcn::recompile(reinterpret_cast<const uint32_t *>(vsA),
                   reinterpret_cast<const uint32_t *>(psA),
                   &g_regs[mmSPI_SHADER_USER_DATA_VS_0],
                   &g_regs[mmSPI_SHADER_USER_DATA_PS_0]);
  }
  // One-time: find the first PS that samples a texture (has an MIMG instruction)
  // and dump how it loads its resources, so we can wire texture sampling.
  static bool g_texProbed = false;
  if (g_trace && !g_texProbed && psA >= 0x1000000000ull && psA < 0x20000000000ull) {
    auto pi = gcn::decode(reinterpret_cast<const uint32_t *>(psA), 256);
    int nMimg = 0, nSmrd = 0;
    for (auto &in : pi) {
      if (in.enc == gcn::Enc::mimg) nMimg++;
      if (in.enc == gcn::Enc::smrd) nSmrd++;
    }
    if (nMimg > 0) {
      g_texProbed = true;
      std::fprintf(stderr, "[gpu] TEXTURED PS @%#lx: mimg=%d smrd=%d\n",
                   (unsigned long)psA, nMimg, nSmrd);
      const uint32_t *pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];
      std::fprintf(stderr, "[gpu]   PS user_data:");
      for (int k = 0; k < 16; k++) std::fprintf(stderr, " %08x", pud[k]);
      std::fprintf(stderr, "\n");
      auto texs = gcn::trackTextures(reinterpret_cast<const uint32_t *>(psA), 256, pud);
      std::fprintf(stderr, "[gpu]   trackTextures -> %zu\n", texs.size());
      if (!texs.empty()) {
        auto &t = texs[0];
        // Dump the texture as a LINEAR interpretation (to inspect tiling).
        auto *px = reinterpret_cast<const uint8_t *>(t.base);
        FILE *f = std::fopen("/tmp/tex_raw.bin", "wb");
        if (f) {
          std::fwrite(px, 1, (size_t)t.width * t.height * 4, f);
          std::fclose(f);
          std::fprintf(stderr, "[gpu]   dumped /tmp/tex_raw.bin (%ux%u rgba)\n",
                       t.width, t.height);
        }
      }
    }
  }
  if (vk::available()) {
    const uint32_t *vud = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
    vk::DrawInfo d;
    d.primType = g_regs[mmVGT_PRIMITIVE_TYPE];
    d.instanceCount = g_numInstances;
    // Index buffer. DRAW_INDEX_2 (5 dwords): maxSize, baseLo, baseHi, indexCount,
    // drawInitiator. The draws are indexed triangle lists; without the index
    // buffer (drawing raw vertices as a strip) batched sprites smear into long
    // diagonal triangles. DRAW_INDEX_AUTO has no index buffer (sequential verts).
    if (op == IT_DRAW_INDEX_2 && count >= 4) {
      uint64_t ibase = (static_cast<uint64_t>(body[2] & 0xFF) << 32) | body[1];
      uint32_t icount = body[3];
      if (ibase >= 0x1000000000ull && ibase < 0x20000000000ull && icount &&
          icount <= 0x100000) {
        d.indexData = reinterpret_cast<const void *>(ibase);
        d.indexCount = icount;
        d.indexType = g_indexType;  // 0 = 16-bit, 1 = 32-bit
      }
    }
    d.rtBase = g_regs.cbColorBase(0);
    d.rtW = fbWidth();
    d.rtH = fbHeight();

    // Per-draw blend state from CB_BLEND0_CONTROL. Bit 30 is the per-target blend
    // enable; when clear the draw writes opaquely (no blend). Isaac's room
    // darkness/vignette and additive effect overlays rely on this: rendered with
    // a single hardcoded blend they came out opaque and blacked out the scene.
    d.blendControl = g_regs[mmCB_BLEND0_CONTROL];
    d.blendEnable = (d.blendControl >> 30) & 1u;
    // Per-MRT channel write mask (MRT0 = bits[3:0]) and overall colour-control mode.
    d.targetMask = g_regs[mmCB_TARGET_MASK];
    d.colorControl = g_regs[mmCB_COLOR_CONTROL];

    // MVP matrix: the VS sgpr[4..7] V# points at the constant buffer whose first
    // 16 floats are the transform (s_buffer_load_dwordx16 in the VS).
    uint64_t cbuf = (static_cast<uint64_t>(vud[5] & 0xFFFF) << 32) | vud[4];
    if (cbuf >= 0x1000000000ull && cbuf < 0x20000000000ull)
      std::memcpy(d.mvp, reinterpret_cast<const void *>(cbuf), 64);

    // Vertex buffer: resource-track the fetch shader (VS sgpr[0..1] ptr).
    uint64_t fetch = (static_cast<uint64_t>(vud[1] & 0xFFFF) << 32) | vud[0];
    if (fetch >= 0x1000000000ull && fetch < 0x20000000000ull) {
      auto vbs = gcn::trackVertexBuffers(reinterpret_cast<const uint32_t *>(fetch),
                                         64, vud);
      if (!vbs.empty()) {
        d.vertexData = reinterpret_cast<const void *>(vbs[0].base);
        d.vertexCount = vbs[0].numRecords;
        d.vertexStride = vbs[0].stride;
        d.posOffset = 0;
        // Per-attribute offsets from the fetch shader's V# bases: each attribute's
        // V# points at vertexBase + attributeOffset, so off = vb.base - pos.base.
        // dfmt 11 = 32_32 (float2 uv), dfmt 10 = 8_8_8_8 (rgba8 colour). This
        // generalises the old hardcoded sprite offsets (pos@0,color@0x10,uv@0x1c)
        // so other vertex layouts (e.g. the stride-36 room floor) sample the right
        // attributes instead of garbage. Falls back to the sprite offsets.
        uint32_t uvOff = 0, colOff = 0;
        for (size_t i = 1; i < vbs.size(); i++) {
          uint64_t off = vbs[i].base - vbs[0].base;
          if (off == 0 || off >= d.vertexStride) continue;
          if (vbs[i].dfmt == 11 && !uvOff) uvOff = static_cast<uint32_t>(off);
          else if (vbs[i].dfmt == 10 && !colOff) colOff = static_cast<uint32_t>(off);
        }
        d.uvOffset = uvOff ? uvOff : (d.vertexStride >= 0x1c ? 0x1c : 0);
        d.colorOffset = colOff ? colOff : (d.vertexStride >= 0x1c ? 0x10 : 0xFFFFFFFFu);
      }
    }

    // Texture: the PS samples an inline T# (image_sample). Isaac's textured
    // sprite vertex format is {pos.xyzw @0, color @0x10, uv.xy @0x1c} in a
    // 64-byte vertex, so the UV lives in the position buffer at offset 0x1c.
    if (psA >= 0x1000000000ull && psA < 0x20000000000ull && d.vertexData) {
      auto texs = gcn::trackTextures(reinterpret_cast<const uint32_t *>(psA), 256,
                                     &g_regs[mmSPI_SHADER_USER_DATA_PS_0]);
      if (!texs.empty()) {
        d.texBase = texs[0].base;
        d.texW = texs[0].width;
        d.texH = texs[0].height;
        d.uvData = d.vertexData;
        d.uvStride = d.vertexStride;
        // d.uvOffset was derived from the fetch shader during vertex extraction.
      }
    }
    // Floor trace (DELTA_GPU_FLOORTRACE): log draws into a room-sized (~832w) RT,
    // including DROPPED ones (no vertex data), to find why the room floor (the
    // large centre fill) is missing while the walls render.
    static const bool floorTrace = std::getenv("DELTA_GPU_FLOORTRACE") != nullptr;
    static int g_floorN = 0;
    if (floorTrace && d.rtW >= 700 && d.rtW <= 900 && g_floorN < 60) {
      g_floorN++;
      // For a textured floor draw, sample the texture: count non-zero pixels and
      // the uv range, and one-time dump the raw bytes so the texture content
      // (black/unloaded vs garbled/tiled vs dark) can be inspected.
      int nz = 0; float uvmin = 1e9f, uvmax = -1e9f;
      if (d.texBase >= 0x1000000000ull && d.texW && d.texH) {
        auto *px = reinterpret_cast<const uint32_t *>(d.texBase);
        uint64_t cnt = (uint64_t)d.texW * d.texH, step = cnt > 4096 ? cnt / 4096 : 1;
        for (uint64_t i = 0; i < cnt; i += step) if (px[i] & 0x00FFFFFF) nz++;
        if (d.uvData) {
          auto *vb = static_cast<const uint8_t *>(d.uvData);
          for (uint32_t v = 0; v < d.vertexCount && v < 64; v++) {
            auto *u = reinterpret_cast<const float *>(vb + (size_t)v * d.uvStride + d.uvOffset);
            uvmin = u[0] < uvmin ? u[0] : uvmin; uvmax = u[0] > uvmax ? u[0] : uvmax;
            uvmin = u[1] < uvmin ? u[1] : uvmin; uvmax = u[1] > uvmax ? u[1] : uvmax;
          }
        }
        static bool dumped = false;
        if (!dumped && d.texW == 512 && d.texH == 512) {
          dumped = true;
          FILE *f = std::fopen("/tmp/floor_tex.bin", "wb");
          if (f) { std::fwrite(px, 1, (size_t)d.texW * d.texH * 4, f); std::fclose(f); }
        }
      }
      // NDC bounds of the floor geometry (apply the extracted MVP), to tell a
      // wrong-MVP (off-screen / degenerate) from a sampling problem.
      float nx0=1e9f,ny0=1e9f,nx1=-1e9f,ny1=-1e9f;
      if (d.vertexData) {
        auto *vb = static_cast<const uint8_t *>(d.vertexData);
        const float *m = d.mvp;
        for (uint32_t v = 0; v < d.vertexCount && v < 64; v++) {
          auto *p = reinterpret_cast<const float *>(vb + (size_t)v * d.vertexStride);
          float cw = m[3]*p[0]+m[7]*p[1]+m[15]; if (cw==0) cw=1;
          float ndx=(m[0]*p[0]+m[4]*p[1]+m[12])/cw, ndy=(m[1]*p[0]+m[5]*p[1]+m[13])/cw;
          nx0=ndx<nx0?ndx:nx0; nx1=ndx>nx1?ndx:nx1; ny0=ndy<ny0?ndy:ny0; ny1=ndy>ny1?ndy:ny1;
        }
      }
      std::fprintf(stderr, "[floor] rt=%#lx %ux%u stride=%u nrec=%u uvOff=%u tex=%#lx "
                   "%ux%u nz=%d/4096 uv[%.2f..%.2f] ndc[%.2f,%.2f..%.2f,%.2f] "
                   "mvp=[%.3f %.3f %.3f] blend=%#x\n",
                   (unsigned long)d.rtBase, d.rtW, d.rtH, d.vertexStride,
                   d.vertexCount, d.uvOffset, (unsigned long)d.texBase, d.texW, d.texH,
                   nz, uvmin, uvmax, nx0,ny0,nx1,ny1, d.mvp[0],d.mvp[5],d.mvp[12],
                   d.blendControl);
    }
    static int g_uvDbg = 0;
    if (g_trace && d.texBase && g_uvDbg < 3 && d.vertexData) {
      g_uvDbg++;
      std::fprintf(stderr, "[gpu] TEX draw tex=%#lx %ux%u nv=%u posStride=%u "
                   "uvStride=%u\n", (unsigned long)d.texBase, d.texW, d.texH,
                   d.vertexCount, d.vertexStride, d.uvStride);
      auto *ps = reinterpret_cast<const float *>(d.vertexData);
      auto *uv = reinterpret_cast<const float *>(d.uvData);
      std::fprintf(stderr, "[gpu]   VB0 base=%p VB1 base=%p\n", d.vertexData, d.uvData);
      for (uint32_t v = 0; v < 4; v++) {
        std::fprintf(stderr, "[gpu]   v%u VB0[", v);
        for (int c = 0; c < 16; c++) std::fprintf(stderr, "%g ", ps[v*16+c]);
        std::fprintf(stderr, "] VB1[");
        if (uv) for (int c = 0; c < 16; c++) std::fprintf(stderr, "%g ", uv[v*16+c]);
        std::fprintf(stderr, "]\n");
      }
    }
    // Recompiled-shader path: recompile the VS/PS pair (cached) and resolve the
    // live vertex-attribute buffers, so the renderer can run the game's actual
    // shaders. The heuristic fields above stay populated as the fallback.
    static const bool recompOn = [] {
      const char *e = std::getenv("DELTA_GPU_RECOMP");
      return !e || std::strcmp(e, "0") != 0;
    }();
    if (recompOn && vsA >= 0x1000000000ull && vsA < 0x20000000000ull &&
        psA >= 0x1000000000ull && psA < 0x20000000000ull) {
      static std::unordered_map<uint64_t, gcn::Recompiled> shCache;
      uint64_t key = vsA * 0x9e3779b97f4a7c15ull ^ psA;
      auto it = shCache.find(key);
      if (it == shCache.end())
        it = shCache.emplace(key, gcn::recompile(
                 reinterpret_cast<const uint32_t *>(vsA),
                 reinterpret_cast<const uint32_t *>(psA),
                 &g_regs[mmSPI_SHADER_USER_DATA_VS_0],
                 &g_regs[mmSPI_SHADER_USER_DATA_PS_0])).first;
      gcn::Recompiled &rc = it->second;
      if (rc.ok && !rc.attrs.empty()) {
        const uint32_t *vud2 = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
        const void *heurVtx0 = d.vertexData; uint32_t heurStride0 = d.vertexStride;
        uint64_t base0 = 0;
        bool good = true;
        for (size_t i = 0; i < rc.attrs.size() && i < 8; i++) {
          auto &a = rc.attrs[i];
          uint64_t tbl = (static_cast<uint64_t>(vud2[a.tableSgpr + 1] & 0xFFFF) << 32) | vud2[a.tableSgpr];
          if (tbl < 0x1000000000ull || tbl >= 0x20000000000ull) { good = false; break; }
          auto vb = gcn::decodeVBuffer(reinterpret_cast<const uint32_t *>(tbl + a.vbufDwordOff * 4));
          if (vb.base < 0x1000000000ull || vb.base >= 0x20000000000ull || !vb.stride) { good = false; break; }
          if (i == 0) { base0 = vb.base; d.vertexData = reinterpret_cast<const void *>(vb.base);
            d.vertexStride = vb.stride; d.vertexCount = vb.numRecords; }
          uint32_t off = (vb.base >= base0) ? (uint32_t)(vb.base - base0) : 0;
          d.vattrs[d.nvattrs++] = {a.location, off, a.numComps, vb.dfmt, vb.nfmt};
          static int rcvN = 0;
          if (std::getenv("DELTA_GPU_SHTRACE") && rcvN < 16) {
            rcvN++;
            std::fprintf(stderr, "[rcv] attr%zu loc=%u tblSgpr=%u dwOff=%u tbl=%#lx "
                         "vb.base=%#lx stride=%u nrec=%u dfmt=%u off=%u | heurVtx=%p heurStride=%u\n",
                         i, a.location, a.tableSgpr, a.vbufDwordOff, (unsigned long)tbl,
                         (unsigned long)vb.base, vb.stride, vb.numRecords, vb.dfmt, off,
                         heurVtx0, heurStride0);
          }
        }
        if (good && d.nvattrs) { d.vsAddr = vsA; d.psAddr = psA; d.recomp = &rc; }
        else d.nvattrs = 0;
      }
    }
    if (!g_frameActive) {
      vk::beginFrame();
      g_frameActive = true;
    }
    // All-draws map (DELTA_GPU_ALLDRAWS): once a busy frame (>15 draws into a room-
    // sized RT) starts, log every draw's target RT, texture, vert/index count and
    // shader for two frames, to see the complete layer structure (which RT the
    // floor fill targets).
    static const bool allDraws = std::getenv("DELTA_GPU_ALLDRAWS") != nullptr;
    static int adFrame = 0, adSeen = 0;
    if (allDraws) {
      if (d.rtW >= 700 && d.rtW <= 900) adSeen++;
      if (adSeen >= 1 && adFrame < 80) {
        adFrame++;
        std::fprintf(stderr, "[ad] rt=%#lx %ux%u vs=%#lx tex=%#lx %ux%u nattr=%u stride=%u "
                     "ic=%u vc=%u prim=%u inst=%u\n",
                     (unsigned long)d.rtBase, d.rtW, d.rtH, (unsigned long)vsA,
                     (unsigned long)d.texBase, d.texW, d.texH, d.nvattrs, d.vertexStride,
                     d.indexCount, d.vertexCount, d.primType, d.instanceCount);
      }
    }
    // Dropped-draw trace (DELTA_GPU_DROPS): log draws we can't issue (no vertex
    // data resolved from the fetch shader) once gameplay is up, to find a missing
    // floor fill that never reaches the renderer.
    if (!d.vertexData && std::getenv("DELTA_GPU_DROPS")) {
      static int dn = 0;
      if (dn++ < 60)
        std::fprintf(stderr, "[drop] vs=%#lx ps=%#lx rt=%#lx %ux%u ic=%u prim=%u fetch=%#lx\n",
                     (unsigned long)vsA, (unsigned long)psA, (unsigned long)d.rtBase,
                     d.rtW, d.rtH, d.indexCount, d.primType, (unsigned long)fetch);
    }
    if (d.vertexData)
      vk::draw(d);
  }
  if (!g_trace)
    return;
  uint64_t cb = g_regs.cbColorBase(0);
  uint32_t cbInfo = g_regs[mmCB_COLOR0_INFO];
  uint32_t cbAttrib = g_regs[mmCB_COLOR0_ATTRIB];
  uint64_t vs = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_VS);
  uint64_t ps = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_PS);
  uint32_t prim = g_regs[mmVGT_PRIMITIVE_TYPE];
  uint32_t scTL = g_regs[mmPA_SC_SCREEN_SCISSOR_TL];
  uint32_t scBR = g_regs[mmPA_SC_SCREEN_SCISSOR_BR];
  uint32_t indices = (count >= 1) ? body[0] : 0;
  std::fprintf(stderr,
               "[gpu] DRAW op=%#x prim=%u indices=%u | RT=%#lx info=%#x "
               "attrib=%#x scissor=[%u,%u..%u,%u] VS=%#lx PS=%#lx\n",
               op, prim, indices, (unsigned long)cb, cbInfo, cbAttrib,
               scTL & 0xFFFF, scTL >> 16, scBR & 0xFFFF, scBR >> 16,
               (unsigned long)vs, (unsigned long)ps);

  // One-time: locate the embedded "OrbShdr" BinaryInfo in the VS/PS GCN code to
  // confirm the recompiler can find shader length+hash. Layout: if code[0] ==
  // 0xBEEB03FF the info is at code + (code[1]+1)*2 dwords; else scan for the
  // 7-byte signature {'O','r','b','S','h','d','r'}.
  static bool shaderProbed = false;
  if (!shaderProbed && vs && ps) {
    shaderProbed = true;
    auto probe = [](const char *tag, uint64_t addr) {
      auto *code = reinterpret_cast<const uint32_t *>(addr);
      const uint8_t *info = nullptr;
      if (code[0] == 0xBEEB03FFu)
        info = reinterpret_cast<const uint8_t *>(code + (code[1] + 1) * 2);
      else {
        auto *b = reinterpret_cast<const uint8_t *>(code);
        for (int k = 0; k < 0x4000; k++)
          if (std::memcmp(b + k, "OrbShdr", 7) == 0) { info = b + k; break; }
      }
      if (info) {
        uint32_t lenField; std::memcpy(&lenField, info + 8, 4);
        uint64_t hash; std::memcpy(&hash, info + 0xC, 8);  // approx offsets
        std::fprintf(stderr, "[gpu]   %s shader @%#lx OrbShdr len=%u hash=%#lx\n",
                     tag, (unsigned long)addr, lenField & 0xFFFFFF,
                     (unsigned long)hash);
      } else {
        std::fprintf(stderr, "[gpu]   %s shader @%#lx: no OrbShdr (code0=%#x)\n",
                     tag, (unsigned long)addr, code[0]);
      }
    };
    probe("VS", vs);
    probe("PS", ps);

    // Dump the user-data SGPRs and decode candidate V#/T#/S# descriptors. A
    // dword pair that forms a plausible guest pointer (0x10_0000_0000.. range)
    // is likely a descriptor or a pointer to a descriptor table.
    auto dumpUd = [](const char *tag, const uint32_t *ud) {
      std::fprintf(stderr, "[gpu]   %s user_data:", tag);
      for (int k = 0; k < 16; k++)
        std::fprintf(stderr, " %08x", ud[k]);
      std::fprintf(stderr, "\n");
      // Decode any 4-dword group as a V# (buffer): base44, stride, num_records.
      for (int k = 0; k + 1 < 16; k += 2) {
        uint64_t base = ((uint64_t)(ud[k + 1] & 0xFFF) << 32) | ud[k];
        if (base >= 0x1000000000ull && base < 0x20000000000ull) {
          uint32_t stride = (ud[k + 1] >> 16) & 0x3FFF;
          uint32_t nrec = ud[k + 2];
          std::fprintf(stderr, "[gpu]     sgpr[%d..]: ptr=%#lx stride=%u nrec=%u fmt=%#x\n",
                       k, (unsigned long)base, stride, nrec, ud[k + 3]);
          // A small vertex buffer (a quad): dump it as floats to learn the layout.
          if (stride && stride <= 64 && nrec && nrec <= 8) {
            auto *f = reinterpret_cast<const float *>(base);
            auto *u = reinterpret_cast<const uint32_t *>(base);
            for (uint32_t v = 0; v < nrec; v++) {
              std::fprintf(stderr, "[gpu]       v%u:", v);
              for (uint32_t c = 0; c < stride / 4; c++)
                std::fprintf(stderr, " %g(%08x)", f[v * (stride / 4) + c],
                             u[v * (stride / 4) + c]);
              std::fprintf(stderr, "\n");
            }
          }
        }
      }
    };
    dumpUd("VS", &g_regs[mmSPI_SHADER_USER_DATA_VS_0]);
    dumpUd("PS", &g_regs[mmSPI_SHADER_USER_DATA_PS_0]);

    // Follow the descriptor-table pointers in the user-data SGPRs and decode the
    // V#/T#/S# sharps inside. A V# (4 dwords): base48, stride, num_records. A T#
    // (8 dwords): base + width/height. This is where the real vertex buffer and
    // texture atlas live for the quad draws.
    auto dumpTable = [](const char *tag, uint64_t ptr) {
      if (ptr < 0x1000000000ull || ptr >= 0x20000000000ull)
        return;
      auto *t = reinterpret_cast<const uint32_t *>(ptr);
      std::fprintf(stderr, "[gpu]   table %s @%#lx:\n", tag, (unsigned long)ptr);
      for (int k = 0; k < 32; k += 4) {
        uint64_t b = ((uint64_t)(t[k + 1] & 0xFFFF) << 32) | t[k];
        uint32_t stride = (t[k + 1] >> 16) & 0x3FFF;
        // V# heuristic
        if (b >= 0x1000000000ull && b < 0x20000000000ull && stride &&
            stride <= 256)
          std::fprintf(stderr, "[gpu]     +%02x V#? base=%#lx stride=%u nrec=%u dfmt=%#x\n",
                       k * 4, (unsigned long)b, stride, t[k + 2], t[k + 3]);
        // T# heuristic: dword2 has width-1[0:13], height-1[14:27]
        uint64_t tb = ((uint64_t)(t[k + 1] & 0xFFFFFF) << 32) | t[k];
        uint32_t w = (t[k + 2] & 0x3FFF) + 1, h = ((t[k + 2] >> 14) & 0x3FFF) + 1;
        if (tb >= 0x1000000000ull && tb < 0x20000000000ull && w > 4 && w <= 8192 &&
            h > 4 && h <= 8192)
          std::fprintf(stderr, "[gpu]     +%02x T#? base=%#lx %ux%u dfmt=%#x\n",
                       k * 4, (unsigned long)tb, w, h, (t[k + 1] >> 20) & 0x3F);
      }
    };
    const uint32_t *vud = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
    dumpTable("VS.sgpr0", ((uint64_t)(vud[1] & 0xFFFF) << 32) | vud[0]);
    dumpTable("VS.sgpr2", ((uint64_t)(vud[3] & 0xFFFF) << 32) | vud[2]);

    // Disassemble the shaders to validate the GCN decoder and reveal the
    // vertex-fetch / resource-load pattern. The fetch shader (sgpr0 ptr, just
    // past the VS code) does the s_load(V# table) + buffer_load(attributes).
    gcn::disassemble(reinterpret_cast<const uint32_t *>(vs), 512, "VS");
    gcn::disassemble(reinterpret_cast<const uint32_t *>(ps), 512, "PS");
    uint64_t fetch = ((uint64_t)(vud[1] & 0xFFFF) << 32) | vud[0];
    if (fetch >= 0x1000000000ull && fetch < 0x20000000000ull) {
      gcn::disassemble(reinterpret_cast<const uint32_t *>(fetch), 128, "VS.fetch");
      // Recover the actual vertex-attribute buffers and dump the first vertices.
      auto vbs = gcn::trackVertexBuffers(reinterpret_cast<const uint32_t *>(fetch),
                                         64, vud);
      for (size_t bi = 0; bi < vbs.size(); bi++) {
        auto &v = vbs[bi];
        std::fprintf(stderr, "[gpu]   VB%zu base=%#lx stride=%u nrec=%u\n", bi,
                     (unsigned long)v.base, v.stride, v.numRecords);
        auto *f = reinterpret_cast<const float *>(v.base);
        for (uint32_t r = 0; r < v.numRecords && r < 6; r++) {
          std::fprintf(stderr, "[gpu]     r%u:", r);
          for (uint32_t c = 0; c < v.stride / 4 && c < 8; c++)
            std::fprintf(stderr, " %g", f[r * (v.stride / 4) + c]);
          std::fprintf(stderr, "\n");
        }
      }
    }
  }
}

}  // namespace

// Called by the Gnm HLE on submit-and-flip: finish the frame and present the
// render target the flip displays.
void endFrame(uint64_t scanoutBase) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_frameActive && vk::available()) {
    vk::endFrame(scanoutBase);
    g_frameActive = false;
  }
}

void submitDcb(const void *dcb, uint32_t sizeBytes) {
  if (!dcb || sizeBytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!g_vkTried) {
    g_vkTried = true;
    vk::init();
  }
  auto *p = static_cast<const uint32_t *>(dcb);
  uint32_t words = sizeBytes / 4;
  uint64_t sn = g_totalSubmits.fetch_add(1) + 1;
  if (g_trace && (sn <= 8 || sn % 256 == 0))
    std::fprintf(stderr, "[gpu] submit #%lu size=%u draws-so-far=%lu\n",
                 (unsigned long)sn, sizeBytes,
                 (unsigned long)g_totalDraws.load());
  // Dump the full packet walk of the first large (real rendering) command
  // buffer so we can see its opcodes / find the draw.
  static bool dumpedBig = false;
  bool dumpThis = g_trace && !dumpedBig && sizeBytes > 4000;
  if (dumpThis) {
    dumpedBig = true;
    std::fprintf(stderr, "[gpu] === big dcb walk (size=%u) ===\n", sizeBytes);
  }
  if (g_trace && g_dcbSeen < 6)
    std::fprintf(stderr, "[gpu] submitDcb dcb=%p sizeBytes=%u words=%u hdr0=%#x\n",
                 dcb, sizeBytes, words, p[0]);
  uint32_t i = 0;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = pm4Type(hdr);
    if (type == Pm4Type::type3) {
      uint32_t op = pm4Opcode(hdr);
      uint32_t count = pm4Count(hdr);  // body dword count
      const uint32_t *body = &p[i + 1];
      if (g_trace) {
        g_opHist[op & 0xFF]++;
        if (dumpThis)
          std::fprintf(stderr, "[gpu]   @%-5u T3 op=%#04x count=%u\n", i, op, count);
      }
      if (i + 1 + count > words)
        break;  // truncated / desync
      switch (op) {
      case IT_SET_CONTEXT_REG: setRegs(kContextRegBase, body, count); break;
      case IT_SET_SH_REG:      setRegs(kShRegBase, body, count); break;
      case IT_SET_UCONFIG_REG: setRegs(kUConfigRegBase, body, count); break;
      case IT_SET_CONFIG_REG:  setRegs(kConfigRegBase, body, count); break;
      case IT_INDEX_TYPE:  // VGT_DMA_INDEX_TYPE: bits[1:0] 0=16-bit 1=32-bit
        if (count >= 1) g_indexType = body[0] & 0x3;
        break;
      case IT_INDEX_BASE:  // index buffer base (byte address) lo/hi
        if (count >= 2)
          g_indexBase = (static_cast<uint64_t>(body[1] & 0xFF) << 32) | body[0];
        break;
      case IT_NUM_INSTANCES:  // instance count for the following draw(s)
        g_numInstances = (count >= 1 && body[0]) ? body[0] : 1;
        break;
      case IT_DMA_DATA:  // GPU memory->memory copy (CP DMA). body: ctrl, srcLo,
                         // srcHi, dstLo, dstHi, command(byteCount[20:0]).
        if (std::getenv("DELTA_GPU_DMATRACE") && count >= 5) {
          static int dmn = 0;
          uint64_t src = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
          uint64_t dst = (static_cast<uint64_t>(body[4] & 0xFFFF) << 32) | body[3];
          uint32_t bytes = count >= 6 ? (body[5] & 0x1FFFFF) : 0;
          if (dmn++ < 80)
            std::fprintf(stderr, "[dma] ctrl=%#x src=%#lx dst=%#lx bytes=%u\n",
                         body[0], (unsigned long)src, (unsigned long)dst, bytes);
        }
        break;
      default:
        if (isDraw(op)) {
          g_totalDraws.fetch_add(1);
          handleDraw(op, body, count);
        }
        break;
      }
      i += 1 + count;
    } else if (type == Pm4Type::type2 || hdr == 0) {
      // Single-dword filler: type-2 NOPs and zero-dword alignment padding that
      // Gnm sprinkles between packets. Skip and keep walking (these are NOT the
      // end of the buffer (real packets resume after the padding).
      i += 1;
    } else {
      // A non-zero type-0 / type-1 header is a genuine desync; stop.
      if (dumpThis)
        std::fprintf(stderr, "[gpu]   @%-5u STOP type%u hdr=%#x\n", i,
                     (uint32_t)type, hdr);
      break;
    }
  }
  if (dumpThis) {
    std::fprintf(stderr, "[gpu] === big dcb walk done: %u/%u words ===\n", i, words);
    // Brute-scan the whole buffer for draw-opcode headers (in case the walker
    // desynced and missed a draw), and dump raw words around the stop point.
    int found = 0;
    for (uint32_t w = 0; w < words; w++) {
      uint32_t h = p[w];
      if ((h >> 30) == 3) {
        uint32_t o = (h >> 8) & 0xFF;
        if (o == 0x2D || o == 0x27 || o == 0x35 || o == 0x30 || o == 0x15) {
          std::fprintf(stderr, "[gpu]   SCAN found draw op=%#x @word %u\n", o, w);
          if (++found > 8) break;
        }
      }
    }
    if (!found)
      std::fprintf(stderr, "[gpu]   SCAN: no draw opcode anywhere in %u words\n", words);
    std::fprintf(stderr, "[gpu]   raw[255..270]:");
    for (uint32_t w = 255; w < 271 && w < words; w++)
      std::fprintf(stderr, " %08x", p[w]);
    std::fprintf(stderr, "\n");
  }
  if (g_trace && g_dcbSeen < 4)
    std::fprintf(stderr, "[gpu] dcb done: walked %u/%u words\n", i, words);
  if (g_trace && ++g_dcbSeen <= 4)
    dumpHist();
}

}  // namespace gpu
