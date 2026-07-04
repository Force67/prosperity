/*
 * PS4Delta : PS4 emulation and research project
 *
 * GPU command processor. See cmd_processor.h.
 */

#include <chrono>
#include "cmd_processor.h"
#include "pm4.h"
#include "liverpool.h"
#include "vk_render.h"
#include "gcn/gcn_decode.h"
#include "gcn/gcn_resource.h"
#include "gcn/gcn_interp.h"
#include "gcn/gcn_translate.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
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

// GPU completion-label writes (EOP / RELEASE_MEM / WRITE_DATA). Our submit is
// synchronous: every draw in the dcb is finished by the time we walk past these
// packets, so the fence/label the GPU would signal is complete the instant we
// process it. Writing it immediately is what lets the guest's CPU-side polls --
// the flip-done / submit-done labels Gnm spins on between frames -- make
// progress. Without it the title stalls after the few in-flight display buffers
// drain (it never sees a flip "complete").
const bool g_eopTrace = std::getenv("DELTA_GPU_EOPTRACE") != nullptr;
// Labels live in guest memory the game allocated (Garlic/Onion 0x10_0000_0000+),
// in low guest heaps (0x2_0000_0000+), or in the GnmDriver area (0xfe00_0000+).
// Accept any plausibly-mapped, non-low address; reject null/garbage.
inline bool labelAddrOk(uint64_t a) {
  return a >= 0x10000ull && a < 0x20000000000ull;
}
void writeLabel(uint64_t addr, uint64_t value, bool is64) {
  if (!labelAddrOk(addr))
    return;
  if (is64)
    *reinterpret_cast<volatile uint64_t *>(addr) = value;
  else
    *reinterpret_cast<volatile uint32_t *>(addr) = static_cast<uint32_t>(value);
}

// EOP/RELEASE_MEM DATA_SEL 3 (GPU clock) and 4 (system clock) tell the GPU to
// write its current 64-bit clock counter into the label, NOT the packet's
// immediate data (which is 0 for these). A title that polls such a label for
// "non-zero == the GPU reached this point" needs a real, monotonically-increasing,
// non-zero value. Our submit is synchronous, so any advancing clock reads as
// "already complete". Without this Doom64's per-frame submit-done wait (a spin on
// this label with a hard 2s timeout) burns the full 2s every frame -> ~0.5 fps.
uint64_t gpuClockTs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
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
      auto texs = gcn::trackTextures(reinterpret_cast<const uint32_t *>(psA), 4096, pud);
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
    d.mrtBase[0] = d.rtBase;
    // Multiple render targets: CB_COLOR1..7. A target is bound when its CB_TARGET_MASK
    // nibble (4 bits per MRT) is set AND its base is a valid guest address. mrtCount is
    // the highest bound index + 1 (stays 1 for the usual single-RT case).
    {
      uint32_t tmask = g_regs[mmCB_TARGET_MASK];
      for (int rt = 1; rt < 8; rt++) {
        uint64_t base = g_regs.cbColorBase(rt);
        if (((tmask >> (rt * 4)) & 0xF) && base >= 0x1000000000ull &&
            base < 0x20000000000ull) {
          d.mrtBase[rt] = base;
          d.mrtCount = rt + 1;
        }
      }
    }

    // Per-draw blend state from CB_BLEND0_CONTROL. Bit 30 is the per-target blend
    // enable; when clear the draw writes opaquely (no blend). Isaac's room
    // darkness/vignette and additive effect overlays rely on this: rendered with
    // a single hardcoded blend they came out opaque and blacked out the scene.
    d.blendControl = g_regs[mmCB_BLEND0_CONTROL];
    d.blendEnable = (d.blendControl >> 30) & 1u;
    // Per-MRT channel write mask (MRT0 = bits[3:0]) and overall colour-control mode.
    d.targetMask = g_regs[mmCB_TARGET_MASK];
    d.colorControl = g_regs[mmCB_COLOR_CONTROL];

    // Constant buffer (transform): default to the sgpr[4..7] V# (the common VS cbuffer
    // slot); the recompiled-shader path below re-resolves it from the SGPR the VS
    // actually reads (rc.vsCbufs) when that differs.
    uint64_t cbuf = (static_cast<uint64_t>(vud[5] & 0xFFFF) << 32) | vud[4];
    if (cbuf >= 0x1000000000ull && cbuf < 0x20000000000ull) {
      std::memcpy(d.mvp, reinterpret_cast<const void *>(cbuf), 64);
      d.cbufBase = cbuf;
    }

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
      auto texs = gcn::trackTextures(reinterpret_cast<const uint32_t *>(psA), 4096,
                                     &g_regs[mmSPI_SHADER_USER_DATA_PS_0]);
      if (!texs.empty()) {
        d.texBase = texs[0].base;
        d.texW = texs[0].width;
        d.texH = texs[0].height;
        d.texTiling = texs[0].tilingIdx;
        d.texPitch = texs[0].pitch;
        d.uvData = d.vertexData;
        d.uvStride = d.vertexStride;
        // All sampled textures (binding order), for multi-texture PS (Doom64 3D).
        d.nTexs = static_cast<uint32_t>(texs.size() < 8 ? texs.size() : 8);
        for (uint32_t i = 0; i < d.nTexs; i++)
          d.texs[i] = {texs[i].base, texs[i].width, texs[i].height,
                       texs[i].tilingIdx, texs[i].pitch};
        // d.uvOffset was derived from the fetch shader during vertex extraction.
        // DELTA_GPU_TEXFMT: dump sampled texture formats (dfmt/nfmt/tiling/dims) to
        // pin a scrambled draw (e.g. Doom64's menu) to a format/tiling we mishandle.
        static const bool texfmt = std::getenv("DELTA_GPU_TEXFMT") != nullptr;
        static int tfN = 0;
        if (texfmt && tfN < 24) {
          tfN++;
          std::fprintf(stderr, "[texfmt] base=%#lx %ux%u pitch=%u dfmt=%u nfmt=%u tiling=%u rt=%ux%u\n",
                       (unsigned long)texs[0].base, texs[0].width, texs[0].height,
                       texs[0].pitch, texs[0].dfmt, texs[0].nfmt, texs[0].tilingIdx,
                       d.rtW, d.rtH);
        }
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
        }
        // Constant buffer: the recompiled VS reads its cbuffer via push constants, so
        // resolve it from the user-data SGPR the VS ACTUALLY uses (recorded in vsCbufs)
        // rather than the hardcoded sgpr[4..7] above. For the common sprite VS this is
        // sgpr 4 (identical to the default), but shaders whose cbuffer V# lives elsewhere
        // (e.g. composite/post-process VS) then get the correct transform instead of the
        // wrong one -- the general fix, not an Isaac special-case.
        if (good && d.nvattrs && !rc.vsCbufs.empty()) {
          uint32_t cbSgpr = rc.vsCbufs[0].udSgpr;
          if (cbSgpr + 2 < 16) {
            uint64_t vbase = (static_cast<uint64_t>(vud2[cbSgpr + 1] & 0xFFFF) << 32) | vud2[cbSgpr];
            if (vbase >= 0x1000000000ull && vbase < 0x20000000000ull) {
              std::memcpy(d.mvp, reinterpret_cast<const void *>(vbase), 64);
              // Full cbuffer base+size for the set-1 UBO. V#: stride[29:16] of word1,
              // num_records word2; byte size = stride ? stride*records : records.
              d.cbufBase = vbase;
              uint32_t stride = (vud2[cbSgpr + 1] >> 16) & 0x3FFF;
              uint32_t records = vud2[cbSgpr + 2];
              d.cbufSize = stride ? stride * records : records;
            }
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
    // DELTA_GPU_BLITDUMP: for the first few draws targeting a wide (scanout-sized)
    // RT, disassemble the PS and report what trackTextures resolved. Pins why
    // Undertale's surface->scanout blit renders untextured (tex=0): is the PS doing
    // an image_sample we miss, or is the T# pointing outside the guest range?
    static const bool blitDump = std::getenv("DELTA_GPU_BLITDUMP") != nullptr;
    static int bdN = 0;
    if (blitDump && d.rtW >= 1280 && bdN < 6) {
      bdN++;
      std::fprintf(stderr,
          "[blit] #%d rt=%#lx %ux%u VS=%#lx PS=%#lx texBase=%#lx %ux%u nvattrs=%u "
          "stride=%u idx=%u blendCtl=%#x\n",
          bdN, (unsigned long)d.rtBase, d.rtW, d.rtH, (unsigned long)vsA,
          (unsigned long)psA, (unsigned long)d.texBase, d.texW, d.texH, d.nvattrs,
          d.vertexStride, d.indexCount, d.blendControl);
      if (psA >= 0x1000000000ull && psA < 0x20000000000ull) {
        auto texs = gcn::trackTextures(reinterpret_cast<const uint32_t *>(psA), 4096,
                                       &g_regs[mmSPI_SHADER_USER_DATA_PS_0]);
        std::fprintf(stderr, "[blit]   trackTextures -> %zu T#\n", texs.size());
        for (auto &t : texs)
          std::fprintf(stderr, "[blit]     T# base=%#lx %ux%u pitch=%u dfmt=%u nfmt=%u tiling=%u\n",
                       (unsigned long)t.base, t.width, t.height, t.pitch, t.dfmt,
                       t.nfmt, t.tilingIdx);
        gcn::disassemble(reinterpret_cast<const uint32_t *>(psA), 64, "blit.PS");
      }
    }
    // DELTA_GPU_DRAWLIST: one line per draw BEFORE any vertexData gating, so draws
    // dropped for null vertexData/recomp (e.g. Doom64's 3D world geometry) are
    // visible -- distinguishes "world draws never submitted" from "submitted but
    // dropped because vertex/shader resolution failed".
    static const bool drawList = std::getenv("DELTA_GPU_DRAWLIST") != nullptr;
    static int dlN = 0;
    // Wall-clock gate: the title/menu floods the early run, so only start logging
    // after DELTA_GPU_DRAWLIST_AFTER seconds (default 90), by when the level has
    // loaded -- then log EVERY draw so the in-level pattern (incl. world geometry,
    // if any reaches us) is captured.
    static const auto dlStart = std::chrono::steady_clock::now();
    static const int dlAfter = [] { const char *e = std::getenv("DELTA_GPU_DRAWLIST_AFTER");
      return e ? std::atoi(e) : 90; }();
    auto dlElapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - dlStart).count();
    if (drawList && dlElapsed >= dlAfter && dlN < 400) {
      dlN++;
      std::fprintf(stderr, "[draw] idx=%u rt=%#lx %ux%u tex=%#lx %ux%u vd=%d recomp=%d nvattrs=%u prim=%u\n",
                   d.indexCount, (unsigned long)d.rtBase, d.rtW, d.rtH,
                   (unsigned long)d.texBase, d.texW, d.texH, d.vertexData ? 1 : 0,
                   (d.recomp && d.recomp->ok) ? 1 : 0, d.nvattrs,
                   g_regs[mmVGT_PRIMITIVE_TYPE]);
    }
    // DELTA_GPU_SPRITEDUMP: for the first few TEXTURED draws, dump the resolved
    // transform + first vertex (pos/uv via the resolved attrs) + texture, to pin
    // why textured draws render black (degenerate MVP vs UV=0 vs blend).
    static const bool spriteDump = std::getenv("DELTA_GPU_SPRITEDUMP") != nullptr;
    static int sdN = 0;
    if (spriteDump && d.recomp && !d.recomp->psTexs.empty() && d.vertexData && sdN < 12) {
      sdN++;
      const float *m = d.mvp;
      std::fprintf(stderr,
          "[sprite] tex=%#lx %ux%u tiling=%u stride=%u nvattrs=%u rt=%#lx %ux%u "
          "blendCtl=%#x mvp=[%.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f / ... / "
          "%.3f %.3f %.3f %.3f]\n",
          (unsigned long)d.texBase, d.texW, d.texH, d.texTiling, d.vertexStride,
          d.nvattrs, (unsigned long)d.rtBase, d.rtW, d.rtH, d.blendControl,
          m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[12], m[13], m[14], m[15]);
      const auto *vb = static_cast<const uint8_t *>(d.vertexData);
      for (uint32_t a = 0; a < d.nvattrs && a < 4; a++) {
        const float *f = reinterpret_cast<const float *>(vb + d.vattrs[a].offset);
        std::fprintf(stderr, "[sprite]   attr%u loc=%u off=%u nc=%u dfmt=%u v0=[%.4f %.4f %.4f %.4f]\n",
            a, d.vattrs[a].location, d.vattrs[a].offset, d.vattrs[a].numComps,
            d.vattrs[a].dfmt, f[0], d.vattrs[a].numComps>1?f[1]:0.f,
            d.vattrs[a].numComps>2?f[2]:0.f, d.vattrs[a].numComps>3?f[3]:0.f);
      }
    }
    // DELTA_GPU_GEOMDUMP: dump the full state of high-index (3D level geometry)
    // draws to diagnose why Doom64's 3D renders black: textured vs vertex-color,
    // the texture format/tiling, the cbuffer transform, and a vertex position
    // (on-screen check). Gated on indexCount>=500 so it only fires in a 3D level.
    static const bool geomDump = std::getenv("DELTA_GPU_GEOMDUMP") != nullptr;
    // DELTA_GPU_GEOMMIN overrides the index-count gate (default 500) so the dump
    // can also catch Doom64's lower-index level draws.
    static const uint32_t geomMin = [] { const char *e = std::getenv("DELTA_GPU_GEOMMIN");
      return e ? (uint32_t)std::strtoul(e, nullptr, 10) : 500u; }();
    static int gdN = 0, gdSeen = 0;
    // Sample periodically across the WHOLE run (every 100th qualifying world draw)
    // so we can see whether the camera/view ever moves -- not just the first frames.
    if (geomDump && d.indexCount >= geomMin && d.vertexData &&
        (gdSeen++ % 100 == 0) && gdN < 300) {
      gdN++;
      const float *cb = d.cbufBase ? reinterpret_cast<const float *>(d.cbufBase) : nullptr;
      const auto *vb = static_cast<const uint8_t *>(d.vertexData);
      const float *p0 = reinterpret_cast<const float *>(vb + d.vattrs[0].offset);
      std::fprintf(stderr,
          "[geom] idx=%u psTexs=%zu tex=%#lx %ux%u tiling=%u pitch=%u rt=%#lx %ux%u "
          "blend=%#x mask=%#x cc=%#x nvattrs=%u stride=%u pos0=[%.1f %.1f %.1f] cbufBase=%#lx\n",
          d.indexCount, d.recomp ? d.recomp->psTexs.size() : 0, (unsigned long)d.texBase, d.texW, d.texH,
          d.texTiling, d.texPitch, (unsigned long)d.rtBase, d.rtW, d.rtH, d.blendControl,
          d.targetMask, d.colorControl, d.nvattrs, d.vertexStride,
          p0[0], p0[1], p0[2], (unsigned long)d.cbufBase);
      if (cb)
        std::fprintf(stderr, "  cbuf=[%.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f]\n",
            cb[0],cb[1],cb[2],cb[3],cb[4],cb[5],cb[6],cb[7],cb[8],cb[9],cb[10],cb[11],cb[12],cb[13],cb[14],cb[15]);
      // Project a handful of vertices through the cbuf MVP (both row- and column-
      // major) and count how many land in NDC [-1,1] -- tells us if the world
      // geometry is on-screen (so the black is a PS/sampling issue) or off-screen
      // (a VS/cbuffer-resolution issue).
      if (cb) {
        auto proj = [&](const float *p, bool colMajor, float out[4]) {
          float v[4] = {p[0], p[1], p[2], 1.0f};
          for (int r = 0; r < 4; r++) {
            float s = 0;
            for (int c = 0; c < 4; c++)
              s += (colMajor ? cb[c * 4 + r] : cb[r * 4 + c]) * v[c];
            out[r] = s;
          }
        };
        int onR = 0, onC = 0, onRfz = 0, onCfz = 0, n = d.indexCount < 64 ? d.indexCount : 64;
        const uint16_t *i16 = (d.indexType == 0) ? static_cast<const uint16_t *>(d.indexData) : nullptr;
        const uint32_t *i32 = (d.indexType == 1) ? static_cast<const uint32_t *>(d.indexData) : nullptr;
        float firstR[4] = {0}, firstC[4] = {0};
        auto onscreen = [](float *o) { if (o[3] <= 0.0001f) return false;
          float x = o[0]/o[3], y = o[1]/o[3], z = o[2]/o[3];
          return x>=-1&&x<=1&&y>=-1&&y<=1&&z>=-1&&z<=1; };
        for (int i = 0; i < n; i++) {
          // Use the INDEX buffer to fetch the real vertex (these are indexed draws;
          // a linear 0..n read hits unused verts at the buffer head).
          uint32_t idx = i16 ? i16[i] : i32 ? i32[i] : (uint32_t)i;
          const float *p = reinterpret_cast<const float *>(vb + (size_t)idx * d.vertexStride + d.vattrs[0].offset);
          float r4[4], c4[4]; proj(p, false, r4); proj(p, true, c4);
          // Same but with z negated -- validates the DELTA_GPU_VSFLIPZ hypothesis
          // (does flipping the position z bring the geometry on-screen?).
          float pf[3] = {p[0], p[1], -p[2]}, rf[4], cf[4];
          proj(pf, false, rf); proj(pf, true, cf);
          if (i == 0) { for (int k = 0; k < 4; k++) { firstR[k] = r4[k]; firstC[k] = c4[k]; } }
          if (onscreen(r4)) onR++; if (onscreen(c4)) onC++;
          if (onscreen(rf)) onRfz++; if (onscreen(cf)) onCfz++;
        }
        std::fprintf(stderr, "  proj n=%d onscreen row=%d col=%d | flipZ row=%d col=%d | v0row=[%.2f %.2f %.2f %.2f] v0col=[%.2f %.2f %.2f %.2f]\n",
                     n, onR, onC, onRfz, onCfz, firstR[0],firstR[1],firstR[2],firstR[3], firstC[0],firstC[1],firstC[2],firstC[3]);
        // The game is in real gameplay, so a valid view transform exists. Maybe the
        // VS projects a DIFFERENT attribute than attr0. Project EACH >=3-comp attr
        // (over the indexed verts) and report which, if any, lands on-screen -- that
        // identifies the true position attribute the renderer should be feeding.
        uint32_t firstIdx = i16 ? i16[0] : i32 ? i32[0] : 0;
        for (uint32_t a = 0; a < d.nvattrs && a < 8; a++) {
          if (d.vattrs[a].numComps < 3) continue;
          int onA = 0;
          for (int i = 0; i < n; i++) {
            uint32_t idx = i16 ? i16[i] : i32 ? i32[i] : (uint32_t)i;
            const float *p = reinterpret_cast<const float *>(vb + (size_t)idx * d.vertexStride + d.vattrs[a].offset);
            float c4[4]; proj(p, true, c4);
            if (onscreen(c4)) onA++;
          }
          const float *pv = reinterpret_cast<const float *>(vb + (size_t)firstIdx * d.vertexStride + d.vattrs[a].offset);
          std::fprintf(stderr, "    attr%u off=%u nc=%u dfmt=%u v0=[%.2f %.2f %.2f] onscreen(col)=%d\n",
                       a, d.vattrs[a].offset, d.vattrs[a].numComps, d.vattrs[a].dfmt, pv[0], pv[1], pv[2], onA);
        }
      }
      // Sample the bound texture's ALPHA: is the source genuinely alpha=0 (so the PS
      // must compute opacity elsewhere / a recompiler alpha bug) or alpha=255 (so our
      // load zeroes it)? This decides why the src-alpha blend makes walls invisible.
      if (d.texBase >= 0x1000000000ull && d.texBase < 0x20000000000ull && d.texW && d.texH) {
        const uint32_t *tp = reinterpret_cast<const uint32_t *>(d.texBase);
        uint64_t n = (uint64_t)d.texW * d.texH, step = n > 4096 ? n / 4096 : 1, aNz = 0, rgbNz = 0;
        for (uint64_t i = 0; i < n; i += step) {
          uint32_t px = tp[i];
          if (px >> 24) aNz++;
          if (px & 0x00FFFFFF) rgbNz++;
        }
        std::fprintf(stderr, "  texAlpha: px0=%#010x alphaNonZero=%llu/%llu rgbNonZero=%llu\n",
                     tp[0], (unsigned long long)aNz, (unsigned long long)(n/step),
                     (unsigned long long)rgbNz);
      }
      // Dump the PS's texture-load pattern: SMRD (op/sdst/sbase/imm/off) + MIMG srsrc,
      // and the first 8 user-data dwords, to see how the 4 T#s are loaded.
      auto psI = gcn::decode(reinterpret_cast<const uint32_t *>(d.psAddr), 4096);
      const uint32_t *pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];
      std::fprintf(stderr, "  ps_ud=[%08x %08x %08x %08x %08x %08x %08x %08x]\n",
                   pud[0],pud[1],pud[2],pud[3],pud[4],pud[5],pud[6],pud[7]);
      for (auto &in : psI) {
        if (in.enc == gcn::Enc::smrd) {
          uint32_t w = in.raw[0];
          std::fprintf(stderr, "  smrd op=%u sdst=%u sbase=%u imm=%u off=%#x\n",
                       (w>>22)&0x1F, (w>>15)&0x7F, (w>>9)&0x3F, (w>>8)&1, w&0xFF);
        } else if (in.enc == gcn::Enc::mimg) {
          std::fprintf(stderr, "  mimg srsrc=%u\n", ((in.raw[1]>>16)&0x1F)*4);
        }
      }
      // Dump the WORLD VS's cbuffer reads: for each s_buffer_load, resolve the V#
      // from the VS user-data and print the 16 floats it actually reads (the REAL
      // matrix the VS uses), vs the heuristic cbuf above -- to find whether the VS
      // reads a different cbuffer/offset (the true MVP) than we bind.
      if (d.vsAddr >= 0x1000000000ull && d.vsAddr < 0x20000000000ull) {
        auto vsI = gcn::decode(reinterpret_cast<const uint32_t *>(d.vsAddr), 4096);
        const uint32_t *vud = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
        std::fprintf(stderr, "  vs_ud=[%08x %08x %08x %08x %08x %08x %08x %08x]\n",
                     vud[0],vud[1],vud[2],vud[3],vud[4],vud[5],vud[6],vud[7]);
        int shown = 0;
        for (auto &in : vsI) {
          if (in.enc != gcn::Enc::smrd || shown >= 6) continue;
          uint32_t w = in.raw[0], op = (w>>22)&0x1F, sbase = (w>>9)&0x3F;
          bool imm = (w>>8)&1; uint32_t off = w&0xFF;
          shown++;
          uint32_t b2 = sbase * 2;
          uint32_t boff = imm ? off * 4 : 0;
          // op<8 = s_load (64-bit pointer in ud[b2..b2+1]); op>=8 = s_buffer_load
          // (V# in ud[b2..b2+3], base in low 44 bits). Resolve + dump the 16 floats
          // it reads (a 2nd matrix here would be the missing view transform).
          uint64_t base = 0;
          if (b2 + 1 < 16) {
            if (op < 0x08) base = ((uint64_t)vud[b2+1] << 32 | vud[b2]);
            else base = ((uint64_t)(vud[b2+1] & 0xFFF) << 32 | vud[b2]);
          }
          std::fprintf(stderr, "  vs_smrd op=%u %s sbase=%u(ud%u) off=%#x -> base=%#lx",
                       op, op<0x08?"sload":"sbufload", sbase, b2, boff, (unsigned long)base);
          if (base >= 0x1000000ull && base < 0x20000000000ull) {
            const float *m = reinterpret_cast<const float *>(base + boff);
            std::fprintf(stderr, " mtx=[%.2f %.2f %.2f %.2f / %.2f %.2f %.2f %.2f / %.2f %.2f %.2f %.2f / %.2f %.2f %.2f %.2f]",
                m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15]);
          }
          std::fprintf(stderr, "\n");
        }
      }
    }
    // DELTA_GPU_SKIPSTALE: drop draws that sample a very wide (>=2048) buffer, used
    // to hide a title's stale full-screen video-buffer blit (Doom64's undecoded 4K
    // menu bg = garbage) so the menu items drawn on top become readable -> lets the
    // correct menu input be derived instead of guessed.
    static const bool skipStale = std::getenv("DELTA_GPU_SKIPSTALE") != nullptr;
    if (skipStale && d.texBase && d.texW >= 2048)
      ; // skip the wide stale-buffer blit
    else if (d.vertexData)
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

// Constant Engine RAM: on-chip scratch the CE fills and dumps to memory as the
// shaders' constant buffers. Liverpool CE RAM is 48 KiB. Every access is bounds-
// checked so a malformed packet can never write outside it or outside guest memory.
uint8_t g_ceRam[48 * 1024];
inline bool ccbGuestRange(uint64_t a, uint64_t bytes) {
  return bytes > 0 && a >= 0x1000000000ull && a + bytes <= 0x20000000000ull;
}

void submitCcb(const void *ccb, uint32_t sizeBytes) {
  if (!ccb || sizeBytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  const uint32_t *p = static_cast<const uint32_t *>(ccb);
  uint32_t words = sizeBytes / 4, i = 0;
  static const bool ccbHist = std::getenv("DELTA_GPU_CCBHIST") != nullptr;
  static const bool ceOff = [] { const char *e = std::getenv("DELTA_GPU_CE"); return e && e[0] == '0'; }();
  static uint32_t hist[256] = {};
  static int histDumps = 0;
  static uint64_t nCcb = 0;
  if (ccbHist && nCcb == 0)
    std::fprintf(stderr, "[ccb] first ccb: %u bytes (%u words)\n", sizeBytes, words);
  nCcb++;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = pm4Type(hdr);
    if (type == Pm4Type::type3) {
      uint32_t op = pm4Opcode(hdr), count = pm4Count(hdr);
      const uint32_t *body = &p[i + 1];
      if (i + 1 + count > words) break;
      hist[op & 0xFF]++;
      if (!ceOff) {
        switch (op) {
        case IT_WRITE_CONST_RAM: {  // body[0]=byte offset; body[1..]=data dwords
          uint32_t off = body[0] & 0xFFFF;
          uint32_t n = count > 1 ? count - 1 : 0;
          if ((uint64_t)off + (uint64_t)n * 4 <= sizeof(g_ceRam))
            std::memcpy(g_ceRam + off, &body[1], (size_t)n * 4);
          break;
        }
        case IT_LOAD_CONST_RAM: {  // addrLo, addrHi, num_dwords, byte offset
          if (count >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
            uint32_t num = body[2] & 0x7FFF, off = body[3] & 0xFFFF;
            if (ccbGuestRange(addr, (uint64_t)num * 4) &&
                (uint64_t)off + (uint64_t)num * 4 <= sizeof(g_ceRam))
              std::memcpy(g_ceRam + off, reinterpret_cast<const void *>(addr), (size_t)num * 4);
          }
          break;
        }
        case IT_DUMP_CONST_RAM:
        case IT_DUMP_CONST_RAM_OFFSET: {  // byte offset, num_dwords, addrLo, addrHi
          if (count >= 4) {
            uint32_t off = body[0] & 0xFFFF, num = body[1] & 0x7FFF;
            uint64_t addr = (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) | body[2];
            if (ccbGuestRange(addr, (uint64_t)num * 4) &&
                (uint64_t)off + (uint64_t)num * 4 <= sizeof(g_ceRam))
              std::memcpy(reinterpret_cast<void *>(addr), g_ceRam + off, (size_t)num * 4);
          }
          break;
        }
        default: break;
        }
      }
      i += 1 + count;
    } else if (type == Pm4Type::type2 || hdr == 0) {
      i += 1;
    } else if (type == Pm4Type::type0) {
      i += 1 + pm4Count(hdr);
    } else {
      break;  // type-1 desync
    }
  }
  if (ccbHist && histDumps < 3 && nCcb >= 50) {
    histDumps++;
    std::fprintf(stderr, "[ccb] opcode histogram (after %lu ccbs, this one %u words):\n",
                 (unsigned long)nCcb, words);
    for (int o = 0; o < 256; o++)
      if (hist[o]) std::fprintf(stderr, "[ccb]   op %#04x x%u\n", o, hist[o]);
  }
}

// IT_DISPATCH_DIRECT: a GPU compute dispatch. body = [dim_x, dim_y, dim_z,
// dispatch_initiator] (workgroup counts). The CS program addr, workgroup size,
// resource/RSRC and user-data come from the COMPUTE_* SH registers set before it.
// Doom64 builds its level texture atlases with these (the T# the 3D world samples
// is written by a CS), so without executing them the atlases stay zero/black.
void handleDispatch(const uint32_t *body, uint32_t count) {
  uint32_t dimX = count >= 1 ? body[0] : 0;
  uint32_t dimY = count >= 2 ? body[1] : 0;
  uint32_t dimZ = count >= 3 ? body[2] : 0;
  uint64_t csAddr = (static_cast<uint64_t>(g_regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
                     g_regs[mmCOMPUTE_PGM_LO]) << 8;
  uint32_t tgx = g_regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF;
  uint32_t tgy = g_regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF;
  uint32_t tgz = g_regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF;
  // RSRC2/settings (user_sgpr count, tgid_enable, lds) sits in the ComputeProgram
  // STRUCT at dword 18-19 = SH 0x212/0x213 (compute SET_SH_REG uses struct-relative
  // offsets), i.e. absolute 0x2E12/0x2E13 -- NOT the canonical 0x20F.
  uint32_t rsrc2hi = g_regs[0x2E13];           // high dword of the settings u64
  uint32_t userSgpr = (rsrc2hi >> 1) & 0x1F;   // num_user_regs (bits 37:33)
  uint32_t tgidEnable = (rsrc2hi >> 7) & 0x7;  // tgid_enable (bits 41:39)
  uint32_t ldsDwords = (rsrc2hi >> 15) & 0x1FF;

  static const bool csDump = std::getenv("DELTA_GPU_CSDUMP") != nullptr;
  static int cdN = 0;
  if (csDump && cdN < 6 && csAddr >= 0x1000000000ull && csAddr < 0x20000000000ull) {
    cdN++;
    const uint32_t *ud = &g_regs[mmCOMPUTE_USER_DATA_0];
    std::fprintf(stderr,
        "[cs] addr=%#lx groups=[%u %u %u] tg=[%u %u %u] usgpr=%u tgiden=%u lds=%u\n",
        (unsigned long)csAddr, dimX, dimY, dimZ, tgx, tgy, tgz,
        userSgpr, tgidEnable, ldsDwords);
    std::fprintf(stderr, "[cs]   user_data:");
    for (int k = 0; k < 16; k++) std::fprintf(stderr, " %08x", ud[k]);
    std::fprintf(stderr, "\n");
    gcn::disassemble(reinterpret_cast<const uint32_t *>(csAddr), 1024, "cs");
  }

  // DELTA_GPU_CSRUN: execute the compute shader on the CPU (WIP) so Doom64's
  // buffer->image atlas builders populate the dest textures the 3D world samples.
  static const bool csRun = std::getenv("DELTA_GPU_CSRUN") != nullptr;
  if (csRun && csAddr >= 0x1000000000ull && csAddr < 0x20000000000ull && tgx && tgy) {
    const uint32_t *ud = &g_regs[mmCOMPUTE_USER_DATA_0];
    gcn::runComputeShader(csAddr, dimX, dimY, dimZ, tgx, tgy, tgz, userSgpr,
                          tgidEnable, ud);
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
      g_opHist[op & 0xFF]++;
      if (g_trace && dumpThis)
        std::fprintf(stderr, "[gpu]   @%-5u T3 op=%#04x count=%u\n", i, op, count);
      if (i + 1 + count > words)
        break;  // truncated / desync
      switch (op) {
      case IT_DISPATCH_DIRECT: handleDispatch(body, count); break;
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
      case IT_DMA_DATA:  // CP DMA. body: ctrl, srcLo/Hi, dstLo/Hi, command(byteCount).
        // Actually PERFORM the memory->memory copy (it was a no-op). Doom64 uploads
        // its level texture atlases to GPU memory via CP DMA, so without this the
        // T# addresses stay zero and the 3D world samples blank (black) textures.
        // ctrl word: SRC_SEL[30:29], DST_SEL[21:20]; sel 0/3 = memory address,
        // 2 = immediate data (a fill, not a copy) -- only copy true mem->mem.
        if (count >= 6) {
          uint32_t ctrl = body[0];
          uint32_t srcSel = (ctrl >> 29) & 0x3;
          uint32_t dstSel = (ctrl >> 20) & 0x3;
          uint64_t src = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
          uint64_t dst = (static_cast<uint64_t>(body[4] & 0xFFFF) << 32) | body[3];
          uint32_t bytes = body[5] & 0x1FFFFF;
          bool srcMem = (srcSel == 0 || srcSel == 3);
          bool dstMem = (dstSel == 0 || dstSel == 3);
          static const bool noCopy = std::getenv("DELTA_GPU_NODMACOPY") != nullptr;
          // Only copy between REAL guest memory: the sel bits report "memory" even
          // for GDS/register targets (e.g. dst=0x3022c), which aren't mapped in our
          // address space and segfault. Every real guest allocation (heap/video/
          // garlic) sits far above 16 MiB, so that floor cleanly excludes the
          // on-chip GDS/low targets while keeping texture/buffer uploads.
          auto memOk = [](uint64_t a) { return a >= 0x1000000ull && a < 0x20000000000ull; };
          if (!noCopy && srcMem && dstMem && bytes && bytes <= 0x1000000u &&
              src != dst && memOk(src) && memOk(src + bytes) &&
              memOk(dst) && memOk(dst + bytes))
            std::memcpy(reinterpret_cast<void *>(dst),
                        reinterpret_cast<const void *>(src), bytes);
          if (std::getenv("DELTA_GPU_DMATRACE")) {
            static int dmn = 0;
            if (dmn++ < 200)
              std::fprintf(stderr, "[dma] ctrl=%#x srcSel=%u dstSel=%u src=%#lx dst=%#lx bytes=%u%s\n",
                           ctrl, srcSel, dstSel, (unsigned long)src,
                           (unsigned long)dst, bytes,
                           (srcMem && dstMem) ? " COPIED" : "");
          }
        }
        break;
      case IT_WRITE_DATA: {  // body: control, dstLo, dstHi, data...
        // Gnm writes its 32/64-bit submit/flip fence labels with WRITE_DATA. The
        // control field's dst_sel encoding varies (the flip-label packet built by
        // sceGnmInsertFlip uses control=5, not the [11:8]=memory form), so don't
        // gate on dst_sel: a memory write resolves to a real guest label address,
        // while a register write (dst_sel=0) yields a tiny offset that labelAddrOk
        // rejects. body[1]=dstLo, body[2]=dstHi, body[3..]=data.
        if (count >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          uint32_t ndw = count - 3;
          if (labelAddrOk(addr) && labelAddrOk(addr + (uint64_t)ndw * 4))
            std::memcpy(reinterpret_cast<void *>(addr), &body[3],
                        (size_t)ndw * 4);
          if (g_eopTrace)
            std::fprintf(stderr, "[eop] WRITE_DATA dst=%#lx ndw=%u v0=%#x\n",
                         (unsigned long)addr, ndw, ndw ? body[3] : 0);
        }
        break;
      }
      case IT_EVENT_WRITE_EOP: {  // body: eventCtrl, addrLo, addrHi+sel, dataLo, dataHi
        if (count >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          uint32_t dataSel = (body[2] >> 29) & 0x7;  // 1=32b, 2=64b, 3/4=clock
          uint64_t val = static_cast<uint64_t>(body[3]) |
                         (static_cast<uint64_t>(count >= 5 ? body[4] : 0) << 32);
          if (dataSel == 1) writeLabel(addr, val, false);
          else if (dataSel == 2) writeLabel(addr, val, true);
          else if (dataSel >= 3) writeLabel(addr, gpuClockTs(), true);
          if (g_eopTrace)
            std::fprintf(stderr, "[eop] EOP addr=%#lx sel=%u val=%#lx\n",
                         (unsigned long)addr, dataSel, (unsigned long)val);
        }
        break;
      }
      case IT_RELEASE_MEM: {  // body: eventCtrl, selBits, addrLo, addrHi, dataLo, dataHi
        if (count >= 5) {
          uint32_t dataSel = (body[1] >> 29) & 0x7;  // 1=32b, 2=64b, 3/4=clock
          uint64_t addr = (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) |
                          (body[2] & ~0x3u);
          uint64_t val = static_cast<uint64_t>(body[4]) |
                         (static_cast<uint64_t>(count >= 6 ? body[5] : 0) << 32);
          if (dataSel == 1) writeLabel(addr, val, false);
          else if (dataSel == 2) writeLabel(addr, val, true);
          else if (dataSel >= 3) writeLabel(addr, gpuClockTs(), true);
          if (g_eopTrace)
            std::fprintf(stderr, "[eop] RELEASE_MEM addr=%#lx sel=%u val=%#lx\n",
                         (unsigned long)addr, dataSel, (unsigned long)val);
        }
        break;
      }
      case IT_EVENT_WRITE_EOS: {  // body: eventCtrl, addrLo, addrHi+cmd, data
        if (count >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          writeLabel(addr, body[3], false);
          if (g_eopTrace)
            std::fprintf(stderr, "[eop] EOS addr=%#lx val=%#x\n",
                         (unsigned long)addr, body[3]);
        }
        break;
      }
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
    } else if (type == Pm4Type::type0) {
      // Type-0 writes a run of consecutive registers (base in hdr[15:0], count in
      // hdr[29:16]+1) directly into the register file. The old walker treated this as
      // a desync and STOPPED -- dropping every later draw (the room floor) in any
      // command buffer that used type-0. Apply the register writes and skip the body.
      uint32_t cnt = pm4Count(hdr);            // body dword count
      uint32_t base = pm4Type0Reg(hdr);        // absolute register dword offset
      for (uint32_t k = 0; k < cnt && i + 1 + k < words; k++) {
        uint32_t idx = base + k;
        if (idx < kRegFileSize) g_regs[idx] = p[i + 1 + k];
      }
      i += 1 + cnt;
    } else {
      // A type-1 header is a genuine desync; stop.
      static const bool desyncTrace = std::getenv("DELTA_GPU_DESYNC") != nullptr;
      if (dumpThis || desyncTrace)
        std::fprintf(stderr, "[gpu]   @%-5u/%u STOP type%u hdr=%#x\n", i, words,
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
  // Cumulative opcode histogram dumped once deep into gameplay (DELTA_GPU_OPHIST):
  // reveals any draw/dispatch opcode the title uses that isDraw() doesn't handle (a
  // silently-skipped draw -- e.g. the non-tutorial room floor).
  static const bool opHist = std::getenv("DELTA_GPU_OPHIST") != nullptr;
  static bool opHistDumped = false;
  // Time-gate (default 100s) so the cumulative histogram includes the in-level
  // command stream (level-load compute/copies), not just the title.
  static const auto ohStart = std::chrono::steady_clock::now();
  static const int ohAfter = [] { const char *e = std::getenv("DELTA_GPU_OPHIST_AFTER");
    return e ? std::atoi(e) : 100; }();
  auto ohElapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - ohStart).count();
  if (opHist && !opHistDumped && ohElapsed >= ohAfter) {
    opHistDumped = true;
    dumpHist();
  }
}

}  // namespace gpu
