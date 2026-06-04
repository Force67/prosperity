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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace gpu {
namespace {

std::mutex g_mtx;
Regs g_regs;  // persistent register state across submits (Gnm relies on this)
const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;
std::atomic<uint64_t> g_totalSubmits{0};
std::atomic<uint64_t> g_totalDraws{0};
bool g_vkTried = false;
bool g_frameActive = false;

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
    d.indexCount = (count >= 1) ? body[0] : 0;
    d.rtBase = g_regs.cbColorBase(0);
    d.rtW = fbWidth();
    d.rtH = fbHeight();

    // Per-draw blend state from CB_BLEND0_CONTROL. Bit 30 is the per-target blend
    // enable; when clear the draw writes opaquely (no blend). Isaac's room
    // darkness/vignette and additive effect overlays rely on this: rendered with
    // a single hardcoded blend they came out opaque and blacked out the scene.
    d.blendControl = g_regs[mmCB_BLEND0_CONTROL];
    d.blendEnable = (d.blendControl >> 30) & 1u;

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
        // Isaac's vertex format: pos.xy@0, color.rgb@0x10, uv.xy@0x1c.
        if (d.vertexStride >= 0x1c)
          d.colorOffset = 0x10;
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
        d.uvOffset = 0x1c;  // float[7],[8]
      }
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
    if (!g_frameActive) {
      vk::beginFrame();
      g_frameActive = true;
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
