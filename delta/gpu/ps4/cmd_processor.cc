/*
 * PS4Delta : PS4 emulation and research project
 *
 * GPU command processor. See cmd_processor.h.
 */

#include "gpu/ps4/cmd_processor.h"
#include <chrono>
#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/ps4/pm4.h"
#include "gpu/rhi/renderer.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <utl/mem.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(uint64_t, kBlitRt, "DELTA_GPU_BLIT_RT", 0);
DELTA_OPTION(bool, kCeOn, "DELTA_GPU_CE", true);
DELTA_OPTION(int, kDlAfter, "DELTA_GPU_DRAWLIST_AFTER", 90);
DELTA_OPTION(uint32_t, kGeomMin, "DELTA_GPU_GEOMMIN", 500);
DELTA_OPTION(int, kMtAfter, "DELTA_GPU_MASKTRACE_AFTER", 0);
DELTA_OPTION(int, kMtMax, "DELTA_GPU_MASKTRACE_MAX", 200);
DELTA_OPTION(uint64_t, kMtRt, "DELTA_GPU_MASKTRACE_RT", 0);
DELTA_OPTION(int, kOhAfter, "DELTA_GPU_OPHIST_AFTER", 100);
DELTA_OPTION(bool, kRecompOn, "DELTA_GPU_RECOMP", true);
DELTA_OPTION(const char *, kSkipShList, "DELTA_GPU_SKIPSH", nullptr);
DELTA_OPTION(int, kTexTrackFrame, "DELTA_GPU_TEXTRACK_FRAME", -1);
DELTA_OPTION(bool, kBlitDump, "DELTA_GPU_BLITDUMP", false);
DELTA_OPTION(bool, kCcbHist, "DELTA_GPU_CCBHIST", false);
DELTA_OPTION(bool, kCsDump, "DELTA_GPU_CSDUMP", false);
DELTA_OPTION(bool, kCsResTrace, "DELTA_GPU_CSRES", false);
DELTA_OPTION(bool, kDbTrace, "DELTA_GPU_DBTRACE", false);
DELTA_OPTION(bool, kDesyncTrace, "DELTA_GPU_DESYNC", false);
DELTA_OPTION(bool, kDrawList, "DELTA_GPU_DRAWLIST", false);
DELTA_OPTION(bool, kEopTrace, "DELTA_GPU_EOPTRACE", false);
DELTA_OPTION(bool, kEudFail, "DELTA_GPU_EUDFAIL", false);
DELTA_OPTION(bool, kForceDepth, "DELTA_GPU_FORCEDEPTH", false);
DELTA_OPTION(bool, kGeomDump, "DELTA_GPU_GEOMDUMP", false);
DELTA_OPTION(bool, kGpuDmatrace, "DELTA_GPU_DMATRACE", false);
DELTA_OPTION(bool, kGpuDrawpkt, "DELTA_GPU_DRAWPKT", false);
DELTA_OPTION(bool, kMaskTrace, "DELTA_GPU_MASKTRACE", false);
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
DELTA_OPTION(bool, kNoCs, "DELTA_GPU_NOCS", false);
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kOpHist, "DELTA_GPU_OPHIST", false);
DELTA_OPTION(bool, kRawBufTrace, "DELTA_GPU_RAWBUF", false);
DELTA_OPTION(bool, kSkipStale, "DELTA_GPU_SKIPSTALE", false);
DELTA_OPTION(bool, kSpriteDis, "DELTA_GPU_SPRITEDIS", false);
DELTA_OPTION(bool, kSpriteDump, "DELTA_GPU_SPRITEDUMP", false);
DELTA_OPTION(bool, kTexfmt, "DELTA_GPU_TEXFMT", false);
DELTA_OPTION(bool, kTrace, "DELTA_GPU_TRACE", false);
DELTA_OPTION(bool, kVattrDump, "DELTA_GPU_VATTRDUMP", false);
}  // namespace

namespace gpu {
namespace {

std::mutex g_mtx;
Regs g_regs;  // persistent register state across submits (Gnm relies on this)
std::atomic<uint64_t> g_total_submits{0};
std::atomic<uint64_t> g_total_draws{0};
bool g_vk_tried = false;
bool g_frame_active = false;
uint32_t g_presented_frames = 0;

struct ShaderKey {
  uint64_t vs = 0, ps = 0, fetch = 0;
  uint32_t ps_input_ena = 0;
  // Which PS samplers read a volume image. Unlike the 2D-array case, whose DA
  // bit lives in the instruction, a 3D descriptor is indistinguishable in the
  // code: the same PS sampled with a 2D T# must translate to a different
  // module, so the dimensionality belongs in the key.
  uint32_t tex_3d_mask = 0;
  bool neo = false;
  bool operator==(const ShaderKey& o) const {
    return vs == o.vs && ps == o.ps && fetch == o.fetch &&
           ps_input_ena == o.ps_input_ena && tex_3d_mask == o.tex_3d_mask &&
           neo == o.neo;
  }
};
struct ShaderKeyHash {
  size_t operator()(const ShaderKey& k) const {
    uint64_t h =
        (k.vs ^ (k.ps + 0x9e3779b97f4a7c15ull + (k.vs << 6) + (k.vs >> 2)));
    h ^= k.fetch + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= k.ps_input_ena + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= k.tex_3d_mask + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(k.neo) << 63;
    return static_cast<size_t>(h);
  }
};

struct ComputeShaderKey {
  uint64_t address = 0;
  uint32_t thread_x = 0, thread_y = 0, thread_z = 0;
  uint32_t user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
  bool neo = false;

  bool operator==(const ComputeShaderKey& other) const = default;
};

struct ComputeShaderKeyHash {
  size_t operator()(const ComputeShaderKey& key) const {
    size_t hash = std::hash<uint64_t>{}(key.address);
    const auto combine = [&](uint32_t value) {
      hash ^= std::hash<uint32_t>{}(value) + 0x9e3779b9u + (hash << 6) +
              (hash >> 2);
    };
    combine(key.thread_x);
    combine(key.thread_y);
    combine(key.thread_z);
    combine(key.user_sgpr);
    combine(key.tgid_enable);
    combine(key.lds_dwords);
    combine(key.neo ? 1 : 0);
    return hash;
  }
};

// Index-buffer state, set by IT_INDEX_TYPE / IT_INDEX_BASE before a draw.
uint32_t g_index_type =
    0;  // 0 = 16-bit, 1 = 32-bit (VGT_DMA_INDEX_TYPE bits[1:0])
uint64_t g_index_base =
    0;  // from IT_INDEX_BASE (DRAW_INDEX_2 carries its own base)
// From IT_SET_BASE with base_index 1: where DRAW_INDIRECT /
// DRAW_INDEX_INDIRECT read their argument structs from.
uint64_t g_indirect_base = 0;
uint32_t g_num_instances =
    1;  // from IT_NUM_INSTANCES; applies to the next draw(s)

// Current render-target / framebuffer geometry, derived from the screen scissor
// (CB regs don't carry an explicit width/height).
uint32_t FbWidth() {
  uint32_t br = g_regs[mmPA_SC_SCREEN_SCISSOR_BR];
  uint32_t w = br & 0xFFFF;
  return w ? w : 1920;
}
uint32_t FbHeight() {
  uint32_t br = g_regs[mmPA_SC_SCREEN_SCISSOR_BR];
  uint32_t h = br >> 16;
  return h ? h : 1080;
}

// How many times the guest has programmed CB_SHADER_MASK / CB_TARGET_MASK.
// Diagnostic only (reported by DELTA_GPU_MASKTRACE): the register file is
// zero-initialised, so a value of 0 is ambiguous between "the driver cleared
// it" and "never written" without a write count.
uint32_t g_shader_mask_writes = 0;
uint32_t g_target_mask_writes = 0;

// Write a run of register values from a SET_*_REG packet body into the file.
void SetRegs(uint32_t base, const uint32_t* body, uint32_t count) {
  // Indexed SET packets use bits 28..31 for the index; only the low 16 bits are
  // the register offset. Treating the complete word as an offset drops Neo
  // register writes such as 0x40000258.
  uint32_t off = Pm4SetRegAddress(base, body[0]);
  for (uint32_t i = 1; i < count; i++) {
    uint32_t idx = off + (i - 1);
    if (idx < kRegFileSize)
      g_regs[idx] = body[i];
    if (idx == mmCB_SHADER_MASK)
      g_shader_mask_writes++;
    else if (idx == mmCB_TARGET_MASK)
      g_target_mask_writes++;
  }
}

bool IsDraw(uint32_t op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDIRECT ||
         op == IT_DRAW_INDEX_INDIRECT || op == IT_DRAW_INDEX_MULTI_AUTO;
}

// GPU completion-label writes (EOP / RELEASE_MEM / WRITE_DATA). Our submit is
// synchronous: every draw in the dcb is finished by the time we walk past these
// packets, so the fence/label the GPU would signal is complete the instant we
// process it. Writing it immediately is what lets the guest's CPU-side polls --
// the flip-done / submit-done labels Gnm spins on between frames -- make
// progress. Without it the title stalls after the few in-flight display buffers
// drain (it never sees a flip "complete").
// Labels live in guest memory the game allocated (Garlic/Onion
// 0x10_0000_0000+), in low guest heaps (0x2_0000_0000+), or in the GnmDriver
// area (0xfe00_0000+). Accept any plausibly-mapped, non-low address; reject
// null/garbage.
inline bool LabelAddrOk(uint64_t a) {
  return a >= 0x10000ull && a < 0x20000000000ull;
}
void WriteLabel(uint64_t addr, uint64_t value, bool is64) {
  if (!LabelAddrOk(addr))
    return;
  if (is64)
    *reinterpret_cast<volatile uint64_t*>(addr) = value;
  else
    *reinterpret_cast<volatile uint32_t*>(addr) = static_cast<uint32_t>(value);
}

// EOP/RELEASE_MEM DATA_SEL 3 (GPU clock) and 4 (system clock) tell the GPU to
// write its current 64-bit clock counter into the label, NOT the packet's
// immediate data (which is 0 for these). A title that polls such a label for
// "non-zero == the GPU reached this point" needs a real,
// monotonically-increasing, non-zero value. Our submit is synchronous, so any
// advancing clock reads as "already complete". Without this Doom64's per-frame
// submit-done wait (a spin on this label with a hard 2s timeout) burns the full
// 2s every frame -> ~0.5 fps.
uint64_t GpuClockTs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count());
}

// Opcode histogram (DELTA_GPU_TRACE): shows what the dcb
// contains and whether the walker reaches a draw or desyncs.
uint32_t g_op_hist[256] = {};
int g_dcb_seen = 0;
void DumpHist() {
  std::fprintf(stderr, "[gpu] dcb opcode histogram (after %d dcbs):\n",
               g_dcb_seen);
  for (int i = 0; i < 256; i++)
    if (g_op_hist[i])
      std::fprintf(stderr, "[gpu]   op %#04x x%u\n", i, g_op_hist[i]);
}

// Issue the current register state as a draw: begin the frame lazily on the
// first draw, then hand the draw to the Vulkan renderer.
void HandleDraw(uint32_t op, const uint32_t* body, uint32_t count) {
  uint64_t vs_a = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS);
  uint64_t ps_a = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  // One-time: find the first PS that samples a texture (has an MIMG
  // instruction) and dump how it loads its resources, so we can wire texture
  // sampling.
  static bool g_tex_probed = false;
  if (kTrace && !g_tex_probed && ps_a >= 0x1000000000ull &&
      ps_a < 0x20000000000ull) {
    auto pi = gcn::Decode(reinterpret_cast<const uint32_t*>(ps_a), 256);
    int n_mimg = 0, n_smrd = 0;
    for (auto& in : pi) {
      if (in.enc == gcn::Enc::kMimg)
        n_mimg++;
      if (in.enc == gcn::Enc::kSmrd)
        n_smrd++;
    }
    if (n_mimg > 0) {
      g_tex_probed = true;
      std::fprintf(stderr, "[gpu] TEXTURED PS @%#lx: mimg=%d smrd=%d\n",
                   (unsigned long)ps_a, n_mimg, n_smrd);
      const uint32_t* pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];
      std::fprintf(stderr, "[gpu]   PS user_data:");
      for (int k = 0; k < 16; k++)
        std::fprintf(stderr, " %08x", pud[k]);
      std::fprintf(stderr, "\n");
      auto texs = gcn::TrackTextures(gcn::CachedProgram(ps_a, 4096), pud, false,
                                     ps_a);
      std::fprintf(stderr, "[gpu]   TrackTextures -> %zu\n", texs.size());
      if (!texs.empty() && texs[0].valid) {
        auto& t = texs[0];
        // Dump the texture as a LINEAR interpretation (to inspect tiling).
        auto* px = reinterpret_cast<const uint8_t*>(t.base);
        FILE* f = std::fopen("/tmp/tex_raw.bin", "wb");
        if (f) {
          std::fwrite(px, 1, (size_t)t.width * t.height * 4, f);
          std::fclose(f);
          std::fprintf(stderr, "[gpu]   dumped /tmp/tex_raw.bin (%ux%u rgba)\n",
                       t.width, t.height);
        }
      }
    }
  }
  if (rhi::DefaultRenderer().available()) {
    rhi::DrawInfo d;
    const uint32_t* vud = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
    std::memcpy(d.vs_user_data, vud, 16 * sizeof(uint32_t));
    std::memcpy(d.ps_user_data, &g_regs[mmSPI_SHADER_USER_DATA_PS_0],
                16 * sizeof(uint32_t));
    d.prim_type = g_regs[mmVGT_PRIMITIVE_TYPE];
    d.instance_count = g_num_instances;
    uint32_t auto_vertex_count =
        op == IT_DRAW_INDEX_AUTO && count >= 1 ? body[0] : 0;
    // DELTA_GPU_DRAWPKT: which draw opcode each draw actually arrives as, with
    // its raw body. A count that never leaves the packet (an opcode we do not
    // decode, or an indirect draw whose count lives in guest memory) shows up
    // downstream only as "too few vertices", which points at the wrong layer.
    {
      if (kGpuDrawpkt) {
        // A histogram over the WHOLE run, not the first N draws: the counts
        // that decline downstream are not the ones a first-N sample catches.
        struct Bucket {
          uint32_t op, prim, auto_count, n;
        };
        static Bucket buckets[32];
        static uint32_t nbuckets = 0, total = 0;
        const uint32_t prim = g_regs[mmVGT_PRIMITIVE_TYPE];
        uint32_t i = 0;
        for (; i < nbuckets; i++)
          if (buckets[i].op == op && buckets[i].prim == prim &&
              buckets[i].auto_count == auto_vertex_count)
            break;
        if (i == nbuckets && nbuckets < 32)
          buckets[nbuckets++] = {op, prim, auto_vertex_count, 0};
        if (i < nbuckets)
          buckets[i].n++;
        if (++total % 2000 == 0) {
          std::fprintf(stderr, "[drawpkt] %u draws:", total);
          for (uint32_t j = 0; j < nbuckets; j++)
            std::fprintf(stderr, " op=%#x/prim=%#x/auto=%u x%u", buckets[j].op,
                         buckets[j].prim, buckets[j].auto_count, buckets[j].n);
          std::fprintf(stderr, "\n");
        }
      }
    }
    // Index buffer. DRAW_INDEX_2 (5 dwords): maxSize, baseLo, baseHi,
    // index_count, drawInitiator. The draws are indexed triangle lists; without
    // the index buffer (drawing raw vertices as a strip) batched sprites smear
    // into long diagonal triangles. DRAW_INDEX_AUTO has no index buffer
    // (sequential verts).
    if (op == IT_DRAW_INDEX_2 && count >= 4) {
      uint64_t ibase = (static_cast<uint64_t>(body[2] & 0xFF) << 32) | body[1];
      uint32_t icount = body[3];
      const bool accepted = ibase >= 0x1000000000ull &&
                            ibase < 0x20000000000ull && icount &&
                            icount <= 0x100000;
      // DELTA_GPU_DRAWPKT: the raw draw packet, before any of the acceptance
      // rules below can silently drop it. A rejected index buffer leaves the
      // draw non-indexed, which then declines for too few vertices -- the two
      // symptoms look unrelated in the logs unless the packet is visible.
      if (kGpuDrawpkt) {
        static int n = 0;
        if (n++ < 128)
          std::fprintf(stderr,
                       "[drawpkt] DRAW_INDEX_2 ibase=%#lx icount=%u itype=%u "
                       "%s%s\n",
                       (unsigned long)ibase, icount, g_index_type,
                       accepted ? "accepted" : "REJECTED",
                       accepted            ? ""
                       : icount > 0x100000 ? " (count cap)"
                       : !icount           ? " (zero count)"
                                           : " (base out of range)");
      }
      if (accepted) {
        d.index_data = reinterpret_cast<const void*>(ibase);
        d.index_count = icount;
        d.index_type = g_index_type;  // 0 = 16-bit, 1 = 32-bit
      }
    }
    // DRAW_INDIRECT / DRAW_INDEX_INDIRECT take their counts from a struct in
    // guest memory at IT_SET_BASE(1) + dataOffset, not from the packet, so the
    // counts never appeared in the command stream at all and the draws fell
    // back to a vertex count of 1 and declined. The structs are
    //   indirect:       vertexCount, instanceCount, startVertex, startInstance
    //   index-indirect: indexCount,  instanceCount, startIndex, baseVertex, ...
    // (Liverpool follows the same layout as the radeon docs' DRAW_*_INDIRECT.)
    if ((op == IT_DRAW_INDIRECT || op == IT_DRAW_INDEX_INDIRECT) &&
        count >= 1 && g_indirect_base) {
      const uint64_t args = g_indirect_base + body[0];
      const uint32_t need = op == IT_DRAW_INDIRECT ? 16u : 20u;
      // The args are usually produced ON the GPU (a compute pass writes the
      // counts a later draw consumes). Our compute working set keeps its
      // results GPU-resident and writes them back lazily, so reading guest
      // memory here without flushing that range first yields the stale zeros
      // the buffer was allocated with.
      rhi::FlushCsWritesRange(rhi::DefaultRenderer(), args, need);
      const bool mapped =
          utl::isMemoryRangeMapped(reinterpret_cast<const void*>(args), need);
      uint32_t a[5] = {0, 0, 0, 0, 0};
      if (mapped)
        std::memcpy(a, reinterpret_cast<const void*>(args), need);
      if (kGpuDrawpkt) {
        static int n = 0;
        if (n++ < 24)
          std::fprintf(stderr,
                       "[drawpkt] %s args=%#lx (base=%#lx +%#x) %s "
                       "count=%u inst=%u start=%u\n",
                       op == IT_DRAW_INDIRECT ? "INDIRECT" : "INDEX_INDIRECT",
                       (unsigned long)args, (unsigned long)g_indirect_base,
                       body[0], mapped ? "mapped" : "UNMAPPED", a[0], a[1],
                       a[2]);
      }
      if (mapped && a[0] && a[0] <= 0x100000) {
        if (op == IT_DRAW_INDIRECT) {
          d.vertex_count = a[0];
        } else if (g_index_base) {
          const uint32_t index_bytes = g_index_type == 1 ? 4 : 2;
          const uint64_t ibase =
              g_index_base + static_cast<uint64_t>(a[2]) * index_bytes;
          if (utl::isMemoryRangeMapped(
                  reinterpret_cast<const void*>(ibase),
                  static_cast<uint64_t>(a[0]) * index_bytes)) {
            d.index_data = reinterpret_cast<const void*>(ibase);
            d.index_count = a[0];
            d.index_type = g_index_type;
          }
        }
        if (a[1])
          d.instance_count = a[1];
      }
    }
    // DRAW_INDEX_OFFSET_2 (5 dwords: maxSize, indexOffset, indexCount,
    // drawInitiator) draws from the index buffer IT_INDEX_BASE already set,
    // starting indexOffset entries in -- unlike DRAW_INDEX_2, which carries its
    // own base. Shadow of the Colossus submits ~47% of its draws this way, and
    // leaving the packet undecoded left them with no index buffer at all: they
    // fell back to the vertex count, which a hand-fetch V# reports as 1, and
    // then declined for having fewer than 3 vertices.
    if (op == IT_DRAW_INDEX_OFFSET_2 && count >= 3) {
      const uint32_t index_offset = body[1], icount = body[2];
      const uint32_t index_bytes = g_index_type == 1 ? 4 : 2;
      const uint64_t ibase =
          g_index_base + static_cast<uint64_t>(index_offset) * index_bytes;
      const bool mapped =
          g_index_base && icount && icount <= 0x100000 &&
          utl::isMemoryRangeMapped(reinterpret_cast<const void*>(ibase),
                                   static_cast<uint64_t>(icount) * index_bytes);
      if (kGpuDrawpkt) {
        static int n = 0;
        if (n++ < 32)
          std::fprintf(stderr,
                       "[drawpkt] OFFSET_2 maxsz=%u ioff=%u icount=%u "
                       "g_index_base=%#lx -> ibase=%#lx itype=%u %s\n",
                       body[0], index_offset, icount,
                       (unsigned long)g_index_base, (unsigned long)ibase,
                       g_index_type,
                       mapped              ? "accepted"
                       : !g_index_base     ? "REJECTED (no INDEX_BASE set)"
                       : !icount           ? "REJECTED (zero count)"
                       : icount > 0x100000 ? "REJECTED (count cap)"
                                           : "REJECTED (unmapped)");
      }
      if (mapped) {
        d.index_data = reinterpret_cast<const void*>(ibase);
        d.index_count = icount;
        d.index_type = g_index_type;
      }
    }
    d.rt_w = FbWidth();
    d.rt_h = FbHeight();
    // A stale CB_COLORn_BASE remains programmed during depth-only passes. Bind
    // a color attachment only when both its write mask and CB_COLORn_INFO
    // format are valid.
    {
      uint32_t tmask = g_regs[mmCB_TARGET_MASK];
      for (int rt = 0; rt < 8; rt++) {
        uint64_t base = g_regs.CbColorBase(rt);
        uint32_t info = g_regs[mmCB_COLOR0_INFO + rt * kCbColorStride];
        if (((tmask >> (rt * 4)) & 0xF) && ((info >> 2) & 0x1F) &&
            base >= 0x1000000000ull && base < 0x20000000000ull) {
          d.mrt_base[rt] = base;
          d.mrt_info[rt] = info;
          d.mrt_count = rt + 1;
        }
      }
      d.rt_base = d.mrt_count ? d.mrt_base[0] : 0;
    }

    // Per-draw blend state from CB_BLEND0_CONTROL. Bit 30 is the per-target
    // blend enable; when clear the draw writes opaquely (no blend). Isaac's
    // room darkness/vignette and additive effect overlays rely on this:
    // rendered with a single hardcoded blend they came out opaque and blacked
    // out the scene.
    d.blend_control = g_regs[mmCB_BLEND0_CONTROL];
    d.blend_enable = (d.blend_control >> 30) & 1u;
    // Per-MRT blend: each target's own CB_BLENDn_CONTROL (1 dword apart), so an
    // MRT draw blends each attachment as the guest programmed it instead of
    // applying target 0's blend to all. Target 0 mirrors
    // blend_control/blend_enable (single-RT unchanged).
    d.mrt_blend[0] = d.blend_control;
    if (d.blend_enable)
      d.mrt_blend_mask |= 1u;
    for (uint32_t rt = 1; rt < 8; rt++) {
      uint32_t bc = g_regs[mmCB_BLEND0_CONTROL + rt * kCbBlendStride];
      d.mrt_blend[rt] = bc;
      if ((bc >> 30) & 1u)
        d.mrt_blend_mask |= (1u << rt);
    }
    // Per-MRT channel write mask (MRT0 = bits[3:0]) and overall colour-control
    // mode.
    d.target_mask = g_regs[mmCB_TARGET_MASK];
    d.shader_mask = g_regs[mmCB_SHADER_MASK];
    // GNM fast clear: RECT_LIST (VGT prim 17), no pixel shader, no vertex
    // attributes, at least one colour target bound. The clear colour is in
    // CB_COLORn_CLEAR_WORD0/1, encoded in each target's own format. A
    // depth-only pass looks similar but binds no colour target, so mrt_count
    // keeps the two apart.
    d.is_clear_rect =
        d.prim_type == 17 && !ps_a && !d.num_vattrs && d.mrt_count != 0;
    if (d.is_clear_rect) {
      for (uint32_t rt = 0; rt < 8; rt++) {
        d.mrt_clear_word[rt][0] =
            g_regs[mmCB_COLOR0_CLEAR_WORD0 + rt * kCbColorStride];
        d.mrt_clear_word[rt][1] =
            g_regs[mmCB_COLOR0_CLEAR_WORD1 + rt * kCbColorStride];
      }
    }
    d.color_control = g_regs[mmCB_COLOR_CONTROL];

    // Depth/stencil state. A 3D title (Doom64, SOTTR) binds a Z buffer and
    // Z-tests the world; 2D titles leave DB_Z_INFO format invalid so
    // depth_valid stays false and no depth attachment is bound (the 2D path is
    // unchanged). We only render to the write base; internal format is always
    // D32F (never read back to the guest).
    {
      uint32_t dc = g_regs[mmDB_DEPTH_CONTROL];
      uint32_t zinfo = kNoDepth ? 0 : g_regs[mmDB_Z_INFO];
      uint64_t zread = static_cast<uint64_t>(g_regs[mmDB_Z_READ_BASE]) << 8;
      uint64_t zbase = static_cast<uint64_t>(g_regs[mmDB_Z_WRITE_BASE]) << 8;
      static int db_n = 0;
      if (kDbTrace && db_n < 24 && (zinfo || dc)) {
        db_n++;
        std::fprintf(stderr,
                     "[db] DEPTH_CONTROL=%#x Z_INFO=%#x Zread=%#lx Zwrite=%#lx "
                     "clear=%#x prim=%u\n",
                     dc, zinfo, (unsigned long)zread, (unsigned long)zbase,
                     g_regs[mmDB_DEPTH_CLEAR], g_regs[mmVGT_PRIMITIVE_TYPE]);
      }
      d.depth_valid = (zinfo & 0x3) != 0;
      if (d.depth_valid && zbase >= 0x1000000000ull &&
          zbase < 0x20000000000ull && ((dc >> 1) & 1u || (dc >> 2) & 1u)) {
        d.depth_base = zbase;
        d.depth_test_enable = (dc >> 1) & 1u;
        d.depth_write_enable = (dc >> 2) & 1u;
        d.depth_func = (dc >> 4) & 0x7;
        std::memcpy(&d.depth_clear, &g_regs[mmDB_DEPTH_CLEAR], 4);
        if (!(d.depth_clear >= 0.0f && d.depth_clear <= 1.0f))
          d.depth_clear = 1.0f;
      } else {
        d.depth_valid = false;
      }
      // DELTA_GPU_FORCEDEPTH: exercise the depth attachment/clear/pipeline path
      // even on titles that bind no Z buffer (Doom64 is painter-ordered,
      // Z_INFO=0), to validate it end-to-end. func=ALWAYS so no fragment is
      // hidden (visible output unchanged); depth_base keys a synthetic depth
      // image off the RT.
      if (kForceDepth && !d.depth_base && d.rt_base) {
        d.depth_base = d.rt_base;
        d.depth_test_enable = true;
        d.depth_write_enable = true;
        d.depth_func = 7;  // ALWAYS
        d.depth_clear = 1.0f;
      }
    }
    // Primitive-setup: face culling + winding (PA_SU_SC_MODE_CNTL).
    {
      uint32_t sc = g_regs[mmPA_SU_SC_MODE_CNTL];
      d.cull_mode = sc & 0x3;  // CULL_FRONT[0] | CULL_BACK[1]
      d.front_ccw = ((sc >> 2) & 1u) == 0;
    }
    std::memcpy(&d.viewport_x_scale, &g_regs[mmPA_CL_VPORT_XSCALE], 4);
    std::memcpy(&d.viewport_x_offset, &g_regs[mmPA_CL_VPORT_XOFFSET], 4);
    std::memcpy(&d.viewport_y_scale, &g_regs[mmPA_CL_VPORT_YSCALE], 4);
    std::memcpy(&d.viewport_y_offset, &g_regs[mmPA_CL_VPORT_YOFFSET], 4);

    // Constant buffer (transform): default to the sgpr[4..7] V# (the common VS
    // cbuffer slot); the recompiled-shader path below re-resolves it from the
    // SGPR the VS actually reads (rc.vs_cbufs) when that differs.
    uint64_t cbuf = (static_cast<uint64_t>(vud[5] & 0xFFFF) << 32) | vud[4];
    if (cbuf >= 0x1000000000ull && cbuf < 0x20000000000ull) {
      std::memcpy(d.mvp, reinterpret_cast<const void*>(cbuf), 64);
      d.cbuf_base = cbuf;
    }

    // Vertex buffer: resource-track the fetch shader (VS sgpr[0..1] ptr).
    uint64_t fetch = (static_cast<uint64_t>(vud[1] & 0xFFFF) << 32) | vud[0];
    if (fetch >= 0x1000000000ull && fetch < 0x20000000000ull) {
      auto vbs = gcn::TrackVertexBuffers(*gcn::CachedProgram(fetch, 64), vud);
      if (!vbs.empty()) {
        d.vertex_data = reinterpret_cast<const void*>(vbs[0].base);
        d.vertex_count = vbs[0].num_records;
        d.vertex_stride = vbs[0].stride;
        d.pos_offset = 0;
        // Per-attribute offsets from the fetch shader's V# bases: each
        // attribute's V# points at vertexBase + attributeOffset, so off =
        // vb.base - pos.base. dfmt 11 = 32_32 (float2 uv), dfmt 10 = 8_8_8_8
        // (rgba8 colour). This generalises the old hardcoded sprite offsets
        // (pos@0,color@0x10,uv@0x1c) so other vertex layouts (e.g. the
        // stride-36 room floor) sample the right attributes instead of garbage.
        // Falls back to the sprite offsets.
        uint32_t uv_off = 0, col_off = 0;
        for (size_t i = 1; i < vbs.size(); i++) {
          uint64_t off = vbs[i].base - vbs[0].base;
          if (off == 0 || off >= d.vertex_stride)
            continue;
          if (vbs[i].dfmt == 11 && !uv_off)
            uv_off = static_cast<uint32_t>(off);
          else if (vbs[i].dfmt == 10 && !col_off)
            col_off = static_cast<uint32_t>(off);
        }
        d.uv_offset = uv_off ? uv_off : (d.vertex_stride >= 0x1c ? 0x1c : 0);
        d.color_offset =
            col_off ? col_off : (d.vertex_stride >= 0x1c ? 0x10 : 0xFFFFFFFFu);
      }
    }

    // Texture: the PS samples an inline T# (image_sample). Isaac's textured
    // sprite vertex format is {pos.xyzw @0, color @0x10, uv.xy @0x1c} in a
    // 64-byte vertex, so the UV lives in the position buffer at offset 0x1c.
    std::shared_ptr<const gcn::Program> ps_prog;
    uint32_t tex_3d_mask = 0;
    if (ps_a >= 0x1000000000ull && ps_a < 0x20000000000ull) {
      ps_prog = gcn::CachedProgram(ps_a, 4096);
      const uint32_t frame = g_presented_frames + 1;
      const bool trace_tex =
          kTexTrackFrame >= 0 && static_cast<int>(frame) == kTexTrackFrame;
      if (trace_tex)
        std::fprintf(stderr, "[textrack] f%u ps=%#lx\n", frame,
                     (unsigned long)ps_a);
      auto texs = gcn::TrackTextures(
          ps_prog, &g_regs[mmSPI_SHADER_USER_DATA_PS_0], trace_tex, ps_a);
      if (!texs.empty()) {
        // Preserve every valid GFX7 T# address. Format support is relevant only
        // when uploading guest memory; a T# with R32F/RG16F/RGBA16F semantics
        // may alias a live renderer RT and must still resolve to that image.
        // https://github.com/torvalds/linux/blob/fce2dfa773ced15f27dd27cd0b482a7473cdcf2a/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h#L6049-L6060
        d.tex_base = texs[0].valid ? texs[0].base : 0;
        d.tex_w = texs[0].width;
        d.tex_h = texs[0].height;
        d.tex_dfmt = texs[0].dfmt;
        d.tex_nfmt = texs[0].nfmt;
        d.tex_tiling = texs[0].tiling_idx;
        d.tex_pitch = texs[0].pitch;
        d.tex_depth = texs[0].depth;
        d.tex_layers = texs[0].layers;
        d.tex_base_array = texs[0].base_array;
        d.tex_view_layers = texs[0].view_layers;
        d.tex_mip_levels = texs[0].mip_levels;
        d.tex_base_mip = texs[0].base_mip;
        d.tex_view_mips = texs[0].view_mips;
        d.tex_min_lod = texs[0].min_lod;
        std::memcpy(d.tex_sampler, texs[0].sampler, sizeof(d.tex_sampler));
        d.tex_pow2_pad = texs[0].pow2_pad;
        d.tex_sampler_valid = texs[0].sampler_valid;
        d.tex_arrayed = texs[0].arrayed;
        d.tex_is_3d = texs[0].is_3d;
        d.tex_force_lod_zero = texs[0].force_lod_zero;
        d.tex_depth_compare = texs[0].depth_compare;
        d.tex_null_descriptor = texs[0].null_descriptor;
        d.uv_data = d.vertex_data;
        d.uv_stride = d.vertex_stride;
        // All sampled textures (binding order), for multi-texture PS (Doom64
        // 3D).
        d.num_texs = static_cast<uint32_t>(texs.size() < 16 ? texs.size() : 16);
        for (uint32_t i = 0; i < d.num_texs; i++) {
          auto& dt = d.texs[i];
          const auto& t = texs[i];
          dt.base = t.valid ? t.base : 0;
          dt.w = t.width;
          dt.h = t.height;
          dt.dfmt = t.dfmt;
          dt.nfmt = t.nfmt;
          dt.tiling = t.tiling_idx;
          dt.pitch = t.pitch;
          dt.depth = t.depth;
          dt.layers = t.layers;
          dt.base_array = t.base_array;
          dt.view_layers = t.view_layers;
          dt.mip_levels = t.mip_levels;
          dt.base_mip = t.base_mip;
          dt.view_mips = t.view_mips;
          dt.min_lod = t.min_lod;
          dt.pow2_pad = t.pow2_pad;
          std::memcpy(dt.sampler, t.sampler, sizeof(dt.sampler));
          dt.sampler_valid = t.sampler_valid;
          dt.arrayed = t.arrayed;
          dt.is_3d = t.is_3d;
          if (t.is_3d)
            tex_3d_mask |= 1u << i;
          dt.force_lod_zero = t.force_lod_zero;
          dt.depth_compare = t.depth_compare;
          dt.storage = t.storage;
          dt.null_descriptor = t.null_descriptor;
        }
        // d.uv_offset was derived from the fetch shader during vertex
        // extraction. DELTA_GPU_TEXFMT: dump sampled texture formats
        // (dfmt/nfmt/tiling/dims) to pin a scrambled draw (e.g. Doom64's menu)
        // to a format/tiling we mishandle.
        static int tf_n = 0;
        if (kTexfmt && tf_n < 24) {
          tf_n++;
          std::fprintf(stderr,
                       "[texfmt] base=%#lx %ux%u pitch=%u dfmt=%u nfmt=%u "
                       "tiling=%u rt=%ux%u\n",
                       (unsigned long)texs[0].base, texs[0].width,
                       texs[0].height, texs[0].pitch, texs[0].dfmt,
                       texs[0].nfmt, texs[0].tiling_idx, d.rt_w, d.rt_h);
        }
      }
    }
    // Recompiled-shader path: Recompile the VS/PS pair (cached) and resolve the
    // live vertex-attribute buffers, so the renderer can run the game's actual
    // shaders. The heuristic fields above stay populated as the fallback.
    // DELTA_GPU_SKIPSH=addr[,addr...] (hex): refuse to recompile draws whose
    // VS or PS lives at one of these guest addresses — shader-hang bisection.
    static const std::vector<uint64_t> kSkipSh = [] {
      std::vector<uint64_t> v;
      if (const char* e = kSkipShList)
        for (const char* p = e; *p;) {
          char* end;
          const uint64_t a = std::strtoull(p, &end, 16);
          if (end == p)
            break;
          v.push_back(a);
          p = *end == ',' ? end + 1 : end;
        }
      return v;
    }();
    const bool sh_skipped =
        !kSkipSh.empty() &&
        (std::find(kSkipSh.begin(), kSkipSh.end(), vs_a) != kSkipSh.end() ||
         std::find(kSkipSh.begin(), kSkipSh.end(), ps_a) != kSkipSh.end());
    const char* recomp_status = kRecompOn ? "bad-address" : "disabled";
    if (sh_skipped)
      recomp_status = "skipsh";
    if (!sh_skipped && kRecompOn && vs_a >= 0x1000000000ull &&
        vs_a < 0x20000000000ull &&
        (!ps_a || (ps_a >= 0x1000000000ull && ps_a < 0x20000000000ull))) {
      // Attribute translation depends on the fetch shader as well as the VS/PS
      // code. Keying only by VS/PS can reuse an attributed plan for a
      // procedural draw (or vice versa), and a single mixed hash can alias
      // unrelated shader pairs.
      static std::unordered_map<ShaderKey, gcn::Recompiled, ShaderKeyHash>
          sh_cache;
      const bool neo = gcn::DefaultIsaMode() == gcn::IsaMode::kNeo;
      const uint32_t ps_input_ena = g_regs[mmSPI_PS_INPUT_ENA];
      ShaderKey key{vs_a, ps_a, fetch, ps_input_ena, tex_3d_mask, neo};
      auto it = sh_cache.find(key);
      if (it == sh_cache.end())
        it = sh_cache
                 .emplace(key, gcn::Recompile(
                                   reinterpret_cast<const uint32_t*>(vs_a),
                                    reinterpret_cast<const uint32_t*>(ps_a),
                                    &g_regs[mmSPI_SHADER_USER_DATA_VS_0],
                                    &g_regs[mmSPI_SHADER_USER_DATA_PS_0],
                                    ps_input_ena, tex_3d_mask))
                 .first;
      gcn::Recompiled& rc = it->second;
      recomp_status = rc.ok ? "bad-attrs" : "rejected";
      if (rc.ok) {
        d.ps4_neo = neo;
        const uint32_t* vud2 = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
        const auto vs_prog = gcn::CachedProgram(vs_a, 4096);
        const auto direct_vbs =
            gcn::ResolveDirectVertexBuffers(vs_prog, rc.attrs, vud2);
        gcn::VBuffer attr_vbs[8];
        uint32_t attr_count = 0;
        bool good = true;
        for (size_t i = 0; i < rc.attrs.size() && i < 8; i++) {
          auto& a = rc.attrs[i];
          if (!a.direct_fetch && a.table_sgpr + 1 >= 16) {
            good = false;
            break;
          }
          gcn::VBuffer vb;
          if (a.direct_fetch) {
            vb = direct_vbs[i];
          } else {
            uint64_t tbl =
                (static_cast<uint64_t>(vud2[a.table_sgpr + 1] & 0xFFFF) << 32) |
                vud2[a.table_sgpr];
            if (tbl < 0x1000000000ull || tbl >= 0x20000000000ull) {
              good = false;
              break;
            }
            vb = gcn::DecodeVBuffer(
                reinterpret_cast<const uint32_t*>(tbl + a.vbuf_dword_off * 4));
          }
          if (vb.base < 0x1000000000ull || vb.base >= 0x20000000000ull) {
            good = false;
            break;
          }
          // A stride-0 V# is a per-draw constant input (all vertices read the
          // same record); it becomes a stride-0 Vulkan binding below. Instance
          // stepping is NOT a stride-0 V# -- it uses a normal stride indexed by
          // the fetch shader -- so a zero stride is unambiguously a constant,
          // not per-instance.
          attr_vbs[attr_count++] = vb;
        }
        // Group attributes into vertex bindings. Attributes that interleave in
        // one buffer (same stride, base within one stride of the binding base)
        // share a binding with distinct offsets; attributes fed from a separate
        // buffer (different stride, or a base more than one stride away) get
        // their own binding. SotC streams position/normal/uv/etc from distinct
        // buffers with distinct strides, which the old single-stream model
        // declined outright.
        uint32_t attr_binding[8] = {};
        if (good && attr_count) {
          for (uint32_t i = 0; i < attr_count; i++) {
            const gcn::VBuffer& vb = attr_vbs[i];
            int sel = -1;
            for (uint32_t j = 0; j < d.num_vbufs; j++) {
              if (d.vbufs[j].stride != vb.stride)
                continue;
              uint64_t b = reinterpret_cast<uint64_t>(d.vbufs[j].data);
              uint64_t lo = b < vb.base ? b : vb.base;
              uint64_t hi = b < vb.base ? vb.base : b;
              if (hi - lo < vb.stride) {
                sel = static_cast<int>(j);
                break;
              }
            }
            if (sel < 0) {
              if (d.num_vbufs >= 8) {
                good = false;
                recomp_status = "attr-nbind";
                break;
              }
              sel = static_cast<int>(d.num_vbufs);
              d.vbufs[d.num_vbufs++] = {reinterpret_cast<const void*>(vb.base),
                                        vb.stride, vb.num_records};
            } else {
              auto& bind = d.vbufs[sel];
              if (vb.base < reinterpret_cast<uint64_t>(bind.data))
                bind.data = reinterpret_cast<const void*>(vb.base);
              bind.num_records = std::min(bind.num_records, vb.num_records);
            }
            attr_binding[i] = static_cast<uint32_t>(sel);
          }
          // Second pass: offsets are relative to each binding's final (lowest)
          // base.
          for (uint32_t i = 0; good && i < attr_count; i++) {
            const gcn::ShaderAttr& a = rc.attrs[i];
            const gcn::VBuffer& vb = attr_vbs[i];
            const uint32_t b = attr_binding[i];
            const uint64_t offset =
                vb.base - reinterpret_cast<uint64_t>(d.vbufs[b].data);
            // Strided bindings must keep every attribute inside one record; a
            // stride-0 (constant) binding has no record extent to bound.
            if (d.vbufs[b].stride && offset >= d.vbufs[b].stride) {
              good = false;
              recomp_status = "attr-offset";
              break;
            }
            d.vattrs[d.num_vattrs++] = {
                a.location,  b,       static_cast<uint32_t>(offset),
                a.num_comps, vb.dfmt, vb.nfmt};
          }
          if (good) {
            // vertex_data/vertex_stride mirror the first per-vertex (strided)
            // binding for the heuristic fallback + clear detection;
            // vertex_count is bounded by the smallest strided binding's record
            // count (stride-0 constant bindings do not constrain the vertex
            // count).
            uint32_t primary = 0;
            while (primary < d.num_vbufs && !d.vbufs[primary].stride)
              primary++;
            d.vertex_data = d.vbufs[primary < d.num_vbufs ? primary : 0].data;
            d.vertex_stride =
                primary < d.num_vbufs ? d.vbufs[primary].stride : 0;
            uint32_t count = UINT32_MAX;
            for (uint32_t j = 0; j < d.num_vbufs; j++)
              if (d.vbufs[j].stride)
                count = std::min(count, d.vbufs[j].num_records);
            d.vertex_count = count == UINT32_MAX ? 0 : count;
          }
        }
        // Resolve every cbuffer V# the emitted VS/PS reads. Bindings are
        // assigned by the translator and shared across both stages in
        // descriptor set 1.
        bool resolved_vs_cbuf = false;
        auto resolve_cbufs =
            [&](const std::vector<gcn::ShaderCbuf>& cbufs,
                const uint32_t* user_data,
                const std::shared_ptr<const gcn::Program>& prog,
                bool vertex_stage) {
              // Resolve every cbuffer V# the stage reads, following EUD/SRT
              // pointer chains (FOX loads the V# through an extended-user-data
              // pointer, so it is not sitting directly in user data at
              // cb.ud_sgpr).
              auto resolved = gcn::ResolveCbuffers(prog, user_data);
              for (const auto& cb : cbufs) {
                if (cb.binding >= 8)
                  continue;
                gcn::VBuffer vb{};
                auto rit =
                    resolved.find(cb.ud_sgpr | (cb.pointer ? 0x100u : 0u));
                if (rit != resolved.end())
                  vb = rit->second;  // EUD-resolved V# (handles indirection)
                else if (cb.pointer && cb.ud_sgpr + 1 < 16)
                  vb.base = user_data[cb.ud_sgpr] |
                            (uint64_t)(user_data[cb.ud_sgpr + 1] & 0xFFFF)
                                << 32;  // flat pointer inline in user data
                else if (!cb.pointer && cb.ud_sgpr + 3 < 16)
                  vb = gcn::DecodeVBuffer(&user_data[cb.ud_sgpr]);  // inline V#
                // An s_load table carries no size; the shader's own highest
                // read bounds the window.
                uint64_t bytes = vb.stride
                                     ? (uint64_t)vb.stride * vb.num_records
                                     : vb.num_records;
                if (cb.pointer)
                  bytes = (uint64_t)cb.num_dwords * 4;
                if (vb.base < 0x1000000000ull || vb.base >= 0x20000000000ull ||
                    !bytes || bytes > 0xFFFFFFFFull ||
                    vb.base + bytes > 0x20000000000ull)
                  continue;
                d.cbufs[cb.binding] = {vb.base, static_cast<uint32_t>(bytes)};
                d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
                if (cb.pointer)
                  continue;  // an SRT root is not a transform buffer
                if (vertex_stage && !resolved_vs_cbuf) {
                  resolved_vs_cbuf = true;
                  d.cbuf_base = vb.base;
                  d.cbuf_size = static_cast<uint32_t>(bytes);
                  if (bytes >= sizeof(d.mvp))
                    std::memcpy(d.mvp, reinterpret_cast<const void*>(vb.base),
                                sizeof(d.mvp));
                }
              }
            };
        // Resolve the live V# behind every raw buffer the emitted VS/PS reads
        // with MUBUF. Same scalar replay as the cbuffers: the descriptor is
        // read at the instruction that consumes it, so an SRT-chained V# lands
        // here as well as an inline one.
        auto resolve_bufs = [&](const std::vector<gcn::ShaderBuffer>& bufs,
                                const uint32_t* user_data,
                                const std::shared_ptr<const gcn::Program>& prog,
                                const char* stage) {
          if (bufs.empty())
            return;
          const auto resolved =
              gcn::ResolveShaderBuffers(prog, bufs, user_data);
          for (size_t i = 0; i < bufs.size(); i++) {
            const gcn::ShaderBuffer& sb = bufs[i];
            if (sb.binding >= rhi::DrawInfo::kMaxBuffers)
              continue;
            gcn::VBuffer vb = resolved[i];
            if (!vb.base && sb.srsrc_sgpr + 3 < 16)
              vb = gcn::DecodeVBuffer(&user_data[sb.srsrc_sgpr]);
            uint64_t bytes = vb.stride ? (uint64_t)vb.stride * vb.num_records
                                       : vb.num_records;
            const bool ok = vb.base >= 0x1000000000ull &&
                            vb.base < 0x20000000000ull && bytes &&
                            bytes <= 0xFFFFFFFFull &&
                            vb.base + bytes <= 0x20000000000ull;
            if (kRawBufTrace) {
              static int n = 0;
              if (n++ < 64)
                std::fprintf(stderr,
                             "[rawbuf] %s vs=%#lx bind=%u pc=%#x s%u "
                             "base=%#lx stride=%u nrec=%u bytes=%lu "
                             "vcount=%u icount=%u %s\n",
                             stage, (unsigned long)vs_a, sb.binding, sb.use_pc,
                             sb.srsrc_sgpr, (unsigned long)vb.base, vb.stride,
                             vb.num_records, (unsigned long)bytes,
                             d.vertex_count, d.index_count,
                             ok ? "resolved" : "UNRESOLVED");
            }
            if (!ok)
              continue;
            d.bufs[sb.binding] = {vb.base, static_cast<uint32_t>(bytes)};
            d.num_bufs = std::max(d.num_bufs, sb.binding + 1);
          }
        };
        if (good) {
          resolve_cbufs(rc.vs_cbufs, vud2, vs_prog, true);
          if (ps_a >= 0x1000000000ull && ps_a < 0x20000000000ull)
            resolve_cbufs(rc.ps_cbufs, &g_regs[mmSPI_SHADER_USER_DATA_PS_0],
                          ps_prog ? ps_prog : gcn::CachedProgram(ps_a, 4096),
                          false);
          for (const auto& cb : rc.vs_cbufs)
            d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
          for (const auto& cb : rc.ps_cbufs)
            d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
          resolve_bufs(rc.vs_bufs, vud2, vs_prog, "vs");
          if (ps_a >= 0x1000000000ull && ps_a < 0x20000000000ull)
            resolve_bufs(rc.ps_bufs, &g_regs[mmSPI_SHADER_USER_DATA_PS_0],
                         ps_prog ? ps_prog : gcn::CachedProgram(ps_a, 4096),
                         "ps");
        }
        if (good) {
          d.vs_addr = vs_a;
          d.ps_addr = ps_a;
          d.recomp = &rc;
          recomp_status = "ok";
        } else {
          d.num_vattrs = 0;
          d.num_vbufs = 0;
        }
      }
    }
    if (auto_vertex_count && auto_vertex_count <= 0x100000)
      d.vertex_count = auto_vertex_count;
    if (!g_frame_active) {
      rhi::BeginFrame(rhi::DefaultRenderer());
      g_frame_active = true;
    }
    // DELTA_GPU_BLITDUMP: for the first few draws targeting a wide
    // (scanout-sized) RT, Disassemble the PS and report what TrackTextures
    // resolved. Pins why Undertale's surface->scanout blit renders untextured
    // (tex=0): is the PS doing an image_sample we miss, or is the T# pointing
    // outside the guest range?
    static int bd_n = 0;
    bool blit_target_bound = !kBlitRt || d.rt_base == kBlitRt;
    for (uint32_t i = 0; kBlitRt && i < std::min(d.mrt_count, 8u); i++)
      blit_target_bound |= d.mrt_base[i] == kBlitRt;
    if (kBlitDump && blit_target_bound && (kBlitRt || d.rt_w >= 1280) &&
        bd_n < 6) {
      bd_n++;
      std::fprintf(stderr,
                   "[blit] #%d rt=%#lx %ux%u VS=%#lx PS=%#lx tex_base=%#lx "
                   "%ux%u num_vattrs=%u "
                   "stride=%u idx=%u blendCtl=%#x\n",
                   bd_n, (unsigned long)d.rt_base, d.rt_w, d.rt_h,
                   (unsigned long)vs_a, (unsigned long)ps_a,
                   (unsigned long)d.tex_base, d.tex_w, d.tex_h, d.num_vattrs,
                   d.vertex_stride, d.index_count, d.blend_control);
      std::fprintf(stderr, "[blit]   MRT(%u):", d.mrt_count);
      for (uint32_t i = 0; i < std::min(d.mrt_count, 8u); i++)
        std::fprintf(stderr, " %#lx", (unsigned long)d.mrt_base[i]);
      std::fprintf(stderr, "\n");
      std::fprintf(stderr,
                   "[blit]   CB0 pitch=%#x slice=%#x info=%#x attrib=%#x\n",
                   g_regs[mmCB_COLOR0_PITCH], g_regs[mmCB_COLOR0_SLICE],
                   g_regs[mmCB_COLOR0_INFO], g_regs[mmCB_COLOR0_ATTRIB]);
      if (ps_a >= 0x1000000000ull && ps_a < 0x20000000000ull) {
        auto texs = gcn::TrackTextures(gcn::CachedProgram(ps_a, 4096),
                                       &g_regs[mmSPI_SHADER_USER_DATA_PS_0],
                                       false, ps_a);
        std::fprintf(stderr, "[blit]   TrackTextures -> %zu T#\n", texs.size());
        for (auto& t : texs)
          std::fprintf(stderr,
                       "[blit]     T# base=%#lx %ux%u pitch=%u dfmt=%u nfmt=%u "
                       "tiling=%u\n",
                       (unsigned long)t.base, t.width, t.height, t.pitch,
                       t.dfmt, t.nfmt, t.tiling_idx);
        const uint32_t* ps_code = reinterpret_cast<const uint32_t*>(ps_a);
        uint32_t ps_words = gcn::CodeLength(ps_code, 4096);
        gcn::Disassemble(ps_code, ps_words ? ps_words : 512, "blit.PS");
        if (bd_n == 1 && d.recomp) {
          const uint32_t* pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];
          std::fprintf(stderr, "[blit]   PS user data:");
          for (uint32_t i = 0; i < 16; i++)
            std::fprintf(stderr, " %08x", pud[i]);
          std::fprintf(stderr, "\n");
          for (const auto& cb : d.recomp->ps_cbufs) {
            const auto& resolved = d.cbufs[cb.binding];
            std::fprintf(stderr,
                         "[blit]   PS CB binding=%u sgpr=%u dwords=%u "
                         "base=%#lx size=%u\n",
                         cb.binding, cb.ud_sgpr, cb.num_dwords,
                         (unsigned long)resolved.base, resolved.size);
            if (!resolved.base)
              continue;
            const uint32_t* words =
                reinterpret_cast<const uint32_t*>(resolved.base);
            uint32_t rows =
                std::min({(cb.num_dwords + 3) / 4, resolved.size / 16, 64u});
            for (uint32_t row = 0; row < rows; row++) {
              float values[4];
              std::memcpy(values, words + row * 4, sizeof(values));
              std::fprintf(stderr,
                           "[blit]     %03x: %08x %08x %08x %08x | %.6g %.6g "
                           "%.6g %.6g\n",
                           row * 4, words[row * 4], words[row * 4 + 1],
                           words[row * 4 + 2], words[row * 4 + 3], values[0],
                           values[1], values[2], values[3]);
            }
          }
        }
      }
    }
    // DELTA_GPU_DRAWLIST: one line per draw BEFORE any vertex_data gating, so
    // draws dropped for null vertex_data/recomp (e.g. Doom64's 3D world
    // geometry) are visible -- distinguishes "world draws never submitted" from
    // "submitted but dropped because vertex/shader resolution failed".
    static int dl_n = 0;
    // Wall-clock gate: the title/menu floods the early run, so only start
    // logging after DELTA_GPU_DRAWLIST_AFTER seconds (default 90), by when the
    // level has loaded -- then log EVERY draw so the in-level pattern (incl.
    // world geometry, if any reaches us) is captured.
    static const auto kDlStart = std::chrono::steady_clock::now();
    if (kDrawList && dl_n < 400 &&
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - kDlStart)
                .count() >= kDlAfter) {
      dl_n++;
      std::fprintf(
          stderr,
          "[draw] count=%u rt=%#lx %ux%u tex=%#lx %ux%u vd=%d "
          "recomp=%s num_vattrs=%u prim=%u VS=%#lx PS=%#lx fetch=%#lx\n",
          d.index_data ? d.index_count : d.vertex_count,
          (unsigned long)d.rt_base, d.rt_w, d.rt_h, (unsigned long)d.tex_base,
          d.tex_w, d.tex_h, d.vertex_data ? 1 : 0, recomp_status, d.num_vattrs,
          g_regs[mmVGT_PRIMITIVE_TYPE], (unsigned long)vs_a,
          (unsigned long)ps_a, (unsigned long)fetch);
    }
    // DELTA_GPU_MASKTRACE: the raw colour-write state of a draw. On GCN the
    // channels a draw writes to MRTn are (CB_TARGET_MASK & CB_SHADER_MASK)
    // nibble n: TARGET_MASK is the app's setRenderTargetMask, SHADER_MASK is
    // set by the driver from what the compiled PS actually exports. We read
    // TARGET_MASK only to decide whether a target is bound and never read
    // SHADER_MASK at all, so this dumps both next to CB_COLOR_CONTROL and the
    // recompiler's own export mask to check they agree.
    //   DELTA_GPU_MASKTRACE_AFTER=<sec>  delay before logging (default 0)
    //   DELTA_GPU_MASKTRACE_MAX=<n>      max lines (default 200)
    //   DELTA_GPU_MASKTRACE_RT=<hex>     only draws touching this CB base
    if (kMaskTrace) {
      static const auto kMtStart = std::chrono::steady_clock::now();
      static int mt_n = 0;
      bool rt_hit = !kMtRt || d.rt_base == kMtRt;
      for (uint32_t i = 0; kMtRt && i < std::min(d.mrt_count, 8u); i++)
        rt_hit |= d.mrt_base[i] == kMtRt;
      if (rt_hit && mt_n < kMtMax &&
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - kMtStart)
                  .count() >= kMtAfter) {
        mt_n++;
        const uint32_t tm = g_regs[mmCB_TARGET_MASK];
        const uint32_t sm = g_regs[mmCB_SHADER_MASK];
        const uint32_t cc = g_regs[mmCB_COLOR_CONTROL];
        std::fprintf(stderr,
                     "[mask] #%d rt=%#lx %ux%u mrt=%u cb0base=%#lx cb1base=%#lx "
                     "info1=%#x TARGET_MASK=%#x "
                     "SHADER_MASK=%#x eff0=%#x COLOR_CONTROL=%#x mode=%u "
                     "rop=%#x BLEND0=%#x en=%u psMrtMask=%#x recomp=%s "
                     "info0=%#x wr(sm/tm)=%u/%u PS=%#lx count=%u\n",
                     mt_n, (unsigned long)d.rt_base, d.rt_w, d.rt_h,
                     d.mrt_count, (unsigned long)g_regs.CbColorBase(0),
                     (unsigned long)g_regs.CbColorBase(1),
                     g_regs[mmCB_COLOR0_INFO + kCbColorStride],
                     tm, sm, (tm & sm) & 0xF, cc, (cc >> 4) & 0x7,
                     (cc >> 16) & 0xFF, d.blend_control, d.blend_enable,
                     d.recomp ? (unsigned)d.recomp->ps_mrt_mask : 0u,
                     recomp_status, d.mrt_info[0],
                     g_shader_mask_writes, g_target_mask_writes,
                     (unsigned long)ps_a,
                     d.index_data ? d.index_count : d.vertex_count);
      }
    }
    // DELTA_GPU_SPRITEDUMP: for the first few TEXTURED draws, dump the resolved
    // transform + first vertex (pos/uv via the resolved attrs) + texture, to
    // pin why textured draws render black (degenerate MVP vs UV=0 vs blend).
    static bool sprite_dis_done = false;
    static int sd_n = 0;
    if (kSpriteDump && d.recomp && !d.recomp->ps_texs.empty() &&
        d.vertex_data && sd_n < 12) {
      sd_n++;
      const float* m = d.mvp;
      std::fprintf(
          stderr,
          "[sprite] tex=%#lx %ux%u tiling=%u stride=%u num_vattrs=%u rt=%#lx "
          "%ux%u VS=%#lx PS=%#lx blendCtl=%#x depth=%#lx/%d/%d/%u "
          "mvp=[%.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f / ... / "
          "%.3f %.3f %.3f %.3f]\n",
          (unsigned long)d.tex_base, d.tex_w, d.tex_h, d.tex_tiling,
          d.vertex_stride, d.num_vattrs, (unsigned long)d.rt_base, d.rt_w,
          d.rt_h, (unsigned long)d.vs_addr, (unsigned long)d.ps_addr,
          d.blend_control, (unsigned long)d.depth_base, d.depth_test_enable,
          d.depth_write_enable, d.depth_func, m[0], m[1], m[2], m[3], m[4],
          m[5], m[6], m[7], m[12], m[13], m[14], m[15]);
      if (kSpriteDis && !sprite_dis_done) {
        sprite_dis_done = true;
        const uint32_t* ps = reinterpret_cast<const uint32_t*>(d.ps_addr);
        const uint32_t words = gcn::CodeLength(ps, 4096);
        gcn::Disassemble(ps, words ? words : 512, "sprite.PS");
        std::fprintf(stderr, "[sprite]   PS user data:");
        for (uint32_t i = 0; i < 16; i++)
          std::fprintf(stderr, " %08x", d.ps_user_data[i]);
        std::fprintf(stderr, "\n");
        for (uint32_t i = 0; i < d.num_texs; i++) {
          const auto& tex = d.texs[i];
          std::fprintf(stderr,
                       "[sprite]   tex%u=%#lx %ux%u pitch=%u dfmt=%u nfmt=%u "
                       "tiling=%u sampler=%08x/%08x/%08x/%08x\n",
                       i, (unsigned long)tex.base, tex.w, tex.h, tex.pitch,
                       tex.dfmt, tex.nfmt, tex.tiling, tex.sampler[0],
                       tex.sampler[1], tex.sampler[2], tex.sampler[3]);
        }
        for (const auto& inst : *gcn::CachedProgram(d.ps_addr, 4096)) {
          const uint32_t w = inst.raw[0], w1 = inst.raw[1];
          if (inst.enc == gcn::Enc::kVintrp) {
            std::fprintf(stderr,
                         "[sprite]   interp pc=%#x op=%u attr=%u chan=%u v%u\n",
                         inst.pc, (w >> 16) & 3, (w >> 10) & 0x3f, (w >> 8) & 3,
                         (w >> 18) & 0xff);
          } else if (inst.enc == gcn::Enc::kMimg) {
            std::fprintf(stderr,
                         "[sprite]   image pc=%#x op=%#x dmask=%#x vaddr=v%u "
                         "vdata=v%u srsrc=s%u ssamp=s%u da=%u\n",
                         inst.pc, inst.opcode, (w >> 8) & 0xf, w1 & 0xff,
                         (w1 >> 8) & 0xff, ((w1 >> 16) & 0x1f) * 4,
                         ((w1 >> 21) & 0x1f) * 4, (w >> 14) & 1);
          } else if (inst.enc == gcn::Enc::kExp) {
            std::fprintf(stderr,
                         "[sprite]   export pc=%#x target=%u en=%#x compr=%u "
                         "v=[%u %u %u %u]\n",
                         inst.pc, (w >> 4) & 0x3f, w & 0xf, (w >> 10) & 1,
                         w1 & 0xff, (w1 >> 8) & 0xff, (w1 >> 16) & 0xff,
                         (w1 >> 24) & 0xff);
          }
        }
      }
      for (uint32_t a = 0; a < d.num_vattrs && a < 8; a++) {
        const auto& attr = d.vattrs[a];
        if (attr.binding >= d.num_vbufs || !d.vbufs[attr.binding].data)
          continue;
        const auto& binding = d.vbufs[attr.binding];
        std::fprintf(stderr,
                     "[sprite]   attr%u loc=%u bind=%u off=%u stride=%u nc=%u "
                     "dfmt=%u nfmt=%u",
                     a, attr.location, attr.binding, attr.offset,
                     binding.stride, attr.num_comps, attr.dfmt, attr.nfmt);
        const uint32_t vertices = std::min(d.vertex_count, 3u);
        for (uint32_t v = 0; v < vertices; v++) {
          const uint8_t* raw = static_cast<const uint8_t*>(binding.data) +
                               static_cast<size_t>(v) * binding.stride +
                               attr.offset;
          if (attr.dfmt == 10) {
            std::fprintf(stderr, " v%u=[%.3f %.3f %.3f %.3f]", v,
                         raw[0] / 255.f, raw[1] / 255.f, raw[2] / 255.f,
                         raw[3] / 255.f);
          } else {
            const float* f = reinterpret_cast<const float*>(raw);
            std::fprintf(stderr, " v%u=[%.4f %.4f %.4f %.4f]", v, f[0],
                         attr.num_comps > 1 ? f[1] : 0.f,
                         attr.num_comps > 2 ? f[2] : 0.f,
                         attr.num_comps > 3 ? f[3] : 0.f);
          }
        }
        std::fprintf(stderr, "\n");
      }
    }
    // DELTA_GPU_VATTRDUMP: raw first-vertex bytes per attribute for small
    // multi-attribute draws (UI quads). A UI pixel shader multiplies its
    // textures by interpolated vertex COLORS, so an all-black UI splits into
    // "the color bytes in guest memory are zero" vs "the fetch path zeroes
    // them" -- this prints the guest bytes.
    static int vattr_dumped = 0;
    if (kVattrDump && d.num_vattrs >= 2 && d.vertex_data &&
        d.vertex_count <= 8 && vattr_dumped < 24) {
      vattr_dumped++;
      std::fprintf(stderr,
                   "[vattr] vcount=%u prim=%#x nattrs=%u nbufs=%u rt=%#lx\n",
                   d.vertex_count, d.prim_type, d.num_vattrs, d.num_vbufs,
                   (unsigned long)d.rt_base);
      for (uint32_t a = 0; a < d.num_vattrs && a < 8; a++) {
        const auto& attr = d.vattrs[a];
        const auto& vb = d.vbufs[attr.binding < d.num_vbufs ? attr.binding : 0];
        const auto* p = static_cast<const uint8_t*>(vb.data);
        std::fprintf(stderr,
                     "[vattr]  loc=%u bind=%u off=%u dfmt=%u nfmt=%u "
                     "comps=%u stride=%u v0=",
                     attr.location, attr.binding, attr.offset, attr.dfmt,
                     attr.nfmt, attr.num_comps, vb.stride);
        const uint32_t bytes = 8;  // covers every <=64-bit attribute format
        if (p && utl::isMemoryRangeMapped(p + attr.offset, bytes))
          for (uint32_t b = 0; b < bytes; b++)
            std::fprintf(stderr, "%02x", p[attr.offset + b]);
        else
          std::fprintf(stderr, "(unmapped)");
        std::fprintf(stderr, "\n");
      }
    }
    // DELTA_GPU_GEOMDUMP: dump the full state of high-index (3D level geometry)
    // draws to diagnose why Doom64's 3D renders black: textured vs
    // vertex-color, the texture format/tiling, the cbuffer transform, and a
    // vertex position (on-screen check). Gated on index_count>=500 so it only
    // fires in a 3D level.
    // DELTA_GPU_GEOMMIN overrides the index-count gate (default 500) so the
    // dump can also catch Doom64's lower-index level draws.
    static int gd_n = 0, gd_seen = 0;
    // Sample periodically across the WHOLE run (every 100th qualifying world
    // draw) so we can see whether the camera/view ever moves -- not just the
    // first frames.
    if (kGeomDump && d.index_count >= kGeomMin && d.vertex_data &&
        (gd_seen++ % 100 == 0) && gd_n < 300) {
      gd_n++;
      const float* cb =
          d.cbuf_base ? reinterpret_cast<const float*>(d.cbuf_base) : nullptr;
      const auto* vb = static_cast<const uint8_t*>(d.vertex_data);
      const float* p0 = reinterpret_cast<const float*>(vb + d.vattrs[0].offset);
      std::fprintf(stderr,
                   "[geom] idx=%u ps_texs=%zu tex=%#lx %ux%u tiling=%u "
                   "pitch=%u rt=%#lx %ux%u "
                   "blend=%#x mask=%#x cc=%#x num_vattrs=%u stride=%u "
                   "pos0=[%.1f %.1f %.1f] cbuf_base=%#lx\n",
                   d.index_count, d.recomp ? d.recomp->ps_texs.size() : 0,
                   (unsigned long)d.tex_base, d.tex_w, d.tex_h, d.tex_tiling,
                   d.tex_pitch, (unsigned long)d.rt_base, d.rt_w, d.rt_h,
                   d.blend_control, d.target_mask, d.color_control,
                   d.num_vattrs, d.vertex_stride, p0[0], p0[1], p0[2],
                   (unsigned long)d.cbuf_base);
      if (cb)
        std::fprintf(stderr,
                     "  cbuf=[%.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f / %.3f "
                     "%.3f %.3f %.3f / %.3f %.3f %.3f %.3f]\n",
                     cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7],
                     cb[8], cb[9], cb[10], cb[11], cb[12], cb[13], cb[14],
                     cb[15]);
      // Project a handful of vertices through the cbuf MVP (both row- and
      // column- major) and count how many land in NDC [-1,1] -- tells us if the
      // world geometry is on-screen (so the black is a PS/sampling issue) or
      // off-screen (a VS/cbuffer-resolution issue).
      if (cb) {
        auto proj = [&](const float* p, bool col_major, float out[4]) {
          float v[4] = {p[0], p[1], p[2], 1.0f};
          for (int r = 0; r < 4; r++) {
            float s = 0;
            for (int c = 0; c < 4; c++)
              s += (col_major ? cb[c * 4 + r] : cb[r * 4 + c]) * v[c];
            out[r] = s;
          }
        };
        int on_r = 0, on_c = 0, on_rfz = 0, on_cfz = 0,
            n = d.index_count < 64 ? d.index_count : 64;
        const uint16_t* i16 = (d.index_type == 0)
                                  ? static_cast<const uint16_t*>(d.index_data)
                                  : nullptr;
        const uint32_t* i32 = (d.index_type == 1)
                                  ? static_cast<const uint32_t*>(d.index_data)
                                  : nullptr;
        float first_r[4] = {0}, first_c[4] = {0};
        auto onscreen = [](float* o) {
          if (o[3] <= 0.0001f)
            return false;
          float x = o[0] / o[3], y = o[1] / o[3], z = o[2] / o[3];
          return x >= -1 && x <= 1 && y >= -1 && y <= 1 && z >= -1 && z <= 1;
        };
        for (int i = 0; i < n; i++) {
          // Use the INDEX buffer to fetch the real vertex (these are indexed
          // draws; a linear 0..n read hits unused verts at the buffer head).
          uint32_t idx = i16 ? i16[i] : i32 ? i32[i] : (uint32_t)i;
          const float* p = reinterpret_cast<const float*>(
              vb + (size_t)idx * d.vertex_stride + d.vattrs[0].offset);
          float r4[4], c4[4];
          proj(p, false, r4);
          proj(p, true, c4);
          // Same but with z negated -- validates the DELTA_GPU_VSFLIPZ
          // hypothesis (does flipping the position z bring the geometry
          // on-screen?).
          float pf[3] = {p[0], p[1], -p[2]}, rf[4], cf[4];
          proj(pf, false, rf);
          proj(pf, true, cf);
          if (i == 0) {
            for (int k = 0; k < 4; k++) {
              first_r[k] = r4[k];
              first_c[k] = c4[k];
            }
          }
          if (onscreen(r4))
            on_r++;
          if (onscreen(c4))
            on_c++;
          if (onscreen(rf))
            on_rfz++;
          if (onscreen(cf))
            on_cfz++;
        }
        std::fprintf(
            stderr,
            "  proj n=%d onscreen row=%d col=%d | flipZ row=%d col=%d | "
            "v0row=[%.2f %.2f %.2f %.2f] v0col=[%.2f %.2f %.2f %.2f]\n",
            n, on_r, on_c, on_rfz, on_cfz, first_r[0], first_r[1], first_r[2],
            first_r[3], first_c[0], first_c[1], first_c[2], first_c[3]);
        // The game is in real gameplay, so a valid view transform exists. Maybe
        // the VS projects a DIFFERENT attribute than attr0. Project EACH
        // >=3-comp attr (over the indexed verts) and report which, if any,
        // lands on-screen -- that identifies the true position attribute the
        // renderer should be feeding.
        uint32_t first_idx = i16 ? i16[0] : i32 ? i32[0] : 0;
        for (uint32_t a = 0; a < d.num_vattrs && a < 8; a++) {
          if (d.vattrs[a].num_comps < 3)
            continue;
          int on_a = 0;
          for (int i = 0; i < n; i++) {
            uint32_t idx = i16 ? i16[i] : i32 ? i32[i] : (uint32_t)i;
            const float* p = reinterpret_cast<const float*>(
                vb + (size_t)idx * d.vertex_stride + d.vattrs[a].offset);
            float c4[4];
            proj(p, true, c4);
            if (onscreen(c4))
              on_a++;
          }
          const float* pv = reinterpret_cast<const float*>(
              vb + (size_t)first_idx * d.vertex_stride + d.vattrs[a].offset);
          std::fprintf(stderr,
                       "    attr%u off=%u nc=%u dfmt=%u v0=[%.2f %.2f %.2f] "
                       "onscreen(col)=%d\n",
                       a, d.vattrs[a].offset, d.vattrs[a].num_comps,
                       d.vattrs[a].dfmt, pv[0], pv[1], pv[2], on_a);
        }
      }
      // Sample the bound texture's ALPHA: is the source genuinely alpha=0 (so
      // the PS must compute opacity elsewhere / a recompiler alpha bug) or
      // alpha=255 (so our load zeroes it)? This decides why the src-alpha blend
      // makes walls invisible.
      if (d.tex_base >= 0x1000000000ull && d.tex_base < 0x20000000000ull &&
          d.tex_w && d.tex_h) {
        const uint32_t* tp = reinterpret_cast<const uint32_t*>(d.tex_base);
        uint64_t n = (uint64_t)d.tex_w * d.tex_h,
                 step = n > 4096 ? n / 4096 : 1, a_nz = 0, rgb_nz = 0;
        for (uint64_t i = 0; i < n; i += step) {
          uint32_t px = tp[i];
          if (px >> 24)
            a_nz++;
          if (px & 0x00FFFFFF)
            rgb_nz++;
        }
        std::fprintf(
            stderr,
            "  texAlpha: px0=%#010x alphaNonZero=%llu/%llu rgbNonZero=%llu\n",
            tp[0], (unsigned long long)a_nz, (unsigned long long)(n / step),
            (unsigned long long)rgb_nz);
      }
      // Dump the PS's texture-load pattern: SMRD (op/sdst/sbase/imm/off) + MIMG
      // srsrc, and the first 8 user-data dwords, to see how the 4 T#s are
      // loaded.
      auto ps_i =
          gcn::Decode(reinterpret_cast<const uint32_t*>(d.ps_addr), 4096);
      const uint32_t* pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];
      std::fprintf(
          stderr, "  ps_ud=[%08x %08x %08x %08x %08x %08x %08x %08x]\n", pud[0],
          pud[1], pud[2], pud[3], pud[4], pud[5], pud[6], pud[7]);
      for (auto& in : ps_i) {
        if (in.enc == gcn::Enc::kSmrd) {
          uint32_t w = in.raw[0];
          std::fprintf(stderr, "  smrd op=%u sdst=%u sbase=%u imm=%u off=%#x\n",
                       (w >> 22) & 0x1F, (w >> 15) & 0x7F, (w >> 9) & 0x3F,
                       (w >> 8) & 1, w & 0xFF);
        } else if (in.enc == gcn::Enc::kMimg) {
          std::fprintf(stderr, "  mimg srsrc=%u\n",
                       ((in.raw[1] >> 16) & 0x1F) * 4);
        }
      }
      // Dump the WORLD VS's cbuffer reads: for each s_buffer_load, resolve the
      // V# from the VS user-data and print the 16 floats it actually reads (the
      // REAL matrix the VS uses), vs the heuristic cbuf above -- to find
      // whether the VS reads a different cbuffer/offset (the true MVP) than we
      // bind.
      if (d.vs_addr >= 0x1000000000ull && d.vs_addr < 0x20000000000ull) {
        auto vs_i =
            gcn::Decode(reinterpret_cast<const uint32_t*>(d.vs_addr), 4096);
        const uint32_t* vud = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
        std::fprintf(
            stderr, "  vs_ud=[%08x %08x %08x %08x %08x %08x %08x %08x]\n",
            vud[0], vud[1], vud[2], vud[3], vud[4], vud[5], vud[6], vud[7]);
        int shown = 0;
        for (auto& in : vs_i) {
          if (in.enc != gcn::Enc::kSmrd || shown >= 6)
            continue;
          uint32_t w = in.raw[0], op = (w >> 22) & 0x1F,
                   sbase = (w >> 9) & 0x3F;
          bool imm = (w >> 8) & 1;
          uint32_t off = w & 0xFF;
          shown++;
          uint32_t b2 = sbase * 2;
          uint32_t boff = imm ? off * 4 : 0;
          // op<8 = s_load (64-bit pointer in ud[b2..b2+1]); op>=8 =
          // s_buffer_load (V# in ud[b2..b2+3], base in low 44 bits). Resolve +
          // dump the 16 floats it reads (a 2nd matrix here would be the missing
          // view transform).
          uint64_t base = 0;
          if (b2 + 1 < 16) {
            if (op < 0x08)
              base = ((uint64_t)vud[b2 + 1] << 32 | vud[b2]);
            else
              base = ((uint64_t)(vud[b2 + 1] & 0xFFF) << 32 | vud[b2]);
          }
          std::fprintf(stderr,
                       "  vs_smrd op=%u %s sbase=%u(ud%u) off=%#x -> base=%#lx",
                       op, op < 0x08 ? "sload" : "sbufload", sbase, b2, boff,
                       (unsigned long)base);
          if (base >= 0x1000000ull && base < 0x20000000000ull) {
            const float* m = reinterpret_cast<const float*>(base + boff);
            std::fprintf(stderr,
                         " mtx=[%.2f %.2f %.2f %.2f / %.2f %.2f %.2f %.2f / "
                         "%.2f %.2f %.2f %.2f / %.2f %.2f %.2f %.2f]",
                         m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
                         m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
          }
          std::fprintf(stderr, "\n");
        }
      }
    }
    // DELTA_GPU_SKIPSTALE: drop draws that sample a very wide (>=2048) buffer,
    // used to hide a title's stale full-screen video-buffer blit (Doom64's
    // undecoded 4K menu bg = garbage) so the menu items drawn on top become
    // readable -> lets the correct menu input be derived instead of guessed.
    if (kSkipStale && d.tex_base && d.tex_w >= 2048)
      ;  // skip the wide stale-buffer blit
    else if (d.vertex_data || (d.recomp && d.recomp->ok && d.num_vattrs == 0))
      rhi::Draw(rhi::DefaultRenderer(), d);
  }
  if (!kTrace)
    return;
  uint64_t cb = g_regs.CbColorBase(0);
  uint32_t cb_info = g_regs[mmCB_COLOR0_INFO];
  uint32_t cb_attrib = g_regs[mmCB_COLOR0_ATTRIB];
  uint64_t vs = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS);
  uint64_t ps = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  uint32_t prim = g_regs[mmVGT_PRIMITIVE_TYPE];
  uint32_t sc_tl = g_regs[mmPA_SC_SCREEN_SCISSOR_TL];
  uint32_t sc_br = g_regs[mmPA_SC_SCREEN_SCISSOR_BR];
  uint32_t indices = (count >= 1) ? body[0] : 0;
  std::fprintf(stderr,
               "[gpu] DRAW op=%#x prim=%u indices=%u | RT=%#lx info=%#x "
               "attrib=%#x scissor=[%u,%u..%u,%u] VS=%#lx PS=%#lx\n",
               op, prim, indices, (unsigned long)cb, cb_info, cb_attrib,
               sc_tl & 0xFFFF, sc_tl >> 16, sc_br & 0xFFFF, sc_br >> 16,
               (unsigned long)vs, (unsigned long)ps);

  // One-time: locate the embedded "OrbShdr" BinaryInfo in the VS/PS GCN code to
  // confirm the recompiler can find shader length+hash. Layout: if code[0] ==
  // 0xBEEB03FF the info is at code + (code[1]+1)*2 dwords; else scan for the
  // 7-byte signature {'O','r','b','S','h','d','r'}.
  static bool shader_probed = false;
  if (!shader_probed && vs && ps) {
    shader_probed = true;
    auto probe = [](const char* tag, uint64_t addr) {
      auto* code = reinterpret_cast<const uint32_t*>(addr);
      const uint8_t* info = nullptr;
      if (code[0] == 0xBEEB03FFu)
        info = reinterpret_cast<const uint8_t*>(code + (code[1] + 1) * 2);
      else {
        auto* b = reinterpret_cast<const uint8_t*>(code);
        for (int k = 0; k < 0x4000; k++)
          if (std::memcmp(b + k, "OrbShdr", 7) == 0) {
            info = b + k;
            break;
          }
      }
      if (info) {
        uint32_t len_field;
        std::memcpy(&len_field, info + 8, 4);
        uint64_t hash;
        std::memcpy(&hash, info + 0xC, 8);  // approx offsets
        std::fprintf(
            stderr, "[gpu]   %s shader @%#lx OrbShdr len=%u hash=%#lx\n", tag,
            (unsigned long)addr, len_field & 0xFFFFFF, (unsigned long)hash);
      } else {
        std::fprintf(stderr,
                     "[gpu]   %s shader @%#lx: no OrbShdr (code0=%#x)\n", tag,
                     (unsigned long)addr, code[0]);
      }
    };
    probe("VS", vs);
    probe("PS", ps);

    // Dump the user-data SGPRs and decode candidate V#/T#/S# descriptors. A
    // dword pair that forms a plausible guest pointer (0x10_0000_0000.. range)
    // is likely a descriptor or a pointer to a descriptor table.
    auto dump_ud = [](const char* tag, const uint32_t* ud) {
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
          std::fprintf(
              stderr,
              "[gpu]     sgpr[%d..]: ptr=%#lx stride=%u nrec=%u fmt=%#x\n", k,
              (unsigned long)base, stride, nrec, ud[k + 3]);
          // A small vertex buffer (a quad): dump it as floats to learn the
          // layout.
          if (stride && stride <= 64 && nrec && nrec <= 8) {
            auto* f = reinterpret_cast<const float*>(base);
            auto* u = reinterpret_cast<const uint32_t*>(base);
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
    dump_ud("VS", &g_regs[mmSPI_SHADER_USER_DATA_VS_0]);
    dump_ud("PS", &g_regs[mmSPI_SHADER_USER_DATA_PS_0]);

    // Follow the descriptor-table pointers in the user-data SGPRs and decode
    // the V#/T#/S# sharps inside. A V# (4 dwords): base48, stride, num_records.
    // A T# (8 dwords): base + width/height. This is where the real vertex
    // buffer and texture atlas live for the quad draws.
    auto dump_table = [](const char* tag, uint64_t ptr) {
      if (ptr < 0x1000000000ull || ptr >= 0x20000000000ull)
        return;
      auto* t = reinterpret_cast<const uint32_t*>(ptr);
      std::fprintf(stderr, "[gpu]   table %s @%#lx:\n", tag,
                   (unsigned long)ptr);
      for (int k = 0; k < 32; k += 4) {
        uint64_t b = ((uint64_t)(t[k + 1] & 0xFFFF) << 32) | t[k];
        uint32_t stride = (t[k + 1] >> 16) & 0x3FFF;
        // V# heuristic
        if (b >= 0x1000000000ull && b < 0x20000000000ull && stride &&
            stride <= 256)
          std::fprintf(
              stderr,
              "[gpu]     +%02x V#? base=%#lx stride=%u nrec=%u dfmt=%#x\n",
              k * 4, (unsigned long)b, stride, t[k + 2], t[k + 3]);
        // T# heuristic: dword2 has width-1[0:13], height-1[14:27]
        uint64_t tb = ((uint64_t)(t[k + 1] & 0xFFFFFF) << 32) | t[k];
        uint32_t w = (t[k + 2] & 0x3FFF) + 1,
                 h = ((t[k + 2] >> 14) & 0x3FFF) + 1;
        if (tb >= 0x1000000000ull && tb < 0x20000000000ull && w > 4 &&
            w <= 8192 && h > 4 && h <= 8192)
          std::fprintf(stderr, "[gpu]     +%02x T#? base=%#lx %ux%u dfmt=%#x\n",
                       k * 4, (unsigned long)tb, w, h, (t[k + 1] >> 20) & 0x3F);
      }
    };
    const uint32_t* vud = &g_regs[mmSPI_SHADER_USER_DATA_VS_0];
    dump_table("VS.sgpr0", ((uint64_t)(vud[1] & 0xFFFF) << 32) | vud[0]);
    dump_table("VS.sgpr2", ((uint64_t)(vud[3] & 0xFFFF) << 32) | vud[2]);

    // Disassemble the shaders to validate the GCN decoder and reveal the
    // vertex-fetch / resource-load pattern. The fetch shader (sgpr0 ptr, just
    // past the VS code) does the s_load(V# table) + buffer_load(attributes).
    gcn::Disassemble(reinterpret_cast<const uint32_t*>(vs), 512, "VS");
    gcn::Disassemble(reinterpret_cast<const uint32_t*>(ps), 512, "PS");
    uint64_t fetch = ((uint64_t)(vud[1] & 0xFFFF) << 32) | vud[0];
    if (fetch >= 0x1000000000ull && fetch < 0x20000000000ull) {
      gcn::Disassemble(reinterpret_cast<const uint32_t*>(fetch), 128,
                       "VS.fetch");
      // Recover the actual vertex-attribute buffers and dump the first
      // vertices.
      auto vbs = gcn::TrackVertexBuffers(*gcn::CachedProgram(fetch, 64), vud);
      for (size_t bi = 0; bi < vbs.size(); bi++) {
        auto& v = vbs[bi];
        std::fprintf(stderr, "[gpu]   VB%zu base=%#lx stride=%u nrec=%u\n", bi,
                     (unsigned long)v.base, v.stride, v.num_records);
        auto* f = reinterpret_cast<const float*>(v.base);
        for (uint32_t r = 0; r < v.num_records && r < 6; r++) {
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

void SetPs4NeoMode(bool enabled) {
  std::lock_guard<std::mutex> lock(g_mtx);
  gcn::SetDefaultIsaMode(enabled ? gcn::IsaMode::kNeo : gcn::IsaMode::kBase);
}

// Called by the Gnm HLE on submit-and-flip: finish the frame and present the
// render target the flip displays.
void EndFrame(uint64_t scanout_base) {
  std::lock_guard<std::mutex> lk(g_mtx);
  // New frame -> shader code may have been rewritten; let CachedProgram
  // revalidate each address once next frame instead of once per draw.
  gcn::NextProgramCacheGeneration();
  if (g_frame_active && rhi::DefaultRenderer().available()) {
    rhi::EndFrame(rhi::DefaultRenderer(), scanout_base);
    g_frame_active = false;
    g_presented_frames++;
  }
}

// Constant Engine RAM: on-chip scratch the CE fills and dumps to memory as the
// shaders' constant buffers. Liverpool CE RAM is 48 KiB. Every access is
// bounds- checked so a malformed packet can never write outside it or outside
// guest memory.
uint8_t g_ce_ram[48 * 1024];
inline bool CcbGuestRange(uint64_t a, uint64_t bytes) {
  return bytes > 0 && a >= 0x1000000000ull && a + bytes <= 0x20000000000ull;
}

void SubmitCcb(const void* ccb, uint32_t size_bytes) {
  if (!ccb || size_bytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  const uint32_t* p = static_cast<const uint32_t*>(ccb);
  uint32_t words = size_bytes / 4, i = 0;
  static uint32_t hist[256] = {};
  static int hist_dumps = 0;
  static uint64_t n_ccb = 0;
  if (kCcbHist && n_ccb == 0)
    std::fprintf(stderr, "[ccb] first ccb: %u bytes (%u words)\n", size_bytes,
                 words);
  n_ccb++;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = Pm4TypeOf(hdr);
    if (type == Pm4Type::kType3) {
      uint32_t op = Pm4Opcode(hdr), count = Pm4Count(hdr);
      const uint32_t* body = &p[i + 1];
      if (i + 1 + count > words)
        break;
      hist[op & 0xFF]++;
      if (kCeOn) {
        switch (op) {
          case IT_WRITE_CONST_RAM: {  // body[0]=byte offset; body[1..]=data
                                      // dwords
            uint32_t off = body[0] & 0xFFFF;
            uint32_t n = count > 1 ? count - 1 : 0;
            if ((uint64_t)off + (uint64_t)n * 4 <= sizeof(g_ce_ram))
              std::memcpy(g_ce_ram + off, &body[1], (size_t)n * 4);
            break;
          }
          case IT_LOAD_CONST_RAM: {  // addrLo, addrHi, num_dwords, byte offset
            if (count >= 4) {
              uint64_t addr =
                  (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
              uint32_t num = body[2] & 0x7FFF, off = body[3] & 0xFFFF;
              if (CcbGuestRange(addr, (uint64_t)num * 4) &&
                  (uint64_t)off + (uint64_t)num * 4 <= sizeof(g_ce_ram))
                std::memcpy(g_ce_ram + off, reinterpret_cast<const void*>(addr),
                            (size_t)num * 4);
            }
            break;
          }
          case IT_DUMP_CONST_RAM:
          case IT_DUMP_CONST_RAM_OFFSET: {  // byte offset, num_dwords, addrLo,
                                            // addrHi
            if (count >= 4) {
              uint32_t off = body[0] & 0xFFFF, num = body[1] & 0x7FFF;
              uint64_t addr =
                  (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) | body[2];
              if (CcbGuestRange(addr, (uint64_t)num * 4) &&
                  (uint64_t)off + (uint64_t)num * 4 <= sizeof(g_ce_ram))
                std::memcpy(reinterpret_cast<void*>(addr), g_ce_ram + off,
                            (size_t)num * 4);
            }
            break;
          }
          default:
            break;
        }
      }
      i += 1 + count;
    } else if (type == Pm4Type::kType2 || hdr == 0) {
      i += 1;
    } else if (type == Pm4Type::kType0) {
      i += 1 + Pm4Count(hdr);
    } else {
      break;  // type-1 desync
    }
  }
  if (kCcbHist && hist_dumps < 3 && n_ccb >= 50) {
    hist_dumps++;
    std::fprintf(
        stderr, "[ccb] opcode histogram (after %lu ccbs, this one %u words):\n",
        (unsigned long)n_ccb, words);
    for (int o = 0; o < 256; o++)
      if (hist[o])
        std::fprintf(stderr, "[ccb]   op %#04x x%u\n", o, hist[o]);
  }
}

// IT_DISPATCH_DIRECT: a GPU compute dispatch. body = [dim_x, dim_y, dim_z,
// dispatch_initiator] (workgroup counts). The CS program addr, workgroup size,
// resource/RSRC and user-data come from the COMPUTE_* SH registers set before
// it. Doom64 builds its level texture atlases with these (the T# the 3D world
// samples is written by a CS), so without executing them the atlases stay
// zero/black.
void HandleDispatch(const uint32_t* body, uint32_t count) {
  uint32_t dim_x = count >= 1 ? body[0] : 0;
  uint32_t dim_y = count >= 2 ? body[1] : 0;
  uint32_t dim_z = count >= 3 ? body[2] : 0;
  uint64_t cs_addr =
      (static_cast<uint64_t>(g_regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
       g_regs[mmCOMPUTE_PGM_LO])
      << 8;
  uint32_t tgx = g_regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF;
  uint32_t tgy = g_regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF;
  uint32_t tgz = g_regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF;
  uint32_t rsrc2hi = g_regs[mmCOMPUTE_PGM_RSRC2];
  uint32_t user_sgpr = (rsrc2hi >> 1) & 0x1F;   // num_user_regs (bits 37:33)
  uint32_t tgid_enable = (rsrc2hi >> 7) & 0x7;  // tgid_enable (bits 41:39)
  uint32_t lds_dwords = (rsrc2hi >> 15) & 0x1FF;

  static std::unordered_set<uint64_t> dumped_cs;
  if (kCsDump && dumped_cs.size() < 32 && cs_addr >= 0x1000000000ull &&
      cs_addr < 0x20000000000ull && dumped_cs.insert(cs_addr).second) {
    const uint32_t* ud = &g_regs[mmCOMPUTE_USER_DATA_0];
    std::fprintf(stderr,
                 "[cs] addr=%#lx groups=[%u %u %u] tg=[%u %u %u] usgpr=%u "
                 "tgiden=%u lds=%u\n",
                 (unsigned long)cs_addr, dim_x, dim_y, dim_z, tgx, tgy, tgz,
                 user_sgpr, tgid_enable, lds_dwords);
    std::fprintf(stderr, "[cs]   user_data:");
    for (int k = 0; k < 16; k++)
      std::fprintf(stderr, " %08x", ud[k]);
    std::fprintf(stderr, "\n");
    gcn::Disassemble(reinterpret_cast<const uint32_t*>(cs_addr), 1024, "cs");
  }

  if (cs_addr < 0x1000000000ull || cs_addr >= 0x20000000000ull || !tgx ||
      !tgy || !dim_x || !dim_y)
    return;
  const uint32_t* ud = &g_regs[mmCOMPUTE_USER_DATA_0];

  if (kNoCs || !rhi::DefaultRenderer().available())
    return;

  // Recompile the CS to a Vulkan compute pipeline, resolve the guest memory
  // ranges its descriptors point at from the live user data, and run it on the
  // GPU. Doom64 builds its world-texture atlases this way. An unsupported CS
  // fails the recompile and is skipped (loud, not silently corrupting memory).
  // The cache key covers everything baked into the module (workgroup shape +
  // RSRC2 state), not just the code address: the same CS can legally be
  // re-dispatched with a different workgroup size.
  const ComputeShaderKey cs_key{
      cs_addr,    tgx,
      tgy,        tgz,
      user_sgpr,  tgid_enable,
      lds_dwords, gcn::DefaultIsaMode() == gcn::IsaMode::kNeo};
  static std::unordered_map<ComputeShaderKey, gcn::RecompiledCs,
                            ComputeShaderKeyHash>
      cs_cache;
  auto cit = cs_cache.find(cs_key);
  if (cit == cs_cache.end()) {
    cit = cs_cache
              .emplace(cs_key,
                       gcn::RecompileCompute(
                           reinterpret_cast<const uint32_t*>(cs_addr), tgx, tgy,
                           tgz, user_sgpr, tgid_enable, lds_dwords))
              .first;
    static int reported = 0;
    if (!cit->second.ok && reported < 8) {
      reported++;
      std::fprintf(
          stderr, "[csgpu] unsupported CS @%#lx groups=[%u %u %u] -- skipped\n",
          (unsigned long)cs_addr, dim_x, dim_y, dim_z);
    }
  }
  gcn::RecompiledCs& rc = cit->second;
  if (!rc.ok)
    return;

  // Resolve each planned resource's live base/size from the descriptor in user
  // data.
  auto guest_range = [](uint64_t a, uint64_t n) {
    constexpr uint64_t lo = 0x1000000000ull, hi = 0x20000000000ull;
    return n && a >= lo && a < hi && n <= hi - a &&
           utl::isMemoryRangeMapped(reinterpret_cast<const void*>(a), n);
  };
  constexpr uint64_t kMaxRes =
      256ull * 1024 * 1024;  // sanity cap per storage buffer
  constexpr uint64_t kMaxUnboundedBuffer = 16ull * 1024 * 1024;
  rhi::ComputeInfo ci;
  ci.cs_addr = cs_addr;
  ci.groups[0] = dim_x;
  ci.groups[1] = dim_y;
  ci.groups[2] = dim_z;
  ci.recomp = &rc;
  for (int i = 0; i < 16; i++)
    ci.user_data[i] = ud[i];
  const auto cs_program = gcn::CachedProgram(cs_addr, 4096);
  auto resolved = gcn::ResolveCsResources(*cs_program, rc, ud);
  // Descriptor chains live in guest memory, which an earlier dispatch may have
  // written — and writebacks are lazy. If any binding failed to resolve, land
  // pending compute writes in guest memory and re-resolve once before falling
  // back to a dummy (the fallback zeroes a real input and silently corrupts
  // whatever pipeline this CS belongs to).
  {
    bool any_unresolved = false;
    for (auto& r : rc.resources)
      if (r.binding >= resolved.size() || !resolved[r.binding].valid)
        any_unresolved = true;
    if (any_unresolved) {
      // Best-effort: a range that could not be written back leaves its chain
      // stale, but the dummy fallback below still beats dropping the dispatch.
      // Only a dead renderer (device fault) makes retrying pointless.
      if (!rhi::FlushCsWrites(rhi::DefaultRenderer()) &&
          !rhi::DefaultRenderer().available())
        return;
      resolved = gcn::ResolveCsResources(*cs_program, rc, ud);
    }
  }
  static std::unordered_set<uint64_t> traced_cs_resources;
  const bool trace_cs_resources = kCsResTrace &&
                                  traced_cs_resources.size() < 64 &&
                                  traced_cs_resources.insert(cs_addr).second;
  bool res_ok = true;
  for (auto& r : rc.resources) {
    const bool resource_resolved =
        r.binding < resolved.size() && resolved[r.binding].valid;
    if (!resource_resolved) {
      if (trace_cs_resources) {
        std::fprintf(stderr,
                     "[csres] cs=%#lx bind=%u kind=%u s%u pc=%#x unresolved; "
                     "using dummy\n",
                     (unsigned long)cs_addr, r.binding, r.kind, r.base_sgpr,
                     r.use_pc);
      }
      // DELTA_GPU_EUDFAIL: raw code dump of an unresolvable CS (once) so the
      // descriptor chain can be decoded offline against the eudfail trace.
      static std::unordered_set<uint64_t> dumped;
      if (kEudFail && dumped.insert(cs_addr).second) {
        const uint32_t* c = reinterpret_cast<const uint32_t*>(cs_addr);
        std::fprintf(stderr, "[csdump] cs=%#lx first 0x360 dwords:\n",
                     (unsigned long)cs_addr);
        for (uint32_t k = 0; k < 0x360; k += 8)
          std::fprintf(
              stderr,
              "[csdump] %04x: %08x %08x %08x %08x %08x %08x %08x %08x\n", k,
              c[k], c[k + 1], c[k + 2], c[k + 3], c[k + 4], c[k + 5], c[k + 6],
              c[k + 7]);
      }
    }
    static constexpr uint32_t kNullDescriptor[8] = {};
    const uint32_t* descriptor =
        resource_resolved ? resolved[r.binding].descriptor : kNullDescriptor;
    uint64_t base = 0, size = 0, guest_size = 0;
    gcn::TImage image;
    bool image_staging = false;
    bool zero_fill = false;
    uint32_t elem_bytes = 4;
    uint32_t stage_elem_bytes = 4;
    if (r.kind == 1) {
      gcn::TImage t = gcn::DecodeTImage(descriptor);
      gcn::TextureLayout32 layout;
      const bool rgba8 = t.dfmt == 10 && (t.nfmt == 0 || t.nfmt == 4);
      const bool r32 =
          t.dfmt == 4 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
      const bool rg16f = t.dfmt == 5 && t.nfmt == 7;
      const bool r16f = t.dfmt == 2 && t.nfmt == 7;
      const bool rg8 = t.dfmt == 3 && t.nfmt == 0;
      const bool rgba16f = t.dfmt == 12 && t.nfmt == 7;
      const bool r11g11b10f = t.dfmt == 6 && t.nfmt == 7;
      const bool supported_type = t.type == 9 || t.type == 13;
      const bool supported_format =
          rgba8 || r32 || rg16f || r16f || rg8 || rgba16f || r11g11b10f;
      elem_bytes = rgba16f ? 8 : (r16f || rg8) ? 2 : 4;
      stage_elem_bytes = r11g11b10f ? 16 : std::max(elem_bytes, 4u);
      if (!supported_type || !supported_format) {
        // Invalid/null T# values can be present on paths the guest shader does
        // not take. The translator guards these descriptors and returns zero.
        zero_fill = true;
        size = 16;
      } else if (!t.valid ||
                 !gcn::BuildTextureLayout32(
                     layout, t.width, t.height, t.pitch, t.layers, t.mip_levels,
                     t.tiling_idx, t.pow2_pad, elem_bytes)) {
        if (trace_cs_resources)
          std::fprintf(stderr,
                       "[csres] cs=%#lx bind=%u unsupported image valid=%d "
                       "base=%#lx type=%u dfmt=%u nfmt=%u tiling=%u %ux%u "
                       "pitch=%u layers=%u words=[%08x %08x %08x %08x %08x "
                       "%08x %08x %08x]\n",
                       (unsigned long)cs_addr, r.binding, t.valid ? 1 : 0,
                       (unsigned long)t.base, t.type, t.dfmt, t.nfmt,
                       t.tiling_idx, t.width, t.height, t.pitch, t.layers,
                       descriptor[0], descriptor[1], descriptor[2],
                       descriptor[3], descriptor[4], descriptor[5],
                       descriptor[6], descriptor[7]);
        res_ok = false;
        break;
      } else {
        base = t.base;
        guest_size = layout.size;
        image_staging = !gcn::TilingIsLinear(t.tiling_idx) ||
                        elem_bytes != stage_elem_bytes;
        if (image_staging) {
          gcn::TextureLayout32 linear;
          const uint32_t stage_tiling = t.tiling_idx == 31 ? 31 : 8;
          if (!gcn::BuildTextureLayout32(linear, t.width, t.height, t.pitch,
                                         t.layers, t.mip_levels, stage_tiling,
                                         t.pow2_pad, stage_elem_bytes)) {
            res_ok = false;
            break;
          }
          size = linear.size;
          image = t;
        } else {
          size = guest_size;
        }
      }
    } else if (r.kind ==
               2) {  // scalar-load pointer into an SRT/descriptor table
      base =
          (static_cast<uint64_t>(descriptor[1] & 0xFFFF) << 32) | descriptor[0];
      size = r.min_bytes;
      if (!guest_range(base, 1)) {
        zero_fill = true;
        base = 0;
        size = std::max<uint64_t>(size, 16);
      }
    } else {  // buffer V#: stride*num_records, else the min hint
      gcn::VBuffer v = gcn::DecodeVBuffer(descriptor);
      base = v.base;
      size = v.stride ? (uint64_t)v.stride * v.num_records : v.num_records;
      if (size < r.min_bytes)
        size = r.min_bytes;
      if (!guest_range(base, 1)) {
        zero_fill = true;
        base = 0;
        size = 16;
      } else if (size > kMaxRes) {
        const uint64_t declared_size = size;
        size = utl::mappedMemoryPrefix(reinterpret_cast<const void*>(base),
                                       kMaxUnboundedBuffer);
        if (trace_cs_resources)
          std::fprintf(stderr,
                       "[csres] cs=%#lx bind=%u windowed buffer declared=%#lx "
                       "mapped=%#lx\n",
                       (unsigned long)cs_addr, r.binding,
                       (unsigned long)declared_size, (unsigned long)size);
      }
    }
    if (!guest_size && !zero_fill)
      guest_size = size;
    if (trace_cs_resources) {
      std::fprintf(stderr,
                   "[csres] cs=%#lx bind=%u kind=%u s%u pc=%#x base=%#lx "
                   "size=%#lx written=%d\n",
                   (unsigned long)cs_addr, r.binding, r.kind, r.base_sgpr,
                   r.use_pc, (unsigned long)base, (unsigned long)size,
                   r.written ? 1 : 0);
    }
    if (size < r.min_bytes || size > kMaxRes || guest_size > kMaxRes ||
        (!zero_fill && !guest_range(base, guest_size)) ||
        ci.num_res >= gcn::kMaxCsResources) {
      if (trace_cs_resources)
        std::fprintf(
            stderr,
            "[csres] cs=%#lx bind=%u invalid range base=%#lx size=%#lx\n",
            (unsigned long)cs_addr, r.binding, (unsigned long)base,
            (unsigned long)guest_size);
      res_ok = false;
      break;
    }
    rhi::ComputeInfo::Res& out = ci.res[ci.num_res];
    out.base = base;
    out.size = size;
    out.guest_size = guest_size;
    out.binding = r.binding;
    out.shader_writes = r.written;
    out.written = r.written && !zero_fill;
    out.zero_fill = zero_fill;
    out.image_staging = image_staging;
    if (image_staging) {
      out.width = image.width;
      out.height = image.height;
      out.pitch = image.pitch;
      out.layers = image.layers;
      out.mip_levels = image.mip_levels;
      out.tiling_idx = image.tiling_idx;
      out.elem_bytes = elem_bytes;
      out.stage_elem_bytes = stage_elem_bytes;
      out.dfmt = image.dfmt;
      out.pow2_pad = image.pow2_pad;
    }
    ci.num_res++;
  }
  if (!res_ok || !ci.num_res)
    return;
  const bool dispatched = rhi::Dispatch(rhi::DefaultRenderer(), ci);
  if (trace_cs_resources)
    std::fprintf(stderr, "[csres] cs=%#lx dispatch %s (%u resources)\n",
                 (unsigned long)cs_addr, dispatched ? "executed" : "failed",
                 ci.num_res);
}

void SubmitDcb(const void* dcb, uint32_t size_bytes) {
  if (!dcb || size_bytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!g_vk_tried) {
    g_vk_tried = true;
    rhi::Init(rhi::DefaultRenderer());
  }
  auto* p = static_cast<const uint32_t*>(dcb);
  uint32_t words = size_bytes / 4;
  uint64_t sn = g_total_submits.fetch_add(1) + 1;
  if (kTrace && (sn <= 8 || sn % 256 == 0))
    std::fprintf(stderr, "[gpu] submit #%lu size=%u draws-so-far=%lu\n",
                 (unsigned long)sn, size_bytes,
                 (unsigned long)g_total_draws.load());
  // Dump the full packet walk of the first large (real rendering) command
  // buffer so we can see its opcodes / find the draw.
  static bool dumped_big = false;
  bool dump_this = kTrace && !dumped_big && size_bytes > 4000;
  if (dump_this) {
    dumped_big = true;
    std::fprintf(stderr, "[gpu] === big dcb walk (size=%u) ===\n", size_bytes);
  }
  if (kTrace && g_dcb_seen < 6)
    std::fprintf(stderr,
                 "[gpu] SubmitDcb dcb=%p size_bytes=%u words=%u hdr0=%#x\n",
                 dcb, size_bytes, words, p[0]);
  uint32_t i = 0;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = Pm4TypeOf(hdr);
    if (type == Pm4Type::kType3) {
      uint32_t op = Pm4Opcode(hdr);
      uint32_t count = Pm4Count(hdr);  // body dword count
      const uint32_t* body = &p[i + 1];
      g_op_hist[op & 0xFF]++;
      if (kTrace && dump_this)
        std::fprintf(stderr, "[gpu]   @%-5u T3 op=%#04x count=%u\n", i, op,
                     count);
      if (i + 1 + count > words)
        break;  // truncated / desync
      switch (op) {
        case IT_DISPATCH_DIRECT:
          HandleDispatch(body, count);
          break;
        case IT_SET_CONTEXT_REG:
          SetRegs(kContextRegBase, body, count);
          break;
        case IT_SET_SH_REG:
        case IT_SET_SH_REG_INDEX:
          SetRegs(kShRegBase, body, count);
          break;
        case IT_SET_UCONFIG_REG:
          SetRegs(kUConfigRegBase, body, count);
          break;
        case IT_SET_CONFIG_REG:
          SetRegs(kConfigRegBase, body, count);
          break;
        case IT_INDEX_TYPE:  // VGT_DMA_INDEX_TYPE: bits[1:0] 0=16-bit 1=32-bit
          if (count >= 1)
            g_index_type = body[0] & 0x3;
          break;
        case IT_INDEX_BASE:  // index buffer base (byte address) lo/hi
          if (count >= 2)
            g_index_base =
                (static_cast<uint64_t>(body[1] & 0xFF) << 32) | body[0];
          break;
        case IT_SET_BASE:
          // base_index 1 = DRAW_INDIRECT_BASE: where the indirect draws below
          // read their argument structs from. body: baseIndex, addrLo, addrHi.
          if (count >= 3 && (body[0] & 0xF) == 1)
            g_indirect_base = (static_cast<uint64_t>(body[2] & 0xFF) << 32) |
                              (body[1] & ~0x3u);
          break;
        case IT_NUM_INSTANCES:  // instance count for the following draw(s)
          g_num_instances = (count >= 1 && body[0]) ? body[0] : 1;
          break;
        case IT_DMA_DATA:  // CP DMA. body: ctrl, srcLo/Hi, dstLo/Hi,
                           // command(byteCount).
          // Actually PERFORM the memory->memory copy (it was a no-op). Doom64
          // uploads its level texture atlases to GPU memory via CP DMA, so
          // without this the T# addresses stay zero and the 3D world samples
          // blank (black) textures. ctrl word: SRC_SEL[30:29], DST_SEL[21:20];
          // sel 0/3 = memory address, 2 = immediate data (a fill, not a copy)
          // -- only copy true mem->mem.
          if (count >= 6) {
            uint32_t ctrl = body[0];
            uint32_t src_sel = (ctrl >> 29) & 0x3;
            uint32_t dst_sel = (ctrl >> 20) & 0x3;
            uint64_t src =
                (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
            uint64_t dst =
                (static_cast<uint64_t>(body[4] & 0xFFFF) << 32) | body[3];
            uint32_t bytes = body[5] & 0x1FFFFF;
            bool src_mem = (src_sel == 0 || src_sel == 3);
            bool dst_mem = (dst_sel == 0 || dst_sel == 3);
            // Only copy between REAL guest memory: the sel bits report "memory"
            // even for GDS/register targets (e.g. dst=0x3022c), which aren't
            // mapped in our address space and segfault. Every real guest
            // allocation (heap/video/ garlic) sits far above 16 MiB, so that
            // floor cleanly excludes the on-chip GDS/low targets while keeping
            // texture/buffer uploads.
            auto mem_ok = [](uint64_t a) {
              return a >= 0x1000000ull && a < 0x20000000000ull;
            };
            bool copied = false;
            if (!kNoCopy && src_mem && dst_mem && bytes &&
                bytes <= 0x1000000u && src != dst && mem_ok(src) &&
                mem_ok(src + bytes) && mem_ok(dst) && mem_ok(dst + bytes)) {
              // src may be CS-written; land pending writes first. Copy even
              // if the flush fails: a possibly-stale source range beats
              // silently dropping the whole guest->guest copy.
              rhi::FlushCsWrites(rhi::DefaultRenderer());
              std::memcpy(reinterpret_cast<void*>(dst),
                          reinterpret_cast<const void*>(src), bytes);
              copied = true;
            }
            if (kGpuDmatrace) {
              static int dmn = 0;
              // Shader-prefetch packets target a low register address and
              // otherwise consume the trace cap before uploads begin. Keep
              // only real guest destinations and immediate fills, and report
              // what was actually done.
              if (dmn < 200 && (src_sel == 2 || mem_ok(dst))) {
                dmn++;
                std::fprintf(stderr,
                             "[dma] ctrl=%#x cmd=%#x srcSel=%u dstSel=%u "
                             "src/data=%#lx dst=%#lx bytes=%u %s\n",
                             ctrl, body[5] & ~0x1fffffu, src_sel, dst_sel,
                             (unsigned long)src, (unsigned long)dst, bytes,
                             copied ? "copied" : "ignored");
              }
            }
          }
          break;
        case IT_WRITE_DATA: {  // body: control, dstLo, dstHi, data...
          // Gnm writes its 32/64-bit submit/flip fence labels with WRITE_DATA.
          // The control field's dst_sel encoding varies (the flip-label packet
          // built by sceGnmInsertFlip uses control=5, not the [11:8]=memory
          // form), so don't gate on dst_sel: a memory write resolves to a real
          // guest label address, while a register write (dst_sel=0) yields a
          // tiny offset that LabelAddrOk rejects. body[1]=dstLo, body[2]=dstHi,
          // body[3..]=data.
          if (count >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                            (body[1] & ~0x3u);
            uint32_t ndw = count - 3;
            if (LabelAddrOk(addr) && LabelAddrOk(addr + (uint64_t)ndw * 4))
              std::memcpy(reinterpret_cast<void*>(addr), &body[3],
                          (size_t)ndw * 4);
            if (kEopTrace)
              std::fprintf(stderr, "[eop] WRITE_DATA dst=%#lx ndw=%u v0=%#x\n",
                           (unsigned long)addr, ndw, ndw ? body[3] : 0);
          }
          break;
        }
        case IT_EVENT_WRITE_EOP: {  // body: eventCtrl, addrLo, addrHi+sel,
                                    // dataLo, dataHi
          if (count >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                            (body[1] & ~0x3u);
            uint32_t data_sel =
                (body[2] >> 29) & 0x7;  // 1=32b, 2=64b, 3/4=clock
            uint64_t val =
                static_cast<uint64_t>(body[3]) |
                (static_cast<uint64_t>(count >= 5 ? body[4] : 0) << 32);
            if (data_sel == 1)
              WriteLabel(addr, val, false);
            else if (data_sel == 2)
              WriteLabel(addr, val, true);
            else if (data_sel >= 3)
              WriteLabel(addr, GpuClockTs(), true);
            if (kEopTrace)
              std::fprintf(stderr, "[eop] EOP addr=%#lx sel=%u val=%#lx\n",
                           (unsigned long)addr, data_sel, (unsigned long)val);
          }
          break;
        }
        case IT_RELEASE_MEM: {  // body: eventCtrl, selBits, addrLo, addrHi,
                                // dataLo, dataHi
          if (count >= 5) {
            uint32_t data_sel =
                (body[1] >> 29) & 0x7;  // 1=32b, 2=64b, 3/4=clock
            uint64_t addr = (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) |
                            (body[2] & ~0x3u);
            uint64_t val =
                static_cast<uint64_t>(body[4]) |
                (static_cast<uint64_t>(count >= 6 ? body[5] : 0) << 32);
            if (data_sel == 1)
              WriteLabel(addr, val, false);
            else if (data_sel == 2)
              WriteLabel(addr, val, true);
            else if (data_sel >= 3)
              WriteLabel(addr, GpuClockTs(), true);
            if (kEopTrace)
              std::fprintf(stderr,
                           "[eop] RELEASE_MEM addr=%#lx sel=%u val=%#lx\n",
                           (unsigned long)addr, data_sel, (unsigned long)val);
          }
          break;
        }
        case IT_EVENT_WRITE_EOS: {  // body: eventCtrl, addrLo, addrHi+cmd, data
          if (count >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                            (body[1] & ~0x3u);
            WriteLabel(addr, body[3], false);
            if (kEopTrace)
              std::fprintf(stderr, "[eop] EOS addr=%#lx val=%#x\n",
                           (unsigned long)addr, body[3]);
          }
          break;
        }
        default:
          if (IsDraw(op)) {
            g_total_draws.fetch_add(1);
            HandleDraw(op, body, count);
          }
          break;
      }
      i += 1 + count;
    } else if (type == Pm4Type::kType2 || hdr == 0) {
      // Single-dword filler: type-2 NOPs and zero-dword alignment padding that
      // Gnm sprinkles between packets. Skip and keep walking (these are NOT the
      // end of the buffer (real packets resume after the padding).
      i += 1;
    } else if (type == Pm4Type::kType0) {
      // Type-0 writes a run of consecutive registers (base in hdr[15:0], count
      // in hdr[29:16]+1) directly into the register file. The old walker
      // treated this as a desync and STOPPED -- dropping every later draw (the
      // room floor) in any command buffer that used type-0. Apply the register
      // writes and skip the body.
      uint32_t cnt = Pm4Count(hdr);      // body dword count
      uint32_t base = Pm4Type0Reg(hdr);  // absolute register dword offset
      for (uint32_t k = 0; k < cnt && i + 1 + k < words; k++) {
        uint32_t idx = base + k;
        if (idx < kRegFileSize)
          g_regs[idx] = p[i + 1 + k];
        if (idx == mmCB_SHADER_MASK)
          g_shader_mask_writes++;
        else if (idx == mmCB_TARGET_MASK)
          g_target_mask_writes++;
      }
      i += 1 + cnt;
    } else {
      // A type-1 header is a genuine desync; stop.
      if (dump_this || kDesyncTrace)
        std::fprintf(stderr, "[gpu]   @%-5u/%u STOP type%u hdr=%#x\n", i, words,
                     (uint32_t)type, hdr);
      break;
    }
  }
  if (dump_this) {
    std::fprintf(stderr, "[gpu] === big dcb walk done: %u/%u words ===\n", i,
                 words);
    // Brute-scan the whole buffer for draw-opcode headers (in case the walker
    // desynced and missed a draw), and dump raw words around the stop point.
    int found = 0;
    for (uint32_t w = 0; w < words; w++) {
      uint32_t h = p[w];
      if ((h >> 30) == 3) {
        uint32_t o = (h >> 8) & 0xFF;
        if (o == 0x2D || o == 0x27 || o == 0x35 || o == 0x30 || o == 0x15) {
          std::fprintf(stderr, "[gpu]   SCAN found draw op=%#x @word %u\n", o,
                       w);
          if (++found > 8)
            break;
        }
      }
    }
    if (!found)
      std::fprintf(
          stderr, "[gpu]   SCAN: no draw opcode anywhere in %u words\n", words);
    std::fprintf(stderr, "[gpu]   raw[255..270]:");
    for (uint32_t w = 255; w < 271 && w < words; w++)
      std::fprintf(stderr, " %08x", p[w]);
    std::fprintf(stderr, "\n");
  }
  if (kTrace && g_dcb_seen < 4)
    std::fprintf(stderr, "[gpu] dcb done: walked %u/%u words\n", i, words);
  if (kTrace && ++g_dcb_seen <= 4)
    DumpHist();
  // Cumulative opcode histogram dumped once deep into gameplay
  // (DELTA_GPU_OPHIST): reveals any draw/dispatch opcode the title uses that
  // IsDraw() doesn't handle (a silently-skipped draw -- e.g. the non-tutorial
  // room floor).
  static bool op_hist_dumped = false;
  // Time-gate (default 100s) so the cumulative histogram includes the in-level
  // command stream (level-load compute/copies), not just the title.
  static const auto kOhStart = std::chrono::steady_clock::now();
  // ...and only once the walker has actually seen packets: the first buffer
  // after the gate is often empty, which used to latch the dump on nothing.
  if (kOpHist && !op_hist_dumped && i > 0 &&
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - kOhStart)
              .count() >= kOhAfter) {
    op_hist_dumped = true;
    DumpHist();
  }
}

}  // namespace gpu
