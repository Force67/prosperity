/*
 * PS4Delta : PS4 emulation and research project
 *
 * GPU command processor. See cmd_processor.h.
 */

#include "cmd_processor.h"
#include "pm4.h"
#include "liverpool.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace gpu {
namespace {

std::mutex g_mtx;
Regs g_regs;  // persistent register state across submits (Gnm relies on this)
const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;

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

// Issue the current register state as a draw. For now: log what we'd render so
// the decode path is verifiable; the Vulkan path attaches here.
void handleDraw(uint32_t op, const uint32_t *body, uint32_t count) {
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
}

}  // namespace

void submitDcb(const void *dcb, uint32_t sizeBytes) {
  if (!dcb || sizeBytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  auto *p = static_cast<const uint32_t *>(dcb);
  uint32_t words = sizeBytes / 4;
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
        if (g_dcbSeen == 0)
          std::fprintf(stderr, "[gpu]   @%-4u T3 op=%#04x count=%u\n", i, op, count);
      }
      if (i + 1 + count > words)
        break;  // truncated / desync
      switch (op) {
      case IT_SET_CONTEXT_REG: setRegs(kContextRegBase, body, count); break;
      case IT_SET_SH_REG:      setRegs(kShRegBase, body, count); break;
      case IT_SET_UCONFIG_REG: setRegs(kUConfigRegBase, body, count); break;
      case IT_SET_CONFIG_REG:  setRegs(kConfigRegBase, body, count); break;
      default:
        if (isDraw(op))
          handleDraw(op, body, count);
        break;
      }
      i += 1 + count;
    } else if (type == Pm4Type::type2) {
      i += 1;  // single-dword filler/NOP
    } else {
      // Gnm draw command buffers contain only type-2 and type-3 packets; a
      // type-0 or type-1 header means we've run off the end of the real
      // commands into trailing zero/garbage padding. Stop there.
      break;
    }
  }
  if (g_trace && g_dcbSeen < 4)
    std::fprintf(stderr, "[gpu] dcb done: walked %u/%u words\n", i, words);
  if (g_trace && ++g_dcbSeen <= 4)
    dumpHist();
}

}  // namespace gpu
