/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceGnmDriver: the GPU command-submission entry points only.
 *
 * The game builds PM4 command buffers (which we keep LLE). The real Sony submit
 * path hands them to a GPU command processor backed by hardware we don't have, so
 * we HLE just the submit/flip/done entry points (the graphics exception to the
 * keep-PRX-LLE rule). Each submit feeds the dcb/ccb to our GCN->Vulkan command
 * processor (gpu::submitDcb/submitCcb -> PM4 decode -> recompiled shaders -> the
 * headless Vulkan renderer); the *AndFlip variants additionally end the frame and
 * drive the flip (present + flip event) through the VideoOut HLE.
 *
 * The non-submitting entry points (SubmitDone/AreSubmitsAllowed/DingDong/
 * FlushGarlic/InsertWaitFlipDone) are intentional no-ops: our submit is synchronous
 * (the draws are already rendered when the call returns), so there is no async ring
 * to ding-dong, no EOP to wait on, and no Garlic write-combine buffer to flush.
 */

#include "libSceGnmDriver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gpu/cmd_processor.h"

// VideoOut HLE flip bridge (same delta_runtime library).
extern "C" void prosperity_videoout_set_flip(int bufferIndex, int64_t flipArg);
extern "C" uint64_t prosperity_videoout_buffer(int bufferIndex);

namespace {
// Feed each command buffer to the GPU command processor. The Constant Engine runs
// ahead of the Draw Engine, so process a submit's ccb (CE RAM -> shader constant
// buffers) before its dcb draws.
void processDcbs(void **dcbGpuAddrs, uint32_t *dcbSizes, void **ccbGpuAddrs,
                 uint32_t *ccbSizes, uint32_t count) {
  if (!dcbGpuAddrs || !dcbSizes)
    return;
  for (uint32_t i = 0; i < count; i++) {
    if (ccbGpuAddrs && ccbSizes && ccbGpuAddrs[i] && ccbSizes[i])
      gpu::submitCcb(ccbGpuAddrs[i], ccbSizes[i]);
    gpu::submitDcb(dcbGpuAddrs[i], dcbSizes[i]);
  }
}
}  // namespace

namespace {
// Env-gated PM4 dump (DELTA_PM4DUMP=1): walk the dcb type-3 packets and tally
// the IT opcodes so we can see what the title actually submits. Guest GPU
// addresses are identity-mapped, so the dcb is directly readable on the host.
const bool g_pm4Dump = std::getenv("DELTA_PM4DUMP") != nullptr;
int g_pm4Frames = 0;

const char *itName(uint32_t op) {
  switch (op) {
  case 0x10: return "NOP";
  case 0x12: return "CLEAR_STATE";
  case 0x22: return "COND_EXEC";
  case 0x28: return "INDEX_BASE";
  case 0x2A: return "INDEX_BUFFER_SIZE";
  case 0x2D: return "DRAW_INDEX_AUTO";
  case 0x27: return "DRAW_INDEX_2";
  case 0x2F: return "DRAW_INDEX_OFFSET_2";
  case 0x36: return "WAIT_REG_MEM";
  case 0x37: return "WRITE_DATA";
  case 0x3C: return "ACQUIRE_MEM";
  case 0x40: return "DMA_DATA";
  case 0x46: return "EVENT_WRITE";
  case 0x47: return "EVENT_WRITE_EOP";
  case 0x49: return "RELEASE_MEM";
  case 0x4C: return "DISPATCH_DIRECT";
  case 0x68: return "SET_CONFIG_REG";
  case 0x69: return "SET_CONTEXT_REG";
  case 0x76: return "SET_SH_REG";
  case 0x79: return "SET_UCONFIG_REG";
  default: return "?";
  }
}

void dumpPm4(void **dcbGpuAddrs, uint32_t *dcbSizes, uint32_t count) {
  if (!g_pm4Dump || g_pm4Frames > 3 || !dcbGpuAddrs || !dcbSizes)
    return;
  g_pm4Frames++;
  for (uint32_t b = 0; b < count; b++) {
    auto *p = static_cast<uint32_t *>(dcbGpuAddrs[b]);
    uint32_t words = dcbSizes[b] / 4;
    std::fprintf(stderr, "[pm4] dcb[%u] @%p words=%u\n", b, (void *)p, words);
    if (!p) continue;
    uint32_t i = 0, draws = 0;
    while (i < words) {
      uint32_t hdr = p[i];
      uint32_t type = hdr >> 30;
      uint32_t cnt = ((hdr >> 16) & 0x3FFF) + 1;  // dword count after header
      if (type == 3) {
        uint32_t op = (hdr >> 8) & 0xFF;
        std::fprintf(stderr, "[pm4]   T3 %-20s op=%#04x cnt=%u\n", itName(op), op, cnt);
        if (op == 0x2D || op == 0x27 || op == 0x2F || op == 0x4C) draws++;
        i += 1 + cnt;
      } else if (type == 2) {
        i += 1;  // filler
      } else if (type == 0) {
        i += 1 + cnt;
      } else {
        break;  // type 1 / desync
      }
    }
    std::fprintf(stderr, "[pm4]   -> %u draw/dispatch packets\n", draws);
  }
}
}  // namespace

extern "C" {

int PS4ABI sceGnmSubmitCommandBuffers(uint32_t count, void **dcbGpuAddrs,
                                     uint32_t *dcbSizes, void **ccbGpuAddrs,
                                     uint32_t *ccbSizes) {
  dumpPm4(dcbGpuAddrs, dcbSizes, count);
  processDcbs(dcbGpuAddrs, dcbSizes, ccbGpuAddrs, ccbSizes, count);
  return 0;
}

int PS4ABI sceGnmSubmitCommandBuffersForWorkload(uint32_t workload, uint32_t count,
                                                void **dcbGpuAddrs,
                                                uint32_t *dcbSizes,
                                                void **ccbGpuAddrs,
                                                uint32_t *ccbSizes) {
  // Same as sceGnmSubmitCommandBuffers but tagged with a workload id; the command
  // buffers must still be processed (this was stubbed, silently dropping every
  // draw the game submitted through the workload path).
  dumpPm4(dcbGpuAddrs, dcbSizes, count);
  processDcbs(dcbGpuAddrs, dcbSizes, ccbGpuAddrs, ccbSizes, count);
  return 0;
}

int PS4ABI sceGnmSubmitAndFlipCommandBuffers(uint32_t count, void **dcbGpuAddrs,
                                            uint32_t *dcbSizes,
                                            void **ccbGpuAddrs,
                                            uint32_t *ccbSizes,
                                            uint32_t videoOutHandle,
                                            uint32_t displayBufferIndex,
                                            uint32_t flipMode, int64_t flipArg) {
  // The flip target buffer is what should be scanned out next; record it so the
  // VideoOut flip pump presents it and posts the flip-complete event.
  dumpPm4(dcbGpuAddrs, dcbSizes, count);
  processDcbs(dcbGpuAddrs, dcbSizes, ccbGpuAddrs, ccbSizes, count);
  // Present the render target that this flip displays (the guest scanout buffer).
  gpu::endFrame(prosperity_videoout_buffer(static_cast<int>(displayBufferIndex)));
  prosperity_videoout_set_flip(static_cast<int>(displayBufferIndex), flipArg);
  return 0;
}

int PS4ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(
    uint32_t workload, uint32_t count, void **dcbGpuAddrs, uint32_t *dcbSizes,
    void **ccbGpuAddrs, uint32_t *ccbSizes, uint32_t videoOutHandle,
    uint32_t displayBufferIndex, uint32_t flipMode, int64_t flipArg) {
  // Was a flip-only stub that dropped the submitted command buffers. Process them
  // (and end the frame on the flip) exactly like the non-workload variant.
  dumpPm4(dcbGpuAddrs, dcbSizes, count);
  processDcbs(dcbGpuAddrs, dcbSizes, ccbGpuAddrs, ccbSizes, count);
  gpu::endFrame(prosperity_videoout_buffer(static_cast<int>(displayBufferIndex)));
  prosperity_videoout_set_flip(static_cast<int>(displayBufferIndex), flipArg);
  return 0;
}

int PS4ABI sceGnmSubmitDone() { return 0; }

int PS4ABI sceGnmAreSubmitsAllowed() { return 1; }

int PS4ABI sceGnmDingDong(uint32_t ringId, uint32_t offset) {
  static int n = 0;
  if (std::getenv("DELTA_GPU_DINGDONG") && n++ < 20)
    std::fprintf(stderr, "[gnm] sceGnmDingDong ring=%u offset=%#x\n", ringId, offset);
  return 0;
}

int PS4ABI sceGnmDingDongForWorkload(uint32_t workload, uint32_t ringId,
                                    uint32_t offset) {
  return 0;
}

int PS4ABI sceGnmFlushGarlic() { return 0; }

int PS4ABI sceGnmInsertWaitFlipDone(void *cmdBuffer, uint32_t size,
                                   uint32_t videoOutHandle, uint32_t bufferIndex) {
  return 0;
}

}  // extern "C"
