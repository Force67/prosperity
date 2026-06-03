/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceGnmDriver: the GPU command-submission entry points only.
 *
 * The real Sony module builds PM4 command buffers (which we keep LLE) but its
 * submit path hands them to a GPU command processor we don't emulate yet: the
 * LLE driver spawns a GPU worker that, with no real GPU/queue backing, jumps
 * through an unset completion pointer and crashes (then corrupts the kernel
 * object table). Until a GCN->Vulkan backend exists, override just the submit/
 * flip/done entry points so they "succeed": the game advances and we drive the
 * flip (present + flip event) through the VideoOut HLE. This is the graphics
 * exception to the keep-PRX-LLE rule (mirrors shadps4, which HLEs Gnm submit).
 *
 * TODO(gpu): capture the dcb/ccb command buffers here and translate PM4 -> Vulkan
 * to actually render into the scanout buffer.
 */

#include "libSceGnmDriver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gpu/cmd_processor.h"

// VideoOut HLE flip bridge (same delta_runtime library).
extern "C" void prosperity_videoout_set_flip(int bufferIndex, int64_t flipArg);

namespace {
// Feed each draw command buffer to the GPU command processor.
void processDcbs(void **dcbGpuAddrs, uint32_t *dcbSizes, uint32_t count) {
  if (!dcbGpuAddrs || !dcbSizes)
    return;
  for (uint32_t i = 0; i < count; i++)
    gpu::submitDcb(dcbGpuAddrs[i], dcbSizes[i]);
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
  processDcbs(dcbGpuAddrs, dcbSizes, count);
  return 0;
}

int PS4ABI sceGnmSubmitCommandBuffersForWorkload(uint32_t workload, uint32_t count,
                                                void **dcbGpuAddrs,
                                                uint32_t *dcbSizes,
                                                void **ccbGpuAddrs,
                                                uint32_t *ccbSizes) {
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
  processDcbs(dcbGpuAddrs, dcbSizes, count);
  prosperity_videoout_set_flip(static_cast<int>(displayBufferIndex), flipArg);
  return 0;
}

int PS4ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(
    uint32_t workload, uint32_t count, void **dcbGpuAddrs, uint32_t *dcbSizes,
    void **ccbGpuAddrs, uint32_t *ccbSizes, uint32_t videoOutHandle,
    uint32_t displayBufferIndex, uint32_t flipMode, int64_t flipArg) {
  prosperity_videoout_set_flip(static_cast<int>(displayBufferIndex), flipArg);
  return 0;
}

int PS4ABI sceGnmSubmitDone() { return 0; }

int PS4ABI sceGnmAreSubmitsAllowed() { return 1; }

int PS4ABI sceGnmDingDong(uint32_t ringId, uint32_t offset) { return 0; }

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
