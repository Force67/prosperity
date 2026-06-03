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

// VideoOut HLE flip bridge (same delta_runtime library).
extern "C" void prosperity_videoout_set_flip(int bufferIndex, int64_t flipArg);

extern "C" {

int PS4ABI sceGnmSubmitCommandBuffers(uint32_t count, void **dcbGpuAddrs,
                                     uint32_t *dcbSizes, void **ccbGpuAddrs,
                                     uint32_t *ccbSizes) {
  // TODO(gpu): parse + render the PM4 in dcbGpuAddrs[0..count]. For now, accept.
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
