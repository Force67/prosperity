#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceGnmDriver: GPU submission entry points. See libSceGnmDriver.cpp.
 */

#include "../vprx.h"

#include <cstdint>

extern "C" {

int PS4ABI sceGnmSubmitCommandBuffers(uint32_t count, void **dcbGpuAddrs,
                                     uint32_t *dcbSizes, void **ccbGpuAddrs,
                                     uint32_t *ccbSizes);
int PS4ABI sceGnmSubmitCommandBuffersForWorkload(uint32_t workload, uint32_t count,
                                                void **dcbGpuAddrs,
                                                uint32_t *dcbSizes,
                                                void **ccbGpuAddrs,
                                                uint32_t *ccbSizes);
int PS4ABI sceGnmSubmitAndFlipCommandBuffers(uint32_t count, void **dcbGpuAddrs,
                                            uint32_t *dcbSizes,
                                            void **ccbGpuAddrs,
                                            uint32_t *ccbSizes,
                                            uint32_t videoOutHandle,
                                            uint32_t displayBufferIndex,
                                            uint32_t flipMode, int64_t flipArg);
int PS4ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(
    uint32_t workload, uint32_t count, void **dcbGpuAddrs, uint32_t *dcbSizes,
    void **ccbGpuAddrs, uint32_t *ccbSizes, uint32_t videoOutHandle,
    uint32_t displayBufferIndex, uint32_t flipMode, int64_t flipArg);
int PS4ABI sceGnmSubmitDone();
int PS4ABI sceGnmAreSubmitsAllowed();
int PS4ABI sceGnmDingDong(uint32_t ringId, uint32_t offset);
int PS4ABI sceGnmDingDongForWorkload(uint32_t workload, uint32_t ringId,
                                    uint32_t offset);
int PS4ABI sceGnmFlushGarlic();
int PS4ABI sceGnmInsertWaitFlipDone(void *cmdBuffer, uint32_t size,
                                   uint32_t videoOutHandle, uint32_t bufferIndex);

}  // extern "C"
