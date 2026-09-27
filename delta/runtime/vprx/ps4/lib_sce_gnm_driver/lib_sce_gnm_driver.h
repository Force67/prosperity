#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceGnmDriver: GPU submission entry points. See lib_sce_gnm_driver.cc.
 */

#include "guest_abi.h"
#include "../../vprx.h"
#include "base/arch.h"


extern "C" {

int PS4ABI sceGnmSubmitCommandBuffers(u32 count, void **dcbGpuAddrs,
                                     u32 *dcbSizes, void **ccbGpuAddrs,
                                     u32 *ccbSizes);
int PS4ABI sceGnmSubmitCommandBuffersForWorkload(u32 workload, u32 count,
                                                void **dcbGpuAddrs,
                                                u32 *dcbSizes,
                                                void **ccbGpuAddrs,
                                                u32 *ccbSizes);
int PS4ABI sceGnmSubmitAndFlipCommandBuffers(u32 count, void **dcbGpuAddrs,
                                            u32 *dcbSizes,
                                            void **ccbGpuAddrs,
                                            u32 *ccbSizes,
                                            u32 videoOutHandle,
                                            u32 displayBufferIndex,
                                            u32 flipMode, i64 flipArg);
int PS4ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(
    u32 workload, u32 count, void **dcbGpuAddrs, u32 *dcbSizes,
    void **ccbGpuAddrs, u32 *ccbSizes, u32 videoOutHandle,
    u32 displayBufferIndex, u32 flipMode, i64 flipArg);
int PS4ABI sceGnmSubmitDone();
int PS4ABI sceGnmAreSubmitsAllowed();
int PS4ABI sceGnmDingDong(u32 ringId, u32 offset);
int PS4ABI sceGnmDingDongForWorkload(u32 workload, u32 ringId,
                                    u32 offset);
int PS4ABI sceGnmFlushGarlic();
int PS4ABI sceGnmInsertWaitFlipDone(void *cmdBuffer, u32 size,
                                   u32 videoOutHandle, u32 bufferIndex);

}  // extern "C"
