#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceGnmDriver: GPU submission entry points. See lib_sce_gnm_driver.cc.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"

extern "C" {

int PS4ABI sceGnmSubmitCommandBuffers(u32 count,
                                      void** dcb_gpu_addrs,
                                      u32* dcb_sizes,
                                      void** ccb_gpu_addrs,
                                      u32* ccb_sizes);
int PS4ABI sceGnmSubmitCommandBuffersForWorkload(u32 workload,
                                                 u32 count,
                                                 void** dcb_gpu_addrs,
                                                 u32* dcb_sizes,
                                                 void** ccb_gpu_addrs,
                                                 u32* ccb_sizes);
int PS4ABI sceGnmSubmitAndFlipCommandBuffers(u32 count,
                                             void** dcb_gpu_addrs,
                                             u32* dcb_sizes,
                                             void** ccb_gpu_addrs,
                                             u32* ccb_sizes,
                                             u32 video_out_handle,
                                             u32 display_buffer_index,
                                             u32 flip_mode,
                                             i64 flip_arg);
int PS4ABI
sceGnmSubmitAndFlipCommandBuffersForWorkload(u32 workload,
                                             u32 count,
                                             void** dcb_gpu_addrs,
                                             u32* dcb_sizes,
                                             void** ccb_gpu_addrs,
                                             u32* ccb_sizes,
                                             u32 video_out_handle,
                                             u32 display_buffer_index,
                                             u32 flip_mode,
                                             i64 flip_arg);
int PS4ABI sceGnmSubmitDone();
int PS4ABI sceGnmAreSubmitsAllowed();
int PS4ABI sceGnmDingDong(u32 ring_id, u32 offset);
int PS4ABI sceGnmDingDongForWorkload(u32 workload, u32 ring_id, u32 offset);
int PS4ABI sceGnmFlushGarlic();
int PS4ABI sceGnmInsertWaitFlipDone(void* cmd_buffer,
                                    u32 size,
                                    u32 video_out_handle,
                                    u32 buffer_index);

}  // extern "C"
