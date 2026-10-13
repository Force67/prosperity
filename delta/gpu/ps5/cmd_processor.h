#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

namespace gpu::ps5 {

void SubmitDcb(const void* dcb, u32 size_bytes, u32 tag = 0);
u32 SubmitDcbRing(const void* dcb,
                  u32 size_bytes,
                  u32 queue,
                  u64 visible_ns = 0);

void SubmitCcb(const void* ccb, u32 size_bytes);

void EndFrame(u64 scanout_base);

}  // namespace gpu::ps5
