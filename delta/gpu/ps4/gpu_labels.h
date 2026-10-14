#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"
#include "base/functional/function.h"

namespace gpu::ps4 {

void WriteAfterGpu(base::Function<void()> write,
                   u64 address = 0,
                   u32 bytes = 0,
                   u64 value = 0);
bool LabelsAtGpu();
void SubmitHeldWrites();
bool HeldWriteValue(u64 address, u32* value);
bool HeldWriteOverlaps(u64 address, u64 bytes);

}  // namespace gpu::ps4
