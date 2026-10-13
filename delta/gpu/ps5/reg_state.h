#pragma once

#include "base/arch.h"

#include "gpu/ps5/agc_regs.h"

namespace gpu::ps5 {

void SetRegs(Regs& regs, u32 base, const u32* body, u32 count);

void SetRegRun(Regs& regs, u32 first_reg, const u32* values, u32 count);

void LoadRegImage(Regs& regs, u32 base, const u32* body, u32 count);

void LoadRegBlock(Regs& regs, u32 base, const u32* body, u32 count);

void WriteDataRegs(Regs& regs, const u32* body, u32 count);

}  // namespace gpu::ps5
