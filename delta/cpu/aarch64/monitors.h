#pragma once

#include "base/arch.h"

namespace FEXCore::Core {
struct CPUState;
}

namespace cpu::fex {

void StartMonitors();

void ReportThreadReturn(u32 id,
                        u64 entry_rip,
                        const FEXCore::Core::CPUState& end);

}  // namespace cpu::fex
