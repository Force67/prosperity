#pragma once

#include <cstdint>
#include "base/arch.h"

namespace cpu {
using MmioWrite32 = bool (*)(uintptr_t address, u32 value);
bool HandleMmioWrite(void* signal_context,
                     uintptr_t address,
                     MmioWrite32 write);
}  // namespace cpu
