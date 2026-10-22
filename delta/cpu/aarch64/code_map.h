#pragma once

#include <cstddef>

#include "base/arch.h"

namespace cpu::fex {

struct ExecRange {
  u64 base, size;
};

void AddExecRange(u64 base, u64 size);
ExecRange FindExecRange(u64 address);

void AddModuleRange(u64 base, u64 size, const char* name);
void SymbolizeModuleAddress(u64 address, char* out, size_t n);

void ResetCodeMap();

}  // namespace cpu::fex
