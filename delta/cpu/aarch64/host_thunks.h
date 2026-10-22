#pragma once

#include "base/arch.h"

namespace cpu::fex {

struct HostThunk {
  void* fn = nullptr;
  const char* name = nullptr;
};

HostThunk LookupHostThunk(u32 idx);

void ResetHostThunks();

}  // namespace cpu::fex
