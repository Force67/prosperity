#pragma once

#include "base/arch.h"

namespace cpu::fex {

struct TraceEvt {
  char kind;  // 's' syscall, 'h' HLE thunk, 0 empty
  u32 id;
  u64 a0, a1, a2, a3;
  u64 ret;
  u64 caller;
  const char* name;
};
constexpr u32 kTraceRing = 96;

TraceEvt& TraceNext();

void BeginSyscall(u32 num);
void EndSyscall();

const TraceEvt* ThreadTraceRing();
const u32* ThreadTracePos();
const bool* ThreadInSyscall();

}  // namespace cpu::fex
