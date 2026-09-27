#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <capstone/capstone.h>
#include "base/arch.h"

namespace kern {
struct procEnv;
}

namespace runtime {
// code analysis:
// convert unsupported code
class CodeLift {
 public:
  CodeLift(u8*& rip, u8* rip_end = nullptr);
  ~CodeLift();

  bool Init();
  bool Transform(u8*, size_t size, u64 base = 0);

 private:
  void EmitSyscall(u8* base, u32 idx);
  void EmitFsbase(u8* base);

  csh handle_ = 0;
  cs_insn* insn_ = nullptr;
  u8*& rip_pointer_;
  u8* rip_end_ = nullptr;  // end of the rip-zone; stop emitting stubs past it
};
}  // namespace runtime