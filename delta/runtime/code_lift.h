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
class codeLift {
public:
  codeLift(u8 *&rip, u8 *ripEnd = nullptr);
  ~codeLift();

  bool init();
  bool transform(u8 *, size_t size, u64 base = 0);

private:
  void emit_syscall(u8 *base, u32 idx);
  void emit_fsbase(u8 *base);

  csh handle = 0;
  cs_insn *insn = nullptr;
  u8 *&ripPointer;
  u8 *ripEnd = nullptr;  // end of the rip-zone; stop emitting stubs past it
};
}