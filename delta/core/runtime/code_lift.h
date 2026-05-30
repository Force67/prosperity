#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <capstone/capstone.h>

namespace krnl {
struct procEnv;
}

namespace runtime {
// code analysis:
// convert unsupported code
class codeLift {
public:
  codeLift(uint8_t *&rip);
  ~codeLift();

  bool init();
  bool transform(uint8_t *, size_t size, uint64_t base = 0);

private:
  void emit_syscall(uint8_t *base, uint32_t idx);
  void emit_fsbase(uint8_t *base);

  csh handle = 0;
  cs_insn *insn = nullptr;
  uint8_t *&ripPointer;
};
}