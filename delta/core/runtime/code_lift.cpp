
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdio>
#include <xbyak.h>

#include "code_lift.h"
#include <logger/logger.h>

#include "kern/proc.h"

namespace krnl {
uintptr_t lv2_get(uint32_t sysIndex);
}

namespace runtime {
static Xbyak::Operand::Code capstone_to_xbyak(x86_reg reg) {
#define CASE_R(x)                                                              \
  case X86_REG_E##x:                                                           \
  case X86_REG_R##x: {                                                         \
    return Xbyak::Operand::R##x;                                               \
  }
#define CASE_N(x)                                                              \
  case X86_REG_R##x##D:                                                        \
  case X86_REG_R##x: {                                                         \
    return Xbyak::Operand::R##x;                                               \
  }
  switch (reg) {
    CASE_R(AX)
    CASE_R(CX)
    CASE_R(DX)
    CASE_R(BX)
    CASE_R(SP)
    CASE_R(BP)
    CASE_R(SI)
    CASE_R(DI)
    CASE_N(8)
    CASE_N(9)
    CASE_N(10)
    CASE_N(11)
    CASE_N(12)
    CASE_N(13)
    CASE_N(14)
    CASE_N(15)
  }
  __debugbreak();
  return Xbyak::Operand::Code::RAX;
#undef CASE_N
#undef CASE_R
}

// for debugging
static void printOpInfo(const cs_x86_op &op) {
  std::printf("Operand: Type %d, Reg %d, (Mem: base %d)\n", op.type, op.reg,
              op.mem.base);
}

codeLift::codeLift(uint8_t *&rip) : ripPointer(rip) {}

codeLift::~codeLift() {
  if (handle) {
    cs_free(insn, 1);
    cs_close(&handle);

    // just to be sure...
    handle = 0;
  }
}

bool codeLift::init() {
  auto err = cs_open(CS_ARCH_X86, CS_MODE_64, &handle);
  if (err == CS_ERR_MEM) {
    LOG_ERROR("codeLift: not enough mem for disasembler");
    return false;
  }

  // setup disasm config
  cs_option(handle, CS_OPT_DETAIL, CS_OPT_ON);
  return true;
}

static bool is_bmi1_instruction(int op) {
  return op == X86_INS_ANDN || op == X86_INS_BEXTR || op == X86_INS_BLSI ||
         op == X86_INS_BLSMSK || op == X86_INS_BLSR || op == X86_INS_TZCNT;
};

bool codeLift::transform(uint8_t *data, size_t size, uint64_t base) {
  insn = cs_malloc(handle);

  const uint8_t *codePtr = data; // iterator
  while (cs_disasm_iter(handle, &codePtr, &size, &base, insn)) {

    auto detail = insn->detail->x86;
    // uint32_t dest = static_cast<uint32_t>(X86_REL_ADDR(*insn));

    auto getOps = [&](int32_t ofs) { return &data[insn->address + ofs]; };

    /*syscall -> custom handler*/
    if (insn->id == X86_INS_SYSCALL) {
      emit_syscall(getOps(-10), *(uint32_t *)(getOps(-7)));
    }

    /*interrupt -> debugbreak*/
    else if (insn->id == X86_INS_INT) {
      uint8_t *tgt = getOps(0);
      *(uint16_t *)(tgt) = 0xCCCC;
    }

    /*else if (is_bmi1_instruction(insn->id)) {
            std::printf("encountered bm1 instruction %p\n", getOps(0));
    }*/

    /*fs base (tls) access*/
    else {
      /*idea inspired by uplift*/
      bool isTls = false;

      for (uint8_t i = 0; i < insn->detail->x86.op_count; i++) {
        auto operand = insn->detail->x86.operands[i];
        if (operand.type == X86_OP_MEM) {
          if (operand.mem.segment == X86_REG_FS) {
            isTls = true;
            break;
          } else if (operand.mem.segment == X86_REG_DS ||
                     operand.mem.segment == X86_REG_ES ||
                     operand.mem.segment == X86_REG_GS) {
            //__debugbreak();
          }
        }
      }

      if (isTls && insn->id == X86_INS_MOV) {
        emit_fsbase(getOps(0));
      }
    }
  }

  return false;
}

void codeLift::emit_syscall(uint8_t *base, uint32_t idx) {
  auto address = krnl::lv2_get(idx);
  if (address) {
    /*call to rax*/
    *(uint16_t *)base = 0xB848;
    *(uint64_t *)(base + 2) = address;
    *(uint16_t *)(base + 10) = 0xE0FF;
  }
}

/*fetch fs base ptr from current process*/
static PS4ABI void *getFsBase() {
  return krnl::proc::getActive()->getEnv().fsBase;
}

/*this implementation is based on uplift*/
void codeLift::emit_fsbase(uint8_t *base) {
  if (insn->detail->x86.op_count != 2)
    __debugbreak();

  auto operands = insn->detail->x86.operands;

  if (operands[0].type != X86_OP_REG)
    __debugbreak();

  if (operands[1].type != X86_OP_MEM || operands[1].mem.segment != X86_REG_FS ||
      operands[1].mem.base != X86_REG_INVALID ||
      operands[1].mem.index != X86_REG_INVALID) {
    __debugbreak();
  }

  if (operands[0].size != 8 && operands[0].size != 4) {
    __debugbreak();
    return;
  }

  /*translate the register that we set*/
  auto reg = Xbyak::Reg64(capstone_to_xbyak(operands[0].reg));

  /*jit assemble a register setter*/
  struct fsGen : Xbyak::CodeGenerator {
    /*note: this is sys-v abi*/
    fsGen(Xbyak::Reg64 reg, uint32_t disp, uint8_t size) {
      int idx = reg.getIdx();

      mov(rax, reinterpret_cast<uintptr_t>(&getFsBase));
      call(rax);

      // sysv returns directly into rax
      if (idx != Xbyak::Reg64::RAX) {
        mov(reg, rax);
      }

      if (disp)
        add(reg, disp);

      if (size == 4)
        mov(reg.cvt32(), ptr[reg]);
      else
        mov(reg, ptr[reg]);

      ret();
    }
  };

  uint32_t dispOff = insn->size > 5 ? *(uint32_t *)(base + 5) : 0;
  fsGen gen(reg, dispOff, operands[0].size);

  /*call directly in rip zone*/
  base[0] = 0xE8;
  auto disp = static_cast<uint32_t>(ripPointer - &base[5]);
  *reinterpret_cast<uint32_t *>(&base[1]) = disp;

  /*pad out any remaining code*/
  if (insn->size > 5)
    std::memset(&base[5], 0x90, insn->size - 5);

  /*align the block and pad out the rest*/
  const auto alignedSize = align_up<size_t>(gen.getSize(), 8);
  if (gen.getSize() < alignedSize)
    std::memset(ripPointer + gen.getSize(), 0xCC, alignedSize - gen.getSize());

  std::memcpy(ripPointer, gen.getCode(), gen.getSize());
  ripPointer += alignedSize;
}
} // namespace runtime