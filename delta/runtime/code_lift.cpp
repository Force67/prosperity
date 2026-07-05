
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// The lifter rewrites guest x86-64 in place (Xbyak codegen) so it runs natively
// on an x86-64 host. It is meaningless (and won't compile) on aarch64, where
// guest code runs through the FEXCore JIT instead (see delta/cpu/fex_backend).
#if defined(DELTA_BACKEND_NATIVE)

#include <base.h>
#include <cstdio>
#include <xbyak.h>

#include "code_lift.h"
#include <logger/logger.h>

#include "kern/proc.h"

namespace krnl {
uintptr_t lv2_get(uint32_t sysIndex);
uintptr_t lv2_get_ps5(uint32_t sysIndex);
}

// returns the calling host thread's guest fs base (per-thread guest TLS).
extern "C" uint64_t krnl_current_fsbase();

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

codeLift::codeLift(uint8_t *&rip, uint8_t *ripEndIn)
    : ripPointer(rip), ripEnd(ripEndIn) {}

codeLift::~codeLift() {
  if (handle) {
    // insn is only allocated in transform(); may be null if it never ran
    if (insn)
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
  // Executable PT_LOAD segments interleave code with rodata (strings, constants,
  // jump tables) that capstone can't decode. A plain linear sweep stops dead at
  // the first such blob, leaving every later instruction un-lifted; that code
  // still runs natively, so its raw `syscall`/`int`/`mov fs:[..]` hit the host
  // CPU (an un-lifted guest TLS write corrupts the host fs base). Resync past the
  // undecodable byte so code after a data blob is lifted too. The rewriters below
  // bail (rather than trap) on anything they don't recognise, so a data byte that
  // briefly mis-decodes as a syscall/fs access is left untouched.
  while (size > 0) {
    if (!cs_disasm_iter(handle, &codePtr, &size, &base, insn)) {
      ++codePtr;
      --size;
      ++base;
      continue;
    }

    auto detail = insn->detail->x86;
    // uint32_t dest = static_cast<uint32_t>(X86_REL_ADDR(*insn));

    auto getOps = [&](int32_t ofs) { return &data[insn->address + ofs]; };

    /*syscall -> custom handler*/
    if (insn->id == X86_INS_SYSCALL) {
      // emit_syscall writes 10 bytes before the insn (the `mov rax,imm/jmp` it
      // builds over the `mov eax,nr; syscall` pair); skip if that would underflow.
      // It also no-ops unless the preceding bytes form a known syscall number, so
      // a stray `0f 05` in resync'd data is left alone.
      if (insn->address >= 10)
        emit_syscall(getOps(-10), *(uint32_t *)(getOps(-7)));
    }

    // `int` is intentionally NOT rewritten: turning every `cd xx` (common in the
    // rodata the resync sweeps over) into a breakpoint would mass-corrupt data,
    // and it bought nothing (a raw guest `int 0x41` already faults like int3). A
    // reached SDK assert stays fatal; handle it in the signal path if needed.

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
  // PS5 titles route to the separate Prospero syscall layer (FreeBSD 11 ABI);
  // never the PS4 table.
  auto *proc = krnl::proc::getActive();
  const bool ps5 = proc && proc->getPlatform() == krnl::proc::platform::ps5;
  auto address = ps5 ? krnl::lv2_get_ps5(idx) : krnl::lv2_get(idx);
  if (std::getenv("DELTA_SYSLIFT_TRACE"))
    std::fprintf(stderr, "[syslift] site=%p idx=%u -> trampoline=%#lx\n",
                 (void *)base, idx, (unsigned long)address);
  if (address) {
    /*call to rax*/
    *(uint16_t *)base = 0xB848;
    *(uint64_t *)(base + 2) = address;
    *(uint16_t *)(base + 10) = 0xE0FF;
  }
}

/*this implementation is based on uplift*/
void codeLift::emit_fsbase(uint8_t *base) {
  auto &x = insn->detail->x86;
  auto *operands = x.operands;
  if (x.op_count != 2)
    return;  // unrecognised form (or a data byte mis-decoded as a mov): leave it

  // Identify the fs-memory operand and the register operand. We handle both the
  // read `mov reg, fs:[disp]` and the write `mov fs:[disp], reg`; the write form
  // is what libc's TLS init uses, and leaving it raw lets it clobber the host fs
  // base. Only the absolute fs:[disp] form (no base/index) and 4/8-byte GPR
  // operands are handled; anything else is left untouched (capstone_to_xbyak
  // only maps 32/64-bit registers, so we must not feed it sub-registers).
  int memIdx = operands[0].type == X86_OP_MEM   ? 0
               : operands[1].type == X86_OP_MEM ? 1
                                                : -1;
  int regIdx = operands[0].type == X86_OP_REG   ? 0
               : operands[1].type == X86_OP_REG ? 1
                                                : -1;
  if (memIdx < 0 || regIdx < 0)
    return;
  auto &mem = operands[memIdx];
  auto &gpr = operands[regIdx];
  if (mem.mem.segment != X86_REG_FS || mem.mem.base != X86_REG_INVALID ||
      mem.mem.index != X86_REG_INVALID)
    return;
  if (gpr.size != 8 && gpr.size != 4)
    return;

  const bool isWrite = (memIdx == 0);  // mov fs:[disp], reg
  auto reg = Xbyak::Reg64(capstone_to_xbyak(gpr.reg));

  // Per-thread guest fs base: call krnl_current_fsbase() (a trivial leaf that
  // does `mov rax, fs:[tpoff]; ret`, clobbering only rax). mov/push/pop/call/ret
  // leave flags untouched. A read clobbers only its destination register; a store
  // must clobber nothing, so the write path preserves rax and rcx (rcx is the
  // value scratch, captured before rax is reused for the helper call).
  struct fsGen : Xbyak::CodeGenerator {
    fsGen(Xbyak::Reg64 reg, int32_t disp, uint8_t size, bool isWrite,
          uintptr_t helper) {
      if (!isWrite) {
        bool isRax = reg.getIdx() == Xbyak::Operand::RAX;
        if (!isRax)
          push(rax);
        mov(rax, helper);
        call(rax);  // rax = this thread's guest fs base
        // disp may be negative (static TLS sits below fs)
        auto dst = isRax ? rax : reg;
        if (size == 4)
          mov(dst.cvt32(), ptr[rax + disp]);
        else
          mov(dst, ptr[rax + disp]);
        if (!isRax)
          pop(rax);
        ret();
      } else {
        push(rax);
        push(rcx);
        mov(rcx, reg);  // capture value first (handles reg == rax/rcx)
        mov(rax, helper);
        call(rax);  // rax = guest fs base
        if (size == 4)
          mov(ptr[rax + disp], ecx);
        else
          mov(ptr[rax + disp], rcx);
        pop(rcx);
        pop(rax);
        ret();
      }
    }
  };

  auto fsDisp = static_cast<int32_t>(mem.mem.disp);
  fsGen gen(reg, fsDisp, gpr.size, isWrite,
            reinterpret_cast<uintptr_t>(&krnl_current_fsbase));

  // Don't run past the rip-zone (sized to the segment in the loader). Leaving a
  // tail access raw is worse than ideal but far better than scribbling past the
  // zone into the next module; in practice the zone is sized so this never trips.
  const auto alignedSize = align_up<size_t>(gen.getSize(), 8);
  if (ripEnd && ripPointer + alignedSize > ripEnd)
    return;

  /*call directly in rip zone*/
  base[0] = 0xE8;
  auto disp = static_cast<uint32_t>(ripPointer - &base[5]);
  *reinterpret_cast<uint32_t *>(&base[1]) = disp;

  /*pad out any remaining code*/
  if (insn->size > 5)
    std::memset(&base[5], 0x90, insn->size - 5);

  /*align the block and pad out the rest*/
  if (gen.getSize() < alignedSize)
    std::memset(ripPointer + gen.getSize(), 0xCC, alignedSize - gen.getSize());

  std::memcpy(ripPointer, gen.getCode(), gen.getSize());
  ripPointer += alignedSize;
}
} // namespace runtime

#endif // DELTA_BACKEND_NATIVE