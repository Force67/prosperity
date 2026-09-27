
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

#include <xbyak.h>
#include <cstdio>
#include "base/arch.h"
#include "base/logging.h"
#include "base/math/alignment.h"
#include "cpu/backend.h"

#include "logger/logger.h"
#include "runtime/code_lift.h"

#include "kern/lv2/dispatch.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kSysliftTrace, "DELTA_SYSLIFT_TRACE", false);
}  // namespace

namespace runtime {
static Xbyak::Operand::Code CapstoneToXbyak(x86_reg reg) {
#define CASE_R(x)                \
  case X86_REG_E##x:             \
  case X86_REG_R##x: {           \
    return Xbyak::Operand::R##x; \
  }
#define CASE_N(x)                \
  case X86_REG_R##x##D:          \
  case X86_REG_R##x: {           \
    return Xbyak::Operand::R##x; \
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
  __builtin_trap();
  return Xbyak::Operand::Code::RAX;
#undef CASE_N
#undef CASE_R
}

// for debugging
static void PrintOpInfo(const cs_x86_op& op) {
  BASE_LOGI("syslift", "Operand: Type {}, Reg {}, (Mem: base {})", (int)op.type,
            (int)op.reg, (int)op.mem.base);
}

CodeLift::CodeLift(u8*& rip, u8* rip_end_in)
    : rip_pointer_(rip), rip_end_(rip_end_in) {}

CodeLift::~CodeLift() {
  if (handle_) {
    // insn is only allocated in transform(); may be null if it never ran
    if (insn_)
      cs_free(insn_, 1);
    cs_close(&handle_);

    // just to be sure...
    handle_ = 0;
  }
}

bool CodeLift::Init() {
  auto err = cs_open(CS_ARCH_X86, CS_MODE_64, &handle_);
  if (err == CS_ERR_MEM) {
    LOG_ERROR("codeLift: not enough mem for disasembler");
    return false;
  }

  // setup disasm config
  cs_option(handle_, CS_OPT_DETAIL, CS_OPT_ON);
  return true;
}

static bool IsBmi1Instruction(int op) {
  return op == X86_INS_ANDN || op == X86_INS_BEXTR || op == X86_INS_BLSI ||
         op == X86_INS_BLSMSK || op == X86_INS_BLSR || op == X86_INS_TZCNT;
};

bool CodeLift::Transform(u8* data, size_t size, u64 base) {
  insn_ = cs_malloc(handle_);

  const u8* code_ptr = data;  // iterator
  uint64_t addr = base;       // capstone's own cursor type
  // Executable PT_LOAD segments interleave code with rodata (strings,
  // constants, jump tables) that capstone can't decode. A plain linear sweep
  // stops dead at the first such blob, leaving every later instruction
  // un-lifted; that code still runs natively, so its raw `syscall`/`int`/`mov
  // fs:[..]` hit the host CPU (an un-lifted guest TLS write corrupts the host
  // fs base). Resync past the undecodable byte so code after a data blob is
  // lifted too. The rewriters below bail (rather than trap) on anything they
  // don't recognise, so a data byte that briefly mis-decodes as a syscall/fs
  // access is left untouched.
  while (size > 0) {
    if (!cs_disasm_iter(handle_, &code_ptr, &size, &addr, insn_)) {
      ++code_ptr;
      --size;
      ++addr;
      continue;
    }

    auto detail = insn_->detail->x86;
    // u32 dest = static_cast<u32>(X86_REL_ADDR(*insn));

    auto get_ops = [&](i32 ofs) { return &data[insn_->address + ofs]; };

    /*syscall -> custom handler*/
    if (insn_->id == X86_INS_SYSCALL) {
      // EmitSyscall writes 10 bytes before the insn (the `mov rax,imm/jmp` it
      // builds over the `mov eax,nr; syscall` pair); skip if that would
      // underflow. It also no-ops unless the preceding bytes form a known
      // syscall number, so a stray `0f 05` in resync'd data is left alone.
      if (insn_->address >= 10)
        EmitSyscall(get_ops(-10), *(u32*)(get_ops(-7)));
    }

    // `int` is intentionally NOT rewritten: turning every `cd xx` (common in
    // the rodata the resync sweeps over) into a breakpoint would mass-corrupt
    // data, and it bought nothing (a raw guest `int 0x41` already faults like
    // int3). A reached SDK assert stays fatal; handle it in the signal path if
    // needed.

    /*fs base (tls) access*/
    else {
      /*idea inspired by uplift*/
      bool is_tls = false;

      for (u8 i = 0; i < insn_->detail->x86.op_count; i++) {
        auto operand = insn_->detail->x86.operands[i];
        if (operand.type == X86_OP_MEM) {
          if (operand.mem.segment == X86_REG_FS) {
            is_tls = true;
            break;
          } else if (operand.mem.segment == X86_REG_DS ||
                     operand.mem.segment == X86_REG_ES ||
                     operand.mem.segment == X86_REG_GS) {
            //__builtin_trap();
          }
        }
      }

      if (is_tls && insn_->id == X86_INS_MOV) {
        EmitFsbase(get_ops(0));
      }
    }
  }

  return false;
}

void CodeLift::EmitSyscall(u8* base, u32 idx) {
  auto address = kern::Lv2Lookup(idx);
  if (kSysliftTrace)
    BASE_LOGI("syslift", "site={:p} idx={} -> trampoline={:#x}", (void*)base,
              idx, (unsigned long)address);
  if (address) {
    // `mov rax, trampoline; call rax` over the stub's 12 bytes. It must be a
    // CALL, not a JMP: the bytes right after the syscall are libkernel's own
    // `jb cerror; ret`, which is what turns the BSD carry/errno return into the
    // -1 + errno every sce* wrapper tests for. Jumping would return past that
    // tail straight to the wrapper's caller, so a failing syscall arrived as
    // rax = errno instead of -1, and a wrapper like sceKernelPollEventFlag
    // (`mov ecx,eax; xor eax,eax; cmp ecx,-1`) then reported success.
    *(u16*)base = 0xB848;
    *(u64*)(base + 2) = address;
    *(u16*)(base + 10) = 0xD0FF;
  }
}

/*this implementation is based on uplift*/
void CodeLift::EmitFsbase(u8* base) {
  auto& x = insn_->detail->x86;
  auto* operands = x.operands;
  if (x.op_count != 2)
    return;  // unrecognised form (or a data byte mis-decoded as a mov): leave
             // it

  // Identify the fs-memory operand and the register operand. We handle both the
  // read `mov reg, fs:[disp]` and the write `mov fs:[disp], reg`; the write
  // form is what libc's TLS init uses, and leaving it raw lets it clobber the
  // host fs base. Only the absolute fs:[disp] form (no base/index) and 4/8-byte
  // GPR operands are handled; anything else is left untouched
  // (capstone_to_xbyak only maps 32/64-bit registers, so we must not feed it
  // sub-registers).
  int mem_idx = operands[0].type == X86_OP_MEM   ? 0
                : operands[1].type == X86_OP_MEM ? 1
                                                 : -1;
  int reg_idx = operands[0].type == X86_OP_REG   ? 0
                : operands[1].type == X86_OP_REG ? 1
                                                 : -1;
  if (mem_idx < 0 || reg_idx < 0)
    return;
  auto& mem = operands[mem_idx];
  auto& gpr = operands[reg_idx];
  if (mem.mem.segment != X86_REG_FS || mem.mem.base != X86_REG_INVALID ||
      mem.mem.index != X86_REG_INVALID)
    return;
  if (gpr.size != 8 && gpr.size != 4)
    return;
  if (insn_->size < 5)
    return;

  const bool is_write = (mem_idx == 0);  // mov fs:[disp], reg
  auto reg = Xbyak::Reg64(CapstoneToXbyak(gpr.reg));

  struct FsGen : Xbyak::CodeGenerator {
    FsGen(Xbyak::Reg64 reg,
          i32 disp,
          u8 size,
          bool is_write,
          i32 guest_fs_offset,
          i32 scratch_offset) {
      if (!is_write) {
        putSeg(fs);
        mov(reg, ptr[guest_fs_offset]);
        if (size == 4)
          mov(reg.cvt32(), ptr[reg + disp]);
        else
          mov(reg, ptr[reg + disp]);
      } else {
        auto tmp = reg.getIdx() == Xbyak::Operand::RAX ? rcx : rax;
        putSeg(fs);
        mov(ptr[scratch_offset], tmp);
        putSeg(fs);
        mov(tmp, ptr[guest_fs_offset]);
        if (size == 4)
          mov(ptr[tmp + disp], reg.cvt32());
        else
          mov(ptr[tmp + disp], reg);
        putSeg(fs);
        mov(tmp, ptr[scratch_offset]);
      }
    }
  };

  auto fs_disp = static_cast<i32>(mem.mem.disp);
  FsGen gen(reg, fs_disp, gpr.size, is_write, cpu::HostGuestFsOffset(),
            cpu::HostFsScratchOffset());

  // Don't run past the rip-zone (sized to the segment in the loader). Leaving a
  // tail access raw is worse than ideal but far better than scribbling past the
  // zone into the next module; in practice the zone is sized so this never
  // trips.
  const auto stub_size = gen.getSize() + 5;
  const auto aligned_size = base::Align<size_t>(stub_size, 8);
  if (rip_end_ && rip_pointer_ + aligned_size > rip_end_)
    return;

  base[0] = 0xE9;
  auto disp = static_cast<u32>(rip_pointer_ - &base[5]);
  *reinterpret_cast<u32*>(&base[1]) = disp;

  /*pad out any remaining code*/
  if (insn_->size > 5)
    std::memset(&base[5], 0x90, insn_->size - 5);

  std::memcpy(rip_pointer_, gen.getCode(), gen.getSize());
  auto* return_jump = rip_pointer_ + gen.getSize();
  return_jump[0] = 0xE9;
  *reinterpret_cast<u32*>(&return_jump[1]) =
      static_cast<u32>(&base[insn_->size] - &return_jump[5]);
  if (stub_size < aligned_size)
    std::memset(rip_pointer_ + stub_size, 0xCC, aligned_size - stub_size);
  rip_pointer_ += aligned_size;
}
}  // namespace runtime

#endif  // DELTA_BACKEND_NATIVE
