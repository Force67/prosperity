
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
#include <cstring>
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
  if (!insn_)
    insn_ = cs_malloc(handle_);
  // Executable PT_LOAD segments interleave code with rodata, and only two
  // shapes need rewriting, so search their bytes instead of disassembling
  // everything (a full Capstone sweep of a 500 MB eboot took most of boot).
  // Each candidate is then decoded on its own; the rewriters bail on anything
  // they don't recognise.
  auto decode_at = [&](size_t off) {
    const u8* code = data + off;
    size_t left = size - off;
    uint64_t addr = base + off;
    return cs_disasm_iter(handle_, &code, &left, &addr, insn_);
  };

  // Whether a linear sweep from a little before `target` lands on it: x86
  // decoding resynchronises within a few instructions, so this is where the
  // full sweep would have put an instruction boundary.
  auto starts_at = [&](size_t target) {
    size_t off = target > 512 ? target - 512 : 0;
    while (off < target)
      off += decode_at(off) ? insn_->size : 1;
    return off == target;
  };
  auto legacy_prefix = [](u8 b) {
    return b == 0x66 || b == 0x67 || b == 0xf2 || b == 0xf3 || b == 0x2e ||
           b == 0x3e || b == 0x26 || b == 0x36 || b == 0x65 || b == 0xf0;
  };

  // `mov fs:[disp], reg` / `mov reg, fs:[disp]`, which may carry more legacy
  // prefixes before the fs one (a REX prefix must sit right before the
  // opcode, so never before it). The patch covers the whole instruction.
  for (const u8* p = data; (p = static_cast<const u8*>(
                                std::memchr(p, 0x64, data + size - p)));
       p++) {
    size_t off = p - data;
    if (!decode_at(off) || insn_->id != X86_INS_MOV)
      continue;
    size_t first = off;
    while (first && off - first < 14 && legacy_prefix(data[first - 1]))
      first--;
    while (first < off && !starts_at(first))
      first++;
    if (first == off && !starts_at(off))
      continue;
    off = first;
    if (!decode_at(off) || insn_->id != X86_INS_MOV)
      continue;
    const auto& x = insn_->detail->x86;
    for (u8 i = 0; i < x.op_count; i++)
      if (x.operands[i].type == X86_OP_MEM &&
          x.operands[i].mem.segment == X86_REG_FS) {
        EmitFsbase(data + off);
        break;
      }
  }

  // libkernel's stub: `mov rax, imm32; mov r10, rcx; syscall`.
  static constexpr u8 kMovRax[] = {0x48, 0xc7, 0xc0};
  static constexpr u8 kSyscall[] = {0x0f, 0x05};
  for (u8* p = data + 10; p + 2 <= data + size; p++) {
    p = static_cast<u8*>(memmem(p, data + size - p, kSyscall, 2));
    if (!p)
      break;
    if (!std::memcmp(p - 10, kMovRax, sizeof(kMovRax)))
      EmitSyscall(p - 10, *reinterpret_cast<u32*>(p - 7));
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

  if (kSysliftTrace)
    BASE_LOGI("fslift", "site={:p} size={}", (void*)base, insn_->size);
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
