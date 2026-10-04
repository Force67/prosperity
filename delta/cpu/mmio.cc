#include "cpu/mmio.h"

#include <ucontext.h>
#include <cstring>

namespace cpu {
bool HandleMmioWrite(void* signal_context,
                     uintptr_t address,
                     MmioWrite32 write) {
  if (!signal_context)
    return false;
  auto* uc = static_cast<ucontext_t*>(signal_context);
#if defined(__x86_64__)
  auto* regs = uc->uc_mcontext.gregs;
  const auto* code = reinterpret_cast<const u8*>(regs[REG_RIP]);
  static constexpr int kRegisters[] = {
      REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
      REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
  size_t length = 0;
  u8 rex = 0;
  if ((code[length] & 0xf0) == 0x40)
    rex = code[length++];
  if (rex & 8)
    return false;
  const u8 opcode = code[length++];
  if (opcode != 0x89 && opcode != 0xc7)
    return false;
  const u8 modrm = code[length++];
  const u8 mod = modrm >> 6, rm = modrm & 7;
  if (mod == 3 || (opcode == 0xc7 && (modrm & 0x38)))
    return false;
  uintptr_t target = 0;
  bool displacement32 = mod == 2, rip_relative = false;
  if (rm == 4) {
    const u8 sib = code[length++];
    const u8 index = ((sib >> 3) & 7) | ((rex & 2) ? 8 : 0);
    if (index != 4)
      target += static_cast<uintptr_t>(regs[kRegisters[index]]) << (sib >> 6);
    if (mod == 0 && (sib & 7) == 5)
      displacement32 = true;
    else
      target += regs[kRegisters[(sib & 7) | ((rex & 1) ? 8 : 0)]];
  } else if (mod == 0 && rm == 5) {
    displacement32 = true;
    rip_relative = true;
  } else {
    target = regs[kRegisters[rm | ((rex & 1) ? 8 : 0)]];
  }
  if (mod == 1)
    target += static_cast<i8>(code[length++]);
  if (displacement32) {
    i32 displacement;
    std::memcpy(&displacement, code + length, 4);
    target += displacement;
    length += 4;
  }
  u32 value;
  if (opcode == 0xc7) {
    std::memcpy(&value, code + length, 4);
    length += 4;
  } else {
    value = regs[kRegisters[((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0)]];
  }
  if (rip_relative)
    target += regs[REG_RIP] + length;
  if (target != address || !write(address, value))
    return false;
  regs[REG_RIP] += length;
  return true;
#elif defined(__aarch64__)
  const u32 instruction = *reinterpret_cast<const u32*>(uc->uc_mcontext.pc);
  const u32 source = instruction & 31, base = (instruction >> 5) & 31;
  uintptr_t target =
      base == 31 ? uc->uc_mcontext.sp : uc->uc_mcontext.regs[base];
  if ((instruction & 0xffc00000) == 0xb9000000) {
    target += ((instruction >> 10) & 0xfff) * 4;
  } else if ((instruction & 0xffe00c00) == 0xb8000000) {
    target += static_cast<i32>(instruction << 11) >> 23;
  } else if ((instruction & 0xfffffc00) != 0x889ffc00) {
    return false;
  }
  const u32 value = source == 31 ? 0 : uc->uc_mcontext.regs[source];
  if (target != address || !write(address, value))
    return false;
  uc->uc_mcontext.pc += 4;
  return true;
#else
  return false;
#endif
}
}  // namespace cpu
