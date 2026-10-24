// getcontext/setcontext/swapcontext through the context trampoline.

#include <gtest/gtest.h>
#include <xbyak.h>
#include <cstring>

#include "guest_abi.h"
#include "kern/lv2/dispatch.h"
#include "kern/lv2/sys_context.h"

namespace {
constexpr u32 kGetcontext = 421;
constexpr u32 kSetcontext = 422;
constexpr u32 kSwapcontext = 423;
constexpr size_t kUcSize = 0x500;
constexpr size_t kMcRip = 0x40 + 0xa0;
constexpr size_t kMcRsp = 0x40 + 0xb8;
constexpr size_t kMcRdi = 0x40 + 0x08;
constexpr size_t kMcRsi = 0x40 + 0x10;

uintptr_t Trampoline(u32 sid) {
  return kern::Lv2ContextTrampoline(&kern::sys_context, sid);
}

struct Saved : Xbyak::CodeGenerator {
  void Prologue() {
    for (const auto& r : {rbx, rbp, r12, r13, r14, r15})
      push(r);
    sub(rsp, 8);
  }
  void Epilogue() {
    add(rsp, 8);
    for (const auto& r : {r15, r14, r13, r12, rbp, rbx})
      pop(r);
    ret();
  }
};

// f(uc, counter): getcontext, clobber, setcontext; the second resume returns
// r12 + r13 + xmm0 as they were at getcontext.
struct RoundTrip : Saved {
  RoundTrip() {
    Xbyak::Label second, fail, out;
    Prologue();
    mov(r14, rdi);
    mov(r15, rsi);
    mov(r12, 0x1111);
    mov(r13, 0x2222);
    movq(xmm0, r12);
    mov(rdi, r14);
    mov(rax, Trampoline(kGetcontext));
    call(rax);
    jc(fail);
    inc(qword[r15]);
    cmp(qword[r15], 2);
    je(second);
    xor_(r12, r12);
    xor_(r13, r13);
    pxor(xmm0, xmm0);
    mov(rdi, r14);
    mov(rax, Trampoline(kSetcontext));
    call(rax);
    mov(rax, -1);  // setcontext returned
    jmp(out);
    L(second);
    movq(rax, xmm0);
    add(rax, r12);
    add(rax, r13);
    jmp(out);
    L(fail);
    mov(rax, -2);
    L(out);
    Epilogue();
  }
};

// getctx(uc): capture a context to edit.
struct Capture : Saved {
  Capture() {
    Prologue();
    mov(rax, Trampoline(kGetcontext));
    call(rax);
    setc(al);
    movzx(eax, al);
    Epilogue();
  }
};

// swap(a, b, counter): swapcontext(a, b); returns *counter once resumed in a.
// `away` is where b starts: it bumps the counter and resumes a.
struct Swap : Saved {
  Xbyak::Label away_label;
  Swap() {
    Xbyak::Label fail, out;
    Prologue();
    mov(rbx, rdx);
    mov(rax, Trampoline(kSwapcontext));
    call(rax);
    jc(fail);
    mov(rax, qword[rbx]);
    jmp(out);
    L(fail);
    mov(rax, -2);
    L(out);
    Epilogue();
    L(away_label);  // rdi = a, rsi = counter, on its own stack
    inc(qword[rsi]);
    mov(rax, Trampoline(kSetcontext));
    call(rax);
    ud2();
  }
};

TEST(Context, GetSetRoundTrip) {
  RoundTrip code;
  alignas(64) u8 uc[kUcSize] = {};
  u64 counter = 0;
  const u64 r = code.getCode<u64 (*)(u8*, u64*)>()(uc, &counter);
  EXPECT_EQ(counter, 2u);
  EXPECT_EQ(r, 0x1111u * 2 + 0x2222u);
}

TEST(Context, SwapToAnotherStackAndBack) {
  Capture capture;
  Swap swap;
  alignas(64) u8 a[kUcSize] = {};
  alignas(64) u8 b[kUcSize] = {};
  ASSERT_EQ(capture.getCode<int (*)(u8*)>()(b), 0);

  static u8 stack[64 * 1024];
  u64 counter = 0;
  const u64 rip = reinterpret_cast<u64>(swap.away_label.getAddress());
  const u64 rsp = (reinterpret_cast<u64>(stack + sizeof(stack)) & ~u64{15}) - 8;
  const u64 rdi = reinterpret_cast<u64>(a);
  const u64 rsi = reinterpret_cast<u64>(&counter);
  std::memcpy(b + kMcRip, &rip, 8);
  std::memcpy(b + kMcRsp, &rsp, 8);
  std::memcpy(b + kMcRdi, &rdi, 8);
  std::memcpy(b + kMcRsi, &rsi, 8);

  const u64 r = swap.getCode<u64 (*)(u8*, u8*, u64*)>()(a, b, &counter);
  EXPECT_EQ(counter, 1u);
  EXPECT_EQ(r, 1u);
}

TEST(Context, RejectsBadMcontextLength) {
  RoundTrip code;  // only for its trampolines
  Capture capture;
  alignas(64) u8 uc[kUcSize] = {};
  ASSERT_EQ(capture.getCode<int (*)(u8*)>()(uc), 0);
  const u64 bad_len = 0x123;
  std::memcpy(uc + 0x40 + 0xc8, &bad_len, 8);
  struct SetOnly : Saved {
    SetOnly() {
      Prologue();
      mov(rax, Trampoline(kSetcontext));
      call(rax);
      setc(cl);
      movzx(ecx, cl);
      shl(rcx, 32);
      or_(rax, rcx);
      Epilogue();
    }
  } set;
  const u64 r = set.getCode<u64 (*)(u8*)>()(uc);
  EXPECT_EQ(r >> 32, 1u);      // carry: error
  EXPECT_EQ(r & 0xffff, 22u);  // EINVAL
}
}  // namespace
