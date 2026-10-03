#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>

#include "base/containers/hash_map.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "cpu/backend.h"
#include "guest/pause.h"
#include "guest/session.h"

namespace cpu {
namespace {
class HostThunk final : public Xbyak::CodeGenerator {
 public:
  HostThunk(uintptr_t target, bool stack_arguments) : CodeGenerator(2048) {
    push(rbp);
    mov(rbp, rsp);
    sub(rsp, 256);
    mov(qword[rsp + 240], rax);
    // Match the existing fourteen-argument HLE bridge, including stack args.
    for (int i = 0; i < (stack_arguments ? 8 : 0); ++i) {
      mov(rax, qword[rbp + 16 + i * 8]);
      mov(qword[rsp + i * 8], rax);
    }
    Xbyak::Label enter;
    mov(rax, reinterpret_cast<uintptr_t>(guest::PauseFlagAddress()));
    cmp(dword[rax], 0);
    je(enter, T_NEAR);
    const Xbyak::Reg64 args[] = {rdi, rsi, rdx, rcx, r8, r9};
    for (int i = 0; i < 6; ++i)
      mov(qword[rsp + 64 + i * 8], args[i]);
    for (int i = 0; i < 8; ++i)
      movdqu(ptr[rsp + 112 + i * 16], Xbyak::Xmm(i));
    mov(rax, reinterpret_cast<uintptr_t>(&guest::PausePoint));
    call(rax);
    for (int i = 0; i < 8; ++i)
      movdqu(Xbyak::Xmm(i), ptr[rsp + 112 + i * 16]);
    for (int i = 0; i < 6; ++i)
      mov(args[i], qword[rsp + 64 + i * 8]);
    L(enter);
    mov(rax, qword[rsp + 240]);
    mov(r11, target);
    call(r11);
    Xbyak::Label done;
    mov(r11, reinterpret_cast<uintptr_t>(guest::PauseFlagAddress()));
    cmp(dword[r11], 0);
    je(done, T_NEAR);
    mov(qword[rsp + 64], rax);
    mov(qword[rsp + 72], rdx);
    movdqu(ptr[rsp + 80], xmm0);
    movdqu(ptr[rsp + 96], xmm1);
    mov(rax, reinterpret_cast<uintptr_t>(&guest::PausePoint));
    call(rax);
    movdqu(xmm0, ptr[rsp + 80]);
    movdqu(xmm1, ptr[rsp + 96]);
    mov(rdx, qword[rsp + 72]);
    mov(rax, qword[rsp + 64]);
    L(done);
    mov(rsp, rbp);
    pop(rbp);
    ret();
  }
};
base::Mutex g_mutex;
base::HashMap<uintptr_t, HostThunk*> g_hle_thunks, g_syscall_thunks;
const guest::SessionReset g_session_reset([] {
  for (auto& [target, thunk] : g_hle_thunks)
    delete thunk;
  for (auto& [target, thunk] : g_syscall_thunks)
    delete thunk;
  guest::ResetResource(g_hle_thunks);
  guest::ResetResource(g_syscall_thunks);
});
uintptr_t FindThunk(uintptr_t target, bool stack_arguments) {
  base::LockGuard<base::Mutex> lock(g_mutex);
  auto& cache = stack_arguments ? g_hle_thunks : g_syscall_thunks;
  auto it = cache.find(target);
  if (it == cache.end())
    it = cache.emplace(target, new HostThunk(target, stack_arguments)).first;
  return reinterpret_cast<uintptr_t>(it->second->getCode());
}
}  // namespace

uintptr_t MakeHostThunk(void* host_fn, const char*) {
  return FindThunk(reinterpret_cast<uintptr_t>(host_fn), true);
}

uintptr_t MakeSyscallPauseThunk(const void* handler) {
  return FindThunk(reinterpret_cast<uintptr_t>(handler), false);
}
}  // namespace cpu
#endif
