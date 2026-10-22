#include "cpu/aarch64/host_thunks.h"

#include <sys/mman.h>
#include <cstring>
#include <initializer_list>

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "cpu/aarch64/code_map.h"
#include "cpu/aarch64/cpu_map.h"
#include "cpu/backend.h"
#include "guest/session.h"
#include "logger/logger.h"
#include "memory_debug/memory_debug.h"

namespace cpu::fex {
namespace {

constexpr size_t kThunkPoolSize = 0x100000;
constexpr size_t kThunkStride = 16;

base::Mutex g_thunk_mutex;
u8* g_thunk_pool = nullptr;
size_t g_thunk_pool_used = 0;
base::Vector<void*> g_host_thunks;
base::Vector<base::String> g_thunk_names;

u8* AllocateLocked(size_t stride) {
  if (!g_thunk_pool) {
    g_thunk_pool = static_cast<u8*>(CpuMap(nullptr, kThunkPoolSize,
                                           PROT_READ | PROT_WRITE | PROT_EXEC,
                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_thunk_pool == MAP_FAILED) {
      g_thunk_pool = nullptr;
      LOG_ERROR("fex: host-thunk pool mmap failed");
      return nullptr;
    }
    LOG_INFO("fex: host-thunk pool {:#x}+{:#x}",
             reinterpret_cast<u64>(g_thunk_pool), (u64)kThunkPoolSize);
    AddExecRange(reinterpret_cast<u64>(g_thunk_pool), kThunkPoolSize);
  }
  if (g_thunk_pool_used + stride > kThunkPoolSize) {
    LOG_ERROR("fex: host-thunk pool exhausted");
    return nullptr;
  }
  u8* t = g_thunk_pool + g_thunk_pool_used;
  g_thunk_pool_used += stride;
  return t;
}

u32 BindLocked(void* fn, const char* name) {
  const u32 idx = static_cast<u32>(g_host_thunks.size());
  g_host_thunks.push_back(fn);
  g_thunk_names.push_back(name);
  return idx;
}

class CodeWriter {
 public:
  explicit CodeWriter(u8* at) : p_(at) {}
  void Emit(std::initializer_list<u8> bytes) {
    for (u8 b : bytes)
      *p_++ = b;
  }
  void Emit32(u32 v) {
    std::memcpy(p_, &v, 4);
    p_ += 4;
  }
  void Emit64(u64 v) {
    std::memcpy(p_, &v, 8);
    p_ += 8;
  }

 private:
  u8* p_;
};

}  // namespace

HostThunk LookupHostThunk(u32 idx) {
  base::LockGuard lk(g_thunk_mutex);
  if (idx >= g_host_thunks.size())
    return {};
  return {g_host_thunks[idx], g_thunk_names[idx].c_str()};
}

void ResetHostThunks() {
  if (g_thunk_pool)
    CpuUnmap(g_thunk_pool, kThunkPoolSize);
  g_thunk_pool = nullptr;
  g_thunk_pool_used = 0;
  guest::ResetResource(g_host_thunks);
  guest::ResetResource(g_thunk_names);
}

}  // namespace cpu::fex

namespace cpu {
using fex::AllocateLocked;
using fex::BindLocked;
using fex::CodeWriter;
using fex::g_host_thunks;
using fex::g_thunk_mutex;
using fex::g_thunk_names;
using fex::g_thunk_pool;
using fex::g_thunk_pool_used;
using fex::kThunkStride;

const char* HostThunkNameForAddr(uintptr_t addr, u32* idx_out) {
  base::LockGuard lk(g_thunk_mutex);
  if (!g_thunk_pool)
    return nullptr;
  const auto base = reinterpret_cast<uintptr_t>(g_thunk_pool);
  if (addr < base || addr >= base + g_thunk_pool_used)
    return nullptr;
  const u32 idx = static_cast<u32>((addr - base) / kThunkStride);
  if (idx_out)
    *idx_out = idx;
  if (idx < g_thunk_names.size() && !g_thunk_names[idx].empty())
    return g_thunk_names[idx].c_str();
  return "";
}

uintptr_t MakeHostThunk(void* host_fn, const char* name) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
  base::LockGuard lk(g_thunk_mutex);
  u8* t = AllocateLocked(kThunkStride);
  if (!t)
    return 0;
  const u32 sc = kHostThunkSyscallBase | BindLocked(host_fn, name ? name : "");
  CodeWriter w(t);
  w.Emit({0x49, 0x89, 0xCA});  // mov r10, rcx
  w.Emit({0xB8});              // mov eax, imm32
  w.Emit32(sc);
  w.Emit({0x0F, 0x05});  // syscall
  w.Emit({0xC3});        // ret
  return reinterpret_cast<uintptr_t>(t);
}

uintptr_t MakeGuestReturnHook(void* real_target,
                              u32 hook_id,
                              void* logger_fn,
                              const char* name) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
  base::LockGuard lk(g_thunk_mutex);
  u8* t = AllocateLocked(64);
  if (!t)
    return 0;
  const u32 magic =
      kHostThunkSyscallBase | BindLocked(logger_fn, name ? name : "guesthook");
  CodeWriter w(t);
  // On entry: rsp%16==8; args rdi,rsi,rdx,rcx; [rsp]=caller return address.
  w.Emit({0x57});                    // push rdi      ; save a0
  w.Emit({0x56});                    // push rsi      ; save a1
  w.Emit({0x52});                    // push rdx      ; save a2
  w.Emit({0x51});                    // push rcx      ; save a3
  w.Emit({0x48, 0x83, 0xEC, 0x08});  // sub rsp, 8    ; realign to 16
  w.Emit({0x49, 0xBB});              // movabs r11, realTarget
  w.Emit64(reinterpret_cast<u64>(real_target));
  w.Emit({0x41, 0xFF, 0xD3});        // call r11      ; run real fn -> rax=ret
  w.Emit({0x48, 0x83, 0xC4, 0x08});  // add rsp, 8
  w.Emit({0x49, 0x89, 0xC1});        // mov r9, rax   ; logger arg6 = ret
  w.Emit({0x41, 0x58});              // pop r8        ; a3 -> r8
  w.Emit({0x59});                    // pop rcx       ; a2 -> rcx
  w.Emit({0x5A});                    // pop rdx       ; a1 -> rdx
  w.Emit({0x5E});                    // pop rsi       ; a0 -> rsi
  w.Emit({0xBF});                    // mov edi, imm32
  w.Emit32(hook_id);                 //   = hookId
  w.Emit({0x50});                    // push rax      ; preserve ret
  w.Emit({0x49, 0x89, 0xCA});        // mov r10, rcx  ; handler reads a2 here
  w.Emit({0xB8});                    // mov eax, imm32
  w.Emit32(magic);                   //   = kHostThunkSyscallBase|loggerIdx
  w.Emit({0x0F, 0x05});              // syscall       ; -> loggerFn(rdi..r9)
  w.Emit({0x58});                    // pop rax       ; restore real ret
  w.Emit({0xC3});                    // ret
  return reinterpret_cast<uintptr_t>(t);
}

uintptr_t MakeGuestLockWrapper(void* real_target,
                               void* lock_fn,
                               void* unlock_fn,
                               const char* name) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
  base::LockGuard lk(g_thunk_mutex);
  u8* t = AllocateLocked(64);
  if (!t)
    return 0;
  const u32 lock_idx = BindLocked(lock_fn, name ? name : "guestlock");
  const u32 unlock_idx = BindLocked(unlock_fn, name ? name : "guestunlock");
  CodeWriter w(t);
  // Reached by `jmp`, so [rsp] is still the caller's return address.
  w.Emit({0x57});  // push rdi        ; save a0
  w.Emit({0x56});  // push rsi        ; save a1
  w.Emit({0x52});  // push rdx        ; save a2
  w.Emit({0x51});  // push rcx        ; save a3
  w.Emit({0xB8});  // mov eax, imm32
  w.Emit32(kHostThunkSyscallBase | lock_idx);
  w.Emit({0x0F, 0x05});              // syscall         ; -> lockFn()
  w.Emit({0x59});                    // pop rcx
  w.Emit({0x5A});                    // pop rdx
  w.Emit({0x5E});                    // pop rsi
  w.Emit({0x5F});                    // pop rdi
  w.Emit({0x48, 0x83, 0xEC, 0x08});  // sub rsp, 8      ; realign for the call
  w.Emit({0x49, 0xBB});              // movabs r11, realTarget
  w.Emit64(reinterpret_cast<u64>(real_target));
  w.Emit({0x41, 0xFF, 0xD3});        // call r11        ; the real function
  w.Emit({0x48, 0x83, 0xC4, 0x08});  // add rsp, 8
  w.Emit({0x50});                    // push rax        ; preserve the result
  w.Emit({0xB8});                    // mov eax, imm32
  w.Emit32(kHostThunkSyscallBase | unlock_idx);
  w.Emit({0x0F, 0x05});  // syscall         ; -> unlockFn()
  w.Emit({0x58});        // pop rax
  w.Emit({0xC3});        // ret
  return reinterpret_cast<uintptr_t>(t);
}

// The prologue must be position-independent and end on an instruction
// boundary.
uintptr_t MakeGuestTrampoline(const void* fn_bytes,
                              u32 prologue_len,
                              const void* continue_at) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
  base::LockGuard lk(g_thunk_mutex);
  const size_t need = prologue_len + 14;
  u8* t = AllocateLocked((need + 15) & ~size_t(15));
  if (!t)
    return 0;
  std::memcpy(t, fn_bytes, prologue_len);
  CodeWriter w(t + prologue_len);
  w.Emit({0xFF, 0x25});  // jmp qword [rip+0]
  w.Emit32(0);
  w.Emit64(reinterpret_cast<u64>(continue_at));
  return reinterpret_cast<uintptr_t>(t);
}

}  // namespace cpu
