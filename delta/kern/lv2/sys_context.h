#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * User contexts: getcontext, setcontext, swapcontext, sigreturn and
 * thr_create. Orbis and Prospero share the ucontext layout (a 0x480-byte
 * mcontext with an embedded XSAVE image); a PS5 mcontext may also carry extra
 * extended state by pointer.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "kern/lv2/dispatch.h"
#include "kern/lv2/sys_signal.h"

namespace kern {

// True for the ids that switch or capture the calling thread's context.
bool IsContextSyscall(u32 sid);

// The context trampoline's handler for those ids.
void PS4ABI sys_context(u32 sid, GuestFrame* frame, u8* xsave);

// A ucontext for the state in `f`/`xsave` with signal mask `mask`.
// `as_syscall_return` makes it resume as the syscall returning 0 (getcontext).
void FillUcontext(const GuestFrame& f,
                  const u8* xsave,
                  const GuestSigset& mask,
                  bool as_syscall_return,
                  u8* uc);

// set_mcontext: validates the mcontext at `mc` and loads it into `f` and
// `xsave`; the fs base it carries, if any, goes to `fsbase`.
int LoadMcontext(GuestFrame* f,
                 u8* xsave,
                 const u8* mc,
                 bool* has_fsbase,
                 u64* fsbase);

int PS4ABI sys_thr_create(const void* ucontext, i64* tid, int flags);

}  // namespace kern
