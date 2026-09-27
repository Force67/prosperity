#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "guest_abi.h"

namespace krnl {
// Process / credential identity.
int PS4ABI sys_getppid();
int PS4ABI sys_getuid();
int PS4ABI sys_geteuid();
int PS4ABI sys_getgid();
int PS4ABI sys_getegid();
int PS4ABI sys_setuid(u32 uid);
int PS4ABI sys_seteuid(u32 uid);
int PS4ABI sys_setgid(u32 gid);
int PS4ABI sys_setegid(u32 gid);
int PS4ABI sys_setresuid(u32 ruid, u32 euid, u32 suid);
int PS4ABI sys_setresgid(u32 rgid, u32 egid, u32 sgid);
int PS4ABI sys_getresuid(u32* ruid, u32* euid, u32* suid);
int PS4ABI sys_getresgid(u32* rgid, u32* egid, u32* sgid);
int PS4ABI sys_issetugid();
int PS4ABI sys_getlogin(char* buf, u32 namelen);
int PS4ABI sys_setlogin(const char* name);
int PS4ABI sys_umask(u32 newmask);

// Process groups / sessions.
int PS4ABI sys_getpgrp();
int PS4ABI sys_setpgid(u32 pid, u32 pgid);
int PS4ABI sys_getpgid(u32 pid);
int PS4ABI sys_setsid();
int PS4ABI sys_getsid(u32 pid);

// Resource accounting / limits.
int PS4ABI sys_getrusage(int who, void* rusage);
int PS4ABI sys_getrlimit(int which, void* rlp);
int PS4ABI sys_setrlimit(int which, const void* rlp);

// System identity.
int PS4ABI sys_uname(void* name);
int PS4ABI sys_gethostname(char* buf, u32 len);
int PS4ABI sys_sethostname(const char* name, u32 len);
int PS4ABI sys_getdtablesize();

// Signals.
int PS4ABI sys_kill(u32 pid, int sig);
int PS4ABI sys_sigpending(void* set);
int PS4ABI sys_sigaltstack(const void* ss, void* oss);
int PS4ABI sys_sigtimedwait(const void* set, void* info, const void* timeout);
int PS4ABI sys_sigwaitinfo(const void* set, void* info);
int PS4ABI sys_sigwait(const void* set, int* sig);
int PS4ABI sys_sigsuspend(const void* sigmask);

// Realtime priority.
int PS4ABI sys_rtprio(int function, u32 pid, void* rtprio);

// Process waiting.
int PS4ABI sys_wait4(u32 pid, int* status, int options, void* rusage);
}  // namespace krnl
