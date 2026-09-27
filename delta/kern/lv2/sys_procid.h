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

namespace kern {
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

int PS4ABI sys_suspend_process();
int PS4ABI sys_resume_process();
int PS4ABI sys_prepare_to_suspend_process();
int PS4ABI sys_prepare_to_resume_process();
int PS4ABI sys_process_terminate();
int PS4ABI sys_suspend_system();
int PS4ABI sys_sandbox_path(const char* path);
int PS4ABI sys_is_development_mode();
int PS4ABI sys_get_self_auth_info(const char* path, void* out);
int PS4ABI sys_get_sdk_compiled_version();
int PS4ABI sys_app_state_change();
int PS4ABI sys_getgroups(int gidsetlen, u32* gidset);
int PS4ABI sys_setgroups();
int PS4ABI sys_sigqueue();
int PS4ABI sys_abort2(const char* msg, int nargs, void** args);

int PS4ABI sys_exit();
int PS4ABI sys_rfork();
int PS4ABI sys_execve();
int PS4ABI sys_sigprocmask(int how, const int* set, int* oset);
int PS4ABI sys_sigaction(int sig, const void* act, void* oact);
int PS4ABI sys_getpid();

}  // namespace kern
