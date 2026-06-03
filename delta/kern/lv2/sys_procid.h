#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

namespace krnl {
// Process / credential identity.
int PS4ABI sys_getppid();
int PS4ABI sys_getuid();
int PS4ABI sys_geteuid();
int PS4ABI sys_getgid();
int PS4ABI sys_getegid();
int PS4ABI sys_setuid(uint32_t uid);
int PS4ABI sys_seteuid(uint32_t uid);
int PS4ABI sys_setgid(uint32_t gid);
int PS4ABI sys_setegid(uint32_t gid);
int PS4ABI sys_setresuid(uint32_t ruid, uint32_t euid, uint32_t suid);
int PS4ABI sys_setresgid(uint32_t rgid, uint32_t egid, uint32_t sgid);
int PS4ABI sys_getresuid(uint32_t *ruid, uint32_t *euid, uint32_t *suid);
int PS4ABI sys_getresgid(uint32_t *rgid, uint32_t *egid, uint32_t *sgid);
int PS4ABI sys_issetugid();
int PS4ABI sys_getlogin(char *buf, uint32_t namelen);
int PS4ABI sys_setlogin(const char *name);
int PS4ABI sys_umask(uint32_t newmask);

// Process groups / sessions.
int PS4ABI sys_getpgrp();
int PS4ABI sys_setpgid(uint32_t pid, uint32_t pgid);
int PS4ABI sys_getpgid(uint32_t pid);
int PS4ABI sys_setsid();
int PS4ABI sys_getsid(uint32_t pid);

// Resource accounting / limits.
int PS4ABI sys_getrusage(int who, void *rusage);
int PS4ABI sys_getrlimit(int which, void *rlp);
int PS4ABI sys_setrlimit(int which, const void *rlp);

// System identity.
int PS4ABI sys_uname(void *name);
int PS4ABI sys_gethostname(char *buf, uint32_t len);
int PS4ABI sys_sethostname(const char *name, uint32_t len);
int PS4ABI sys_getdtablesize();

// Signals.
int PS4ABI sys_kill(uint32_t pid, int sig);
int PS4ABI sys_sigpending(void *set);
int PS4ABI sys_sigaltstack(const void *ss, void *oss);
int PS4ABI sys_sigtimedwait(const void *set, void *info, const void *timeout);
int PS4ABI sys_sigwaitinfo(const void *set, void *info);
int PS4ABI sys_sigwait(const void *set, int *sig);
int PS4ABI sys_sigsuspend(const void *sigmask);

// Realtime priority.
int PS4ABI sys_rtprio(int function, uint32_t pid, void *rtprio);

// Process waiting.
int PS4ABI sys_wait4(uint32_t pid, int *status, int options, void *rusage);
}  // namespace krnl
