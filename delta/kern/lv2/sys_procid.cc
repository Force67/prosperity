/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "cpu/backend.h"
#include "guest/session.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_procid.h"

// One emulated process pretending to be a running game; pid matches sys_getpid.
// Reported as a normal non-root user (uid/gid 1): root makes some guests take
// privileged paths we don't model. No real OS state behind these; setters
// accept, getters report the fixed fake identity.

namespace kern {
static constexpr int kGamePid = 0x1337;

// A finite open-file ceiling the guest can size fd tables / fd_sets against.
// getrlimit(NOFILE) and getdtablesize must agree on it.
static constexpr i64 kMaxFiles = 4096;

int PS4ABI sys_getppid() {
  return 0;
}

int PS4ABI sys_getuid() {
  return 1;
}
int PS4ABI sys_geteuid() {
  return 1;
}
int PS4ABI sys_getgid() {
  return 1;
}
int PS4ABI sys_getegid() {
  return 1;
}

int PS4ABI sys_setuid(u32 uid) {
  return 0;
}
int PS4ABI sys_seteuid(u32 uid) {
  return 0;
}
int PS4ABI sys_setgid(u32 gid) {
  return 0;
}
int PS4ABI sys_setegid(u32 gid) {
  return 0;
}
int PS4ABI sys_setresuid(u32 ruid, u32 euid, u32 suid) {
  return 0;
}
int PS4ABI sys_setresgid(u32 rgid, u32 egid, u32 sgid) {
  return 0;
}

int PS4ABI sys_getresuid(u32* ruid, u32* euid, u32* suid) {
  if (ruid)
    *ruid = 1;
  if (euid)
    *euid = 1;
  if (suid)
    *suid = 1;
  return 0;
}
int PS4ABI sys_getresgid(u32* rgid, u32* egid, u32* sgid) {
  if (rgid)
    *rgid = 1;
  if (egid)
    *egid = 1;
  if (sgid)
    *sgid = 1;
  return 0;
}

// Report "clean" so libc doesn't disable env-based behaviour or harden itself.
int PS4ABI sys_issetugid() {
  return 0;
}

int PS4ABI sys_getlogin(char* buf, u32 namelen) {
  if (!buf || namelen == 0)
    return -SysError::eINVAL;
  static const char kName[] = "game";
  u32 n = sizeof(kName);  // includes NUL
  if (n > namelen)
    n = namelen;
  std::memcpy(buf, kName, n);
  buf[n - 1] = '\0';
  return 0;
}

int PS4ABI sys_setlogin(const char* name) {
  return 0;
}

// Track the mask so the set/get-previous round trip stays consistent.
int PS4ABI sys_umask(u32 newmask) {
  static u32 mask = 022;
  u32 prev = mask;
  mask = newmask & 0777;
  return prev;
}

int PS4ABI sys_getpgrp() {
  return kGamePid;
}
int PS4ABI sys_setpgid(u32 pid, u32 pgid) {
  return 0;
}
int PS4ABI sys_getpgid(u32 pid) {
  return kGamePid;
}
int PS4ABI sys_setsid() {
  return kGamePid;
}
int PS4ABI sys_getsid(u32 pid) {
  return kGamePid;
}

// struct rusage is ~144 bytes on LP64; we keep no accounting, so zero it.
int PS4ABI sys_getrusage(int who, void* rusage) {
  if (rusage)
    std::memset(rusage, 0, 144);
  return 0;
}

// FreeBSD rlimit {rlim_cur, rlim_max}, RLIM_INFINITY = INT64_MAX. Most
// resources are genuinely unbounded, but a few must read finite or the guest
// sizes structures against infinity: NOFILE (fd tables/fd_sets), NPROC/NPTS
// (process/pty caps), CORE 0 (a retail box never dumps core).
int PS4ABI sys_getrlimit(int which, void* rlp) {
  if (!rlp)
    return -SysError::eFAULT;
  // The kernel rejects a resource index past RLIM_NLIMITS-1 (0xC) with EINVAL.
  if (static_cast<unsigned>(which) > 0xC)
    return -SysError::eINVAL;
  enum { kCore = 4, kNproc = 7, kNofile = 8, kNpts = 11 };
  i64 lim = INT64_MAX;
  switch (which) {
    case kNofile:
      lim = kMaxFiles;
      break;
    case kNproc:
      lim = 256;
      break;
    case kNpts:
      lim = 256;
      break;
    case kCore:
      lim = 0;
      break;
    default:
      break;  // unlimited
  }
  auto* r = static_cast<i64*>(rlp);
  r[0] = lim;  // rlim_cur
  r[1] = lim;  // rlim_max
  return 0;
}

int PS4ABI sys_setrlimit(int which, const void* rlp) {
  if (static_cast<unsigned>(which) > 0xC)
    return -SysError::eINVAL;
  return 0;  // accepted; we don't enforce soft/hard limits
}

// FreeBSD utsname: five char[SYS_NMLN(=32)] fields back to back.
int PS4ABI sys_uname(void* name) {
  if (!name)
    return -SysError::eFAULT;

  constexpr size_t NMLN = 32;  // NOLINT(readability-identifier-naming)
  char* p = static_cast<char*>(name);
  std::memset(p, 0, NMLN * 5);

  std::strcpy(p + NMLN * 0, "FreeBSD");      // sysname
  std::strcpy(p + NMLN * 1, "ps4");          // nodename
  std::strcpy(p + NMLN * 2, "9.0-RELEASE");  // release
  std::strcpy(p + NMLN * 3, "PS4Delta");     // version
  std::strcpy(p + NMLN * 4, "x86_64");       // machine
  return 0;
}

int PS4ABI sys_gethostname(char* buf, u32 len) {
  if (!buf || len == 0)
    return -SysError::eINVAL;
  static const char kHost[] = "ps4";
  u32 n = sizeof(kHost);  // includes NUL
  if (n > len)
    n = len;
  std::memcpy(buf, kHost, n);
  buf[n - 1] = '\0';
  return 0;
}

int PS4ABI sys_sethostname(const char* name, u32 len) {
  return 0;
}

int PS4ABI sys_getdtablesize() {
  return static_cast<int>(kMaxFiles);
}

// We deliver no real signals; log the attempt and pretend it landed.
int PS4ABI sys_kill(u32 pid, int sig) {
  BASE_LOGI("procid", "kill(pid={}, sig={}) ignored", pid, sig);
  return 0;
}

// FreeBSD sigset_t is 16 bytes (4x uint32). Nothing is ever pending.
int PS4ABI sys_sigpending(void* set) {
  if (set)
    std::memset(set, 0, 16);
  return 0;
}

// stack_t{void* ss_sp; size_t ss_size; int ss_flags} is ~24 bytes. We model no
// alternate signal stack, so report "none installed" via a zeroed old.
int PS4ABI sys_sigaltstack(const void* ss, void* oss) {
  if (oss)
    std::memset(oss, 0, 24);
  return 0;
}

// No signal ever arrives, so the timed/blocking waits report "nothing pending"
// rather than hanging forever.
int PS4ABI sys_sigtimedwait(const void* set, void* info, const void* timeout) {
  BASE_LOGI("procid", "sigtimedwait -> EAGAIN (no signals)");
  return -SysError::eAGAIN;
}
int PS4ABI sys_sigwaitinfo(const void* set, void* info) {
  BASE_LOGI("procid", "sigwaitinfo -> EAGAIN (no signals)");
  return -SysError::eAGAIN;
}

int PS4ABI sys_sigwait(const void* set, int* sig) {
  if (sig)
    *sig = 0;
  return 0;
}

// BSD sigsuspend always returns EINTR once a handler would have run.
int PS4ABI sys_sigsuspend(const void* sigmask) {
  return -SysError::eINTR;
}

// struct rtprio{uint16 type; uint16 prio}: report RTP_PRIO_NORMAL/prio 0.
int PS4ABI sys_rtprio(int function, u32 pid, void* rtprio) {
  if (rtprio) {
    auto* rp = static_cast<u16*>(rtprio);
    rp[0] = 2;  // RTP_PRIO_NORMAL
    rp[1] = 0;
  }
  return 0;
}

// Single process, no children.
int PS4ABI sys_wait4(u32 pid, int* status, int options, void* rusage) {
  return -SysError::eCHILD;
}

// We run a single process and never freeze it.
int PS4ABI sys_suspend_process() {
  return 0;
}
int PS4ABI sys_resume_process() {
  return 0;
}
int PS4ABI sys_prepare_to_suspend_process() {
  return 0;
}
int PS4ABI sys_prepare_to_resume_process() {
  return 0;
}
int PS4ABI sys_process_terminate() {
  return 0;
}
int PS4ABI sys_suspend_system() {
  return 0;
}

// sys_sandbox_path is a SETTER: the system process hands in the title's sandbox
// root. System ucred only; a game gets EPERM on hardware, and we have no
// per-title jail, so deny exactly as hardware would.
int PS4ABI sys_sandbox_path(const char* path) {
  (void)path;
  return -SysError::ePERM;
}

// The kernel returns boot_parameter(0): 1 on dev/kit firmware, 0 on retail.
// We run retail, so 0 is the accurate answer.
int PS4ABI sys_is_development_mode() {
  return 0;
}

// Reads the SceSelfAuthInfo (0x88 / 136 bytes) from the calling process's SELF
// and copies it to `out`. The first arg is the SELF path; we don't parse SELF
// headers, so we synthesise a non-privileged application identity instead.
int PS4ABI sys_get_self_auth_info(const char* path, void* out) {
  (void)path;
  if (!out)
    return 0;
  std::memset(out, 0, 136);
  auto* p = reinterpret_cast<u64*>(out);
  p[0] = 0x3100000000000001ull;  // auth_id: regular application
  p[2] = 0x2000038000000000ull;  // capability bits
  p[4] = 0x4000400040000000ull;  // attributes / shared
  return 0;
}

// The SDK version the title was compiled against. 5.05 matches kern.sdk_version
// reported via sysctl in sys_info.cc.
int PS4ABI sys_get_sdk_compiled_version() {
  return 0x05050001;
}

int PS4ABI sys_app_state_change() {
  return 0;
}

// Report membership in a single group (gid 1). A zero-length query returns just
// the count.
int PS4ABI sys_getgroups(int gidsetlen, u32* gidset) {
  if (gidsetlen >= 1 && gidset)
    gidset[0] = 1;
  return 1;
}

int PS4ABI sys_setgroups() {
  return 0;
}

int PS4ABI sys_sigqueue() {
  return 0;
}

// The real syscall terminates the process with a diagnostic. We log the message
// and continue rather than killing boot.
int PS4ABI sys_abort2(const char* msg, int nargs, void** args) {
  (void)nargs;
  (void)args;
  BASE_LOGI("abort2", "{}", msg ? msg : "(null)");
  return 0;
}

int PS4ABI sys_exit() {
  guest::RequestStop();
  cpu::ExitGuestThread();
  return 0;
}

int PS4ABI sys_rfork() {
  __builtin_trap();
  return 0;
}

int PS4ABI sys_execve() {
  __builtin_trap();
  return 0;
}

// We deliver no signals, so the mask is inert; still, a caller that saves the
// old mask here to restore it later must not read uninitialised memory. The
// FreeBSD sigset_t is 16 bytes (4x uint32). Report an empty old mask.
int PS4ABI sys_sigprocmask(int how, const int* set, int* oset) {
  (void)how;
  (void)set;
  if (oset)
    std::memset(oset, 0, 16);
  return 0;
}

// Likewise report "no previous handler" rather than leaving the caller's oldact
// buffer uninitialised. struct sigaction on amd64 is 32 bytes (handler pointer
// + flags + 16-byte sa_mask, padded).
int PS4ABI sys_sigaction(int sig, const void* act, void* oact) {
  (void)sig;
  (void)act;
  if (oact)
    std::memset(oact, 0, 32);
  return 0;
}

int PS4ABI sys_getpid() {
  return 0x1337;
}

}  // namespace kern
