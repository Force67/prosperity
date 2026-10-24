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
#include "host_memory/host_memory.h"

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

// A title's identity is fixed: naming the current id (or -1 where the call
// allows it) changes nothing, any other needs a privilege it does not have.
static bool KeepsId(u32 id, u32 current) {
  return id == current || id == static_cast<u32>(-1);
}

int PS4ABI sys_setuid(u32 uid) {
  return uid == sys_getuid() ? 0 : -SysError::ePERM;
}
int PS4ABI sys_seteuid(u32 uid) {
  return uid == sys_getuid() ? 0 : -SysError::ePERM;
}
int PS4ABI sys_setgid(u32 gid) {
  return gid == sys_getgid() ? 0 : -SysError::ePERM;
}
int PS4ABI sys_setegid(u32 gid) {
  return gid == sys_getgid() ? 0 : -SysError::ePERM;
}
int PS4ABI sys_setresuid(u32 ruid, u32 euid, u32 suid) {
  const u32 uid = sys_getuid();
  return KeepsId(ruid, uid) && KeepsId(euid, uid) && KeepsId(suid, uid)
             ? 0
             : -SysError::ePERM;
}
int PS4ABI sys_setresgid(u32 rgid, u32 egid, u32 sgid) {
  const u32 gid = sys_getgid();
  return KeepsId(rgid, gid) && KeepsId(egid, gid) && KeepsId(sgid, gid)
             ? 0
             : -SysError::ePERM;
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
  return -SysError::ePERM;
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

// struct rtprio{uint16 type; uint16 prio}: report RTP_PRIO_NORMAL/prio 0.
int PS4ABI sys_rtprio(int function, u32 pid, void* rtprio) {
  if (rtprio) {
    auto* rp = static_cast<u16*>(rtprio);
    rp[0] = 2;  // RTP_PRIO_NORMAL
    rp[1] = 0;
  }
  return 0;
}

// Reaping children is for system processes; the kernel refuses a title.
int PS4ABI sys_wait4(u32 pid, int* status, int options, void* rusage) {
  return -SysError::ePERM;
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

// Logs why and ends the process with SIGABRT, which no handler sees.
int PS4ABI sys_abort2(const char* msg, int nargs, void** args) {
  constexpr int kSigAbrt = 6;
  const bool readable = msg && host_memory::IsMemoryRangeMapped(msg, 1);
  BASE_LOGI("abort2", "{}", readable ? msg : "(null)");
  if (nargs > 0 && nargs <= 16 && args &&
      host_memory::IsMemoryRangeMapped(args, sizeof(void*) * nargs))
    for (int i = 0; i < nargs; i++)
      BASE_LOGI("abort2", "  arg{} = {:p}", i, args[i]);
  TerminateBySignal(kSigAbrt);
  return nargs < 0 || nargs > 16 ? -SysError::eINVAL : 0;
}

int PS4ABI sys_exit(int code) {
  BASE_LOGI("exit", "guest exit({})", code);
  guest::RequestStop();
  cpu::ExitGuestThread();
  return 0;
}

// Creating and replacing processes is for system processes; the kernel refuses
// a title.
int PS4ABI sys_rfork() {
  return -SysError::ePERM;
}

int PS4ABI sys_execve() {
  return -SysError::ePERM;
}

int PS4ABI sys_getpid() {
  return 0x1337;
}

int PS4ABI sys_setreuid(u32 ruid, u32 euid) {
  const u32 uid = sys_getuid();
  return KeepsId(ruid, uid) && KeepsId(euid, uid) ? 0 : -SysError::ePERM;
}

int PS4ABI sys_setregid(u32 rgid, u32 egid) {
  const u32 gid = sys_getgid();
  return KeepsId(rgid, gid) && KeepsId(egid, gid) ? 0 : -SysError::ePERM;
}

// Profiling arms a tick-driven histogram; nothing ever samples, so only the
// argument check is real.
int PS4ABI sys_profil(void* /*samples*/,
                      size_t /*size*/,
                      size_t /*offset*/,
                      u32 scale) {
  return scale > 0x10000 ? -SysError::eINVAL : 0;
}

}  // namespace kern
