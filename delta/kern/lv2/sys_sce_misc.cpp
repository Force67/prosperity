
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>

#include <atomic>
#include <cstdio>
#include <cstring>

#include "error_table.h"
#include "sys_mem.h"
#include "sys_sce_misc.h"
#include "kern/proc.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kAioTrace, "DELTA_AIO_TRACE", false);
DELTA_OPTION(bool, kBlockpoolTrace, "DELTA_BLOCKPOOL_TRACE", false);
}  // namespace

namespace krnl {
// From sys_budget.cpp: the proc telemetry "process type". Mirrored so per-budget
// ptype queries agree with sys_get_proc_type_info.
extern int sys_budget_get_ptype();

// Log a message the first time a given handler runs. Used for the stubs whose
// fakery would silently break if a title actually exercised the subsystem.
static void logOnce(std::atomic<bool> &flag, const char *msg) {
  if (!flag.exchange(true))
    BASE_LOGI("sce", "{}", msg);
}

// The JIT shm object the guest later mmaps to hold generated code. The real
// syscall returns an fd; we return an RWX anonymous region instead, so a guest
// that maps the result by fd (mmap(fd)) will NOT see this region. Loud once
// because that mismatch breaks runtime code generation if a title relies on it.
i64 PS4ABI sys_jitshm_create(size_t len, u32 flags) {
  (void)flags;
  static std::atomic<bool> once{false};
  logOnce(once, "jitshm_create returns a raw RWX region, not an fd; JIT may break");
  return (i64)sys_mmap(nullptr, len, 7 /*rwx*/, 0x1000 /*anon*/, -1, 0);
}

int PS4ABI sys_jitshm_alias() { return -SysError::eOPNOTSUPP; }

// sys_dl_get_list/get_info: gated on a debugger/coredump/syscore process; a retail
// title gets EPERM (the dbglogger system process is the only legitimate enumerator).
// Arg block {pid@0, ids[]@8, max@16, count@24}.
int PS4ABI sys_dl_get_list() { return -SysError::ePERM; }

// Kernel arg block: {pid@0, handle@8, info@16}; fills a 0xA50-byte module-info
// struct. Same debugger gate, same EPERM for a title.
int PS4ABI sys_dl_get_info() { return -SysError::ePERM; }

// The kernel's sys_dl_notify_event returns ENOSYS unconditionally: dynlib
// event delivery to a debugger is not wired on the console either.
int PS4ABI sys_dl_notify_event() { return -SysError::eNOSYS; }

// The kernel stores the debugger protocol version once and answers EBUSY to any
// repeat init, so a second sys_debug_init from the guest is already an error we
// mirror rather than fake away.
int PS4ABI sys_debug_init(const int *version) {
  static std::atomic<bool> once{false};
  if (!version || once.exchange(true))
    return -SysError::eBUSY;
  return 0;
}

// We run a single process and never freeze it.
int PS4ABI sys_suspend_process() { return 0; }
int PS4ABI sys_resume_process() { return 0; }
int PS4ABI sys_prepare_to_suspend_process() { return 0; }
int PS4ABI sys_prepare_to_resume_process() { return 0; }
int PS4ABI sys_process_terminate() { return 0; }
int PS4ABI sys_suspend_system() { return 0; }

// Hardware performance counters: no PMU, so every op is a silent success and
// readers see idle counters.
int PS4ABI sys_opmc_enable() { return 0; }
int PS4ABI sys_opmc_disable() { return 0; }
int PS4ABI sys_opmc_set_ctl() { return 0; }
int PS4ABI sys_opmc_set_ctr() { return 0; }
int PS4ABI sys_opmc_get_ctr() { return 0; }
int PS4ABI sys_opmc_set_hw() { return 0; }
int PS4ABI sys_opmc_get_hw() { return 0; }

// Budget objects gate flexible memory / resource pools; unenforced, but the guest
// stores the id and passes it back, so hand out a fixed non-zero id. Logged once:
// an overcommit shows up here first. Args {name@0, ptype@8 (0..3), res@16,
// nres@24 (0..10), resOut@32}; object = 64 bytes; system ucred only (else 78).
int PS4ABI sys_budget_create() {
  static std::atomic<bool> once{false};
  logOnce(once, "budget_create granted unconditionally (no enforcement)");
  return 0x2001;
}
int PS4ABI sys_budget_delete() { return 0; }
// The libkernel_sys wrappers forward args straight through; the kernel fills a buffer
// no libkernel function reads back (layout unrecoverable), and callers pre-zero it,
// so success reads as an all-zero budget. Return success, not an errno: with carry
// clear, a negative return would drive the wrapper's errno path on a stale errno.
int PS4ABI sys_budget_get() { return 0; }
int PS4ABI sys_budget_set() { return 0; }
// sys_budget_getid: system-ucred only (a game gets 78 on hardware), but the game's
// wrapper passes the id back to budget_get/delete, so keep the benign fixed id
// rather than an error the title was never coded to handle.
int PS4ABI sys_budget_getid() { return 0x2001; }
int PS4ABI sys_budget_get_ptype_of_budget() { return sys_budget_get_ptype(); }

int PS4ABI sys_mdbg_call() { return 0; }

// Sony "system block" critical sections. We don't arbitrate them, so enter/exit
// are uncontended no-ops.
int PS4ABI sys_sblock_create() { return 0; }
int PS4ABI sys_sblock_delete() { return 0; }
int PS4ABI sys_sblock_enter() { return 0; }
int PS4ABI sys_sblock_exit() { return 0; }
int PS4ABI sys_sblock_xenter() { return 0; }
int PS4ABI sys_sblock_xexit() { return 0; }

// Event-port objects for kqueue-style delivery; unrouted, so return a fixed handle
// and swallow trigger/delete. Logged once: a title waiting on an eport event we
// never deliver stalls, and this trace explains it. Kernel eport (~0x60 bytes):
// name, mtx, cv, waiter list, open-count, attr; trigger sets the pattern and
// broadcasts; named eports share across processes (attr bit 0x100).
int PS4ABI sys_eport_create() {
  static std::atomic<bool> once{false};
  logOnce(once, "eport_create returns a fake handle; events are never delivered");
  return 0x3001;
}
int PS4ABI sys_eport_delete() { return 0; }
int PS4ABI sys_eport_trigger() { return 0; }
int PS4ABI sys_eport_open() { return 0x3001; }
int PS4ABI sys_eport_close() { return 0; }

int PS4ABI sys_dynlib_dlclose() { return 0; }
int PS4ABI sys_dynlib_prepare_dlclose() { return 0; }

// sys_sandbox_path is a SETTER: the system process hands in the title's sandbox root.
// System ucred only; a game gets EPERM on hardware, and we have no per-title jail,
// so deny exactly as hardware would.
int PS4ABI sys_sandbox_path(const char *path) {
  (void)path;
  return -SysError::ePERM;
}

// dup a descriptor into another process; no multi-proc, so deny.
int PS4ABI sys_rdup() { return -SysError::eOPNOTSUPP; }

// Same debugger gate as sys_dl_get_list/get_info; arg block is
// {pid@0, handle@8, meta@16, metasize@24, sizeOut@32}.
int PS4ABI sys_dl_get_metadata() { return -SysError::ePERM; }

// The kernel returns boot_parameter(0): 1 on dev/kit firmware, 0 on retail.
// We run retail, so 0 is the accurate answer.
int PS4ABI sys_is_development_mode() { return 0; }

// Reads the SceSelfAuthInfo (0x88 / 136 bytes) from the calling process's SELF
// and copies it to `out`. The first arg is the SELF path; we don't parse SELF
// headers, so we synthesise a non-privileged application identity instead.
int PS4ABI sys_get_self_auth_info(const char *path, void *out) {
  (void)path;
  if (!out)
    return 0;
  std::memset(out, 0, 136);
  auto *p = reinterpret_cast<u64 *>(out);
  p[0] = 0x3100000000000001ull; // auth_id: regular application
  p[2] = 0x2000038000000000ull; // capability bits
  p[4] = 0x4000400040000000ull; // attributes / shared
  return 0;
}

int PS4ABI sys_get_paging_stats_of_all_threads() { return 0; }
int PS4ABI sys_get_paging_stats_of_all_objects() { return 0; }

int PS4ABI sys_get_resident_count() { return 0; }
int PS4ABI sys_get_resident_fmem_count() { return 0; }

// General-purpose output (debug LEDs/pins). No hardware; accept and ignore.
int PS4ABI sys_set_gpo() { return 0; }
int PS4ABI sys_get_gpo() { return 0; }

int PS4ABI sys_test_debug_rwmem() { return -SysError::eOPNOTSUPP; }

int PS4ABI sys_get_cpu_usage_all() { return 0; }
int PS4ABI sys_get_cpu_usage_proc() { return 0; }

// Physically-contiguous shared memory; we don't back it, so deny and let the
// guest fall back to ordinary memory.
int PS4ABI sys_physhm_open() { return -SysError::eOPNOTSUPP; }
int PS4ABI sys_physhm_unlink() { return 0; }

int PS4ABI sys_resume_internal_hdd() { return 0; }

int PS4ABI sys_set_phys_fmem_limit() { return 0; }

int PS4ABI sys_set_uevt() { return 0; }

int PS4ABI sys_set_chicken_switches() { return 0; }

int PS4ABI sys_unk645() { return 0; }

int PS4ABI sys_get_kernel_mem_statistics(void *out) {
  if (out)
    std::memset(out, 0, 0x40);
  return 0;
}

// The SDK version the title was compiled against. 5.05 matches kern.sdk_version
// reported via sysctl in sys_info.cpp.
int PS4ABI sys_get_sdk_compiled_version() { return 0x05050001; }

int PS4ABI sys_app_state_change() { return 0; }

// Carve a mapping out of a block pool. We don't model pools, so back it with a
// plain anonymous RW region of the requested length. Logged once because the
// pool handle and any accounting it implies are ignored.
i64 PS4ABI sys_blockpool_map(i64 pool, size_t len, u32 prot,
                                 u32 flags) {
  (void)pool;
  (void)prot;
  (void)flags;
  static std::atomic<bool> once{false};
  logOnce(once, "blockpool_map backs the pool with a plain anon region");
  return (i64)sys_mmap(nullptr, len, 3 /*rw*/, 0x1000 /*anon*/, -1, 0);
}

int PS4ABI sys_blockpool_unmap() { return 0; }
i64 PS4ABI sys_blockpool_batch(u64 a0, u64 a1, u64 a2,
                                   u64 a3, u64 a4, u64 a5) {
  if (kBlockpoolTrace) {
    BASE_LOGI("blockpool_batch",
              "a0={:#x} a1={:#x} a2={:#x} a3={:#x} a4={:#x} a5={:#x}",
              (unsigned long long)a0, (unsigned long long)a1,
              (unsigned long long)a2, (unsigned long long)a3,
              (unsigned long long)a4, (unsigned long long)a5);
    // a1 commonly points at the command array; dump a few 64-bit words.
    if (a1 > 0x10000) {
      auto *w = reinterpret_cast<u64 *>(a1);
      BASE_LOGI("blockpool_batch",
                "  cmd[0..7]: {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
                (unsigned long long)w[0], (unsigned long long)w[1],
                (unsigned long long)w[2], (unsigned long long)w[3],
                (unsigned long long)w[4], (unsigned long long)w[5],
                (unsigned long long)w[6], (unsigned long long)w[7]);
    }
  }
  return 0;
}

int PS4ABI sys_dynlib_get_info_for_libdbg() { return 0; }
int PS4ABI sys_dynlib_get_list_for_libdbg() { return 0; }
int PS4ABI sys_dynlib_get_list2() { return 0; }
int PS4ABI sys_dynlib_get_info2() { return 0; }
int PS4ABI sys_get_page_table_stats() { return 0; }

// We don't model async IO. Failing with eOPNOTSUPP makes guests fall back to
// synchronous IO. Every AIO entry point funnels here, so the log can't name
// which one; pair it with FEX_SCTRACE to attribute the call.
int PS4ABI sys_aio_unsupported() {
  static std::atomic<int> n{0};
  int c = ++n;
  if (kAioTrace && c <= 200)
    BASE_LOGI("aio", "unsupported call #{}", c);
  else {
    static std::atomic<bool> once{false};
    logOnce(once, "aio unsupported; guest should fall back to sync IO");
  }
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_get_bio_usage_all() { return 0; }
int PS4ABI sys_aio_init() { return 0; }

// Report membership in a single group (gid 1). A zero-length query returns just
// the count.
int PS4ABI sys_getgroups(int gidsetlen, u32 *gidset) {
  if (gidsetlen >= 1 && gidset)
    gidset[0] = 1;
  return 1;
}

int PS4ABI sys_setgroups() { return 0; }

int PS4ABI sys_setpriority() { return 0; }
int PS4ABI sys_getpriority() { return 0; }

int PS4ABI sys_setsockopt() { return 0; }

int PS4ABI sys_getsockopt(int fd, int level, int name, void *val,
                          u32 *len) {
  (void)fd;
  (void)level;
  (void)name;
  if (val && len && *len >= 4)
    *reinterpret_cast<int *>(val) = 0;
  return 0;
}

int PS4ABI sys_sync() { return 0; }

// The PS4 uses a 16 KiB page size.
int PS4ABI sys_getpagesize() { return 16384; }

int PS4ABI sys_flock() { return 0; }

int PS4ABI sys_utimes() { return 0; }
int PS4ABI sys_futimes() { return 0; }

// pathconf/fpathconf/lpathconf: concrete values, not the -1 sentinel, which is
// indistinguishable from an errno in rax and would misread as failure when sizing
// a buffer. Values = FreeBSD defaults for a UFS-like filesystem.
static i64 pathconf_value(int name) {
  switch (name) {
  case 1:  return 32767; // _PC_LINK_MAX
  case 2:  return 255;   // _PC_MAX_CANON
  case 3:  return 255;   // _PC_MAX_INPUT
  case 4:  return 255;   // _PC_NAME_MAX
  case 5:  return 1024;  // _PC_PATH_MAX
  case 6:  return 512;   // _PC_PIPE_BUF
  case 7:  return 1;     // _PC_CHOWN_RESTRICTED
  case 8:  return 1;     // _PC_NO_TRUNC
  case 9:  return 255;   // _PC_VDISABLE
  case 11: return 64;    // _PC_ACL_PATH_MAX
  case 12: return 64;    // _PC_FILESIZEBITS -> at least 64-bit offsets
  default: return -SysError::eINVAL;
  }
}
int PS4ABI sys_pathconf(const char *path, int name) {
  (void)path;
  return static_cast<int>(pathconf_value(name));
}
int PS4ABI sys_fpathconf(int fd, int name) {
  (void)fd;
  return static_cast<int>(pathconf_value(name));
}
int PS4ABI sys_lpathconf(const char *path, int name) {
  (void)path;
  return static_cast<int>(pathconf_value(name));
}

int PS4ABI sys_sigqueue() { return 0; }

// The real syscall terminates the process with a diagnostic. We log the message
// and continue rather than killing boot.
int PS4ABI sys_abort2(const char *msg, int nargs, void **args) {
  (void)nargs;
  (void)args;
  BASE_LOGI("abort2", "{}", msg ? msg : "(null)");
  return 0;
}

int PS4ABI sys_thr_sleep() { return 0; }
int PS4ABI sys_thr_wakeup() { return 0; }

int PS4ABI sys_posix_fallocate() { return 0; }
int PS4ABI sys_posix_fadvise() { return 0; }

} // namespace krnl
