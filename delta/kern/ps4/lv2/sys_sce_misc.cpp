
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

#include <atomic>
#include <cstdio>
#include <cstring>

#include "error_table.h"
#include "sys_mem.h"
#include "sys_sce_misc.h"
#include "../../proc.h"
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
    std::printf("[sce] %s\n", msg);
}

// The JIT shm object the guest later mmaps to hold generated code. The real
// syscall returns an fd; we return an RWX anonymous region instead, so a guest
// that maps the result by fd (mmap(fd)) will NOT see this region. Loud once
// because that mismatch breaks runtime code generation if a title relies on it.
int64_t PS4ABI sys_jitshm_create(size_t len, uint32_t flags) {
  (void)flags;
  static std::atomic<bool> once{false};
  logOnce(once, "jitshm_create returns a raw RWX region, not an fd; JIT may break");
  return (int64_t)sys_mmap(nullptr, len, 7 /*rwx*/, 0x1000 /*anon*/, -1, 0);
}

int PS4ABI sys_jitshm_alias() { return -SysError::eOPNOTSUPP; }

// The kernel's sys_dl_get_list is gated on a debugger/coredump/syscore process;
// anything else -- a retail game title -- gets 1 (EPERM). Same for get_info
// below: the debugger (the system process running dbglogger) is the only caller
// that may enumerate loaded modules. arg block is
// {pid@0, ids[]@8, max@16, count@24} in the kernel.
int PS4ABI sys_dl_get_list() { return -SysError::ePERM; }

// Kernel arg block: {pid@0, handle@8, info@16}; fills a 0xA50-byte module-info
// struct. Same debugger gate, same EPERM for a title.
int PS4ABI sys_dl_get_info() { return -SysError::ePERM; }

// The kernel's sys_dl_notify_event returns ENOSYS unconditionally -- dynlib
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

// Budget objects gate flexible memory / resource pools. We don't enforce them,
// but the guest stores the id and passes it back, so hand out a fixed non-zero
// id. Logged once: if a title's allocations are actually capped by a budget we
// granted unconditionally, that overcommit shows up here first.
//
// Kernel budget_create args: {char* name@0; uint32 ptype@8 (0..3);
// SceBudgetResource* res@16; uint32 nres@24 (0..10); SceBudgetResource* resOut@32}.
// The budget object (64 bytes) holds the name, resource array, ptype, nres, and
// an open-count. Only a system ucred may create one (returns 78 otherwise).
int PS4ABI sys_budget_create() {
  static std::atomic<bool> once{false};
  logOnce(once, "budget_create granted unconditionally (no enforcement)");
  return 0x2001;
}
int PS4ABI sys_budget_delete() { return 0; }
// The libkernel_sys wrappers (stub at 0x25c0) forward the args straight to the
// syscall; the kernel fills a caller-provided buffer that no libkernel function
// reads back, so the out-struct layout isn't recoverable from libkernel alone.
// Callers pre-zero that buffer, so success leaves it reading as an all-zero
// budget. We return success rather than an errno: with the carry flag cleared,
// a negative return would drive the wrapper's errno path on a stale errno.
int PS4ABI sys_budget_get() { return 0; }
int PS4ABI sys_budget_set() { return 0; }
// Kernel (sys_budget_getid): only a system-ucred process may ask -- it gets the
// proc's own budget id, or 2/ENOENT when none is set -- and everyone else gets
// 78 (ENOSYS). A game is the latter, but the game's libkernel wrapper takes the
// id and passes it back to budget_get/delete, so keep the benign fixed id the
// wrapper path expects rather than turning the call into an error the title was
// never coded to handle.
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

// Event-port objects for kqueue-style delivery. We don't route events through
// them, so return a fixed handle and swallow trigger/delete. Logged once: a
// title waiting on an eport event we never deliver would stall, and this is the
// trace that explains it.
//
// Kernel eport object (~0x60 bytes): name (32 bytes), mtx, cv, waiter list,
// open-count, attr. eport_trigger sets the pattern and broadcasts cv; waiters
// wake and read the triggered pattern. Like evf/osem, named eports can be
// shared across processes via the global name table (attr bit 0x100).
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

// The kernel's sys_sandbox_path is a SETTER, not a getter: the system process
// hands the title's sandbox-root string in and the kernel stores it in the
// proc. Only a system ucred may do that; a game gets 1 (EPERM). We have no
// per-title jail (the mount table is the sandbox) and the game is not the
// system, so deny exactly the way hardware would.
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
  auto *p = reinterpret_cast<uint64_t *>(out);
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
int64_t PS4ABI sys_blockpool_map(int64_t pool, size_t len, uint32_t prot,
                                 uint32_t flags) {
  (void)pool;
  (void)prot;
  (void)flags;
  static std::atomic<bool> once{false};
  logOnce(once, "blockpool_map backs the pool with a plain anon region");
  return (int64_t)sys_mmap(nullptr, len, 3 /*rw*/, 0x1000 /*anon*/, -1, 0);
}

int PS4ABI sys_blockpool_unmap() { return 0; }
int64_t PS4ABI sys_blockpool_batch(uint64_t a0, uint64_t a1, uint64_t a2,
                                   uint64_t a3, uint64_t a4, uint64_t a5) {
  if (kBlockpoolTrace) {
    std::fprintf(stderr, "[blockpool_batch] a0=%#llx a1=%#llx a2=%#llx a3=%#llx a4=%#llx a5=%#llx\n",
                 (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                 (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    // a1 commonly points at the command array; dump a few 64-bit words.
    if (a1 > 0x10000) {
      auto *w = reinterpret_cast<uint64_t *>(a1);
      std::fprintf(stderr, "  cmd[0..7]: %#llx %#llx %#llx %#llx %#llx %#llx %#llx %#llx\n",
                   (unsigned long long)w[0], (unsigned long long)w[1], (unsigned long long)w[2],
                   (unsigned long long)w[3], (unsigned long long)w[4], (unsigned long long)w[5],
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
    std::fprintf(stderr, "[aio] unsupported call #%d\n", c);
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
int PS4ABI sys_getgroups(int gidsetlen, uint32_t *gidset) {
  if (gidsetlen >= 1 && gidset)
    gidset[0] = 1;
  return 1;
}

int PS4ABI sys_setgroups() { return 0; }

int PS4ABI sys_setpriority() { return 0; }
int PS4ABI sys_getpriority() { return 0; }

int PS4ABI sys_setsockopt() { return 0; }

int PS4ABI sys_getsockopt(int fd, int level, int name, void *val,
                          uint32_t *len) {
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

// pathconf/fpathconf/lpathconf: report the configurable limit for `name`. We
// return concrete values rather than the -1 "indeterminate" sentinel: a -1 in
// rax is indistinguishable from an errno under the syscall return convention, so
// a caller sizing a buffer against it would misread a failure. Values match
// FreeBSD's defaults for a UFS-like filesystem.
static int64_t pathconf_value(int name) {
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
  std::printf("[abort2] %s\n", msg ? msg : "(null)");
  return 0;
}

int PS4ABI sys_thr_sleep() { return 0; }
int PS4ABI sys_thr_wakeup() { return 0; }

int PS4ABI sys_posix_fallocate() { return 0; }
int PS4ABI sys_posix_fadvise() { return 0; }

} // namespace krnl
