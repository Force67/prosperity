/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// The Prospero syscall table. A PS5 process routes here (never the Orbis
// table). Prospero shares syscall numbers 0x000..0x2a4 with Orbis (both are
// Sony-customized FreeBSD), so those reuse the shared, proven handlers via
// lv2_get(). Prospero then diverges: 0x2a5 is sys_wait6 (Orbis had
// get_phys_page_size there, which PS5 moved to 0x2c1), followed by the new
// FreeBSD-11 / Sony additions 0x2a6..0x2d2. Those are named + stubbed below from
// the authoritative PS4/PS5 syscall map. Enumerate with DELTA_PS5_SYSTRACE.

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstring>
#include <set>

#include "kern/lv2/dispatch.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/ps5/sys_info.h"
#include "kern/lv2/sys_info.h"
#include "kern/module.h"
#include "kern/proc.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kPs5SysTrace, "DELTA_PS5_SYSTRACE", false);
}  // namespace

namespace krnl {

// stub handlers (BSD convention applied by the trampoline)
static int PS4ABI ps5_ok() { return 0; }
static int PS4ABI ps5_nosys() { return -SysError::eNOSYS; }
static int PS4ABI ps5_get_phys_page_size() { return 0x4000; } // 16 KiB base page

// Kind of stub for each Prospero-specific syscall.
enum ps5Kind { PK_OK, PK_NOSYS, PK_PAGESIZE };

struct ps5Sys {
  const char *name;
  ps5Kind kind;
};

// Prospero additions past the shared table, indexed by (sid - kPs5Base). Names
// from the PS4/PS5 syscall map; OK (return 0) for advisory/ctrl/notify calls,
// NOSYS for fd-returning POSIX calls we don't implement (so the caller errors
// instead of using a bogus fd) and unassigned numbers, PAGESIZE for 0x2c1.
constexpr u32 kPs5Base = 0x2a5; // 677
static const ps5Sys kPs5Extra[] = {
    {"wait6", PK_OK},                     // 0x2a5
    {"cap_rights_limit", PK_OK},          // 0x2a6
    {"cap_ioctls_limit", PK_OK},          // 0x2a7
    {"cap_ioctls_get", PK_OK},            // 0x2a8
    {"cap_fcntls_limit", PK_OK},          // 0x2a9
    {"cap_fcntls_get", PK_OK},            // 0x2aa
    {"bindat", PK_OK},                    // 0x2ab
    {"connectat", PK_OK},                 // 0x2ac
    {"chflagsat", PK_OK},                 // 0x2ad
    {"accept4", PK_NOSYS},                // 0x2ae
    {"pipe2", PK_NOSYS},                  // 0x2af
    {"aio_mlock", PK_OK},                 // 0x2b0
    {"procctl", PK_OK},                   // 0x2b1
    {"ppoll", PK_NOSYS},                  // 0x2b2
    {"futimens", PK_OK},                  // 0x2b3
    {"utimensat", PK_OK},                 // 0x2b4
    {"numa_getaffinity", PK_OK},          // 0x2b5
    {"numa_setaffinity", PK_OK},          // 0x2b6
    {"number695", PK_NOSYS},              // 0x2b7
    {"number696", PK_NOSYS},              // 0x2b8
    {"number697", PK_NOSYS},              // 0x2b9
    {"number698", PK_NOSYS},              // 0x2ba
    {"number699", PK_NOSYS},              // 0x2bb
    {"apr_submit", PK_OK},                // 0x2bc
    {"apr_resolve", PK_OK},               // 0x2bd
    {"apr_stat", PK_OK},                  // 0x2be
    {"apr_wait", PK_OK},                  // 0x2bf
    {"apr_ctrl", PK_OK},                  // 0x2c0
    {"get_phys_page_size", PK_PAGESIZE},  // 0x2c1
    {"begin_app_mount", PK_OK},           // 0x2c2
    {"end_app_mount", PK_OK},             // 0x2c3
    {"fsc2h_ctrl", PK_OK},                // 0x2c4
    {"streamwrite", PK_OK},               // 0x2c5
    {"app_save", PK_OK},                  // 0x2c6
    {"app_restore", PK_OK},               // 0x2c7
    {"saved_app_delete", PK_OK},          // 0x2c8
    {"get_ppr_sdk_compiled_version", PK_OK}, // 0x2c9
    {"notify_app_event", PK_OK},          // 0x2ca
    {"ioreq", PK_OK},                     // 0x2cb
    {"openintr", PK_OK},                  // 0x2cc
    {"dl_get_info_2", PK_OK},             // 0x2cd
    {"acinfo_add", PK_OK},                // 0x2ce
    {"acinfo_delete", PK_OK},             // 0x2cf
    {"acinfo_get_all_for_coredump", PK_OK}, // 0x2d0
    {"ampr_ctrl_debug", PK_OK},           // 0x2d1
    {"workspace_ctrl", PK_OK},            // 0x2d2
};

// dynlib_get_obj_member index 8 (module param): the PS5 kernel reports the
// param with the SDK dword at +0x10 rewritten to the process' compiled SDK
// version. libSceSysmodule's load probe XOR-compares that dword against
// sceKernelGetCompiledSdkVersion (top 16 bits must match) and unloads the
// module with 0x80020063 on mismatch, so handing out the raw firmware value
// (0x11090001 on 08.40) fails every probe: Demon's Souls then panics with
// "Required cell system module(s) could not be loaded".
static int PS4ABI ps5_dynlib_get_obj_member(u32 handle, u8 index,
                                            void **value) {
  if (index != 8)
    return sys_dynlib_get_obj_member(handle, index, value);
  auto *proc = proc::getActive();
  auto mod = proc->getModule(handle);
  if (!mod)
    return -SysError::eSRCH;
  auto &info = mod->getInfo();
  if (info.moduleParam && info.moduleParamSize >= 0x14)
    *reinterpret_cast<u32 *>(info.moduleParam + 0x10) = proc->getSdkVersion();
  *value = info.moduleParam;
  return 0;
}

// kern.proc.35 is sceKernelGetAppInfo. PS5 libkernel asks for 0x58 bytes, 16
// more than the PS4 struct the shared handler zeroes, so the tail stays caller
// stack garbage. libSceAgcDriver and libSceGnmDriver read the app mode from
// +0x48 -- the first byte past the PS4 struct -- and anything above 2 leaves
// the GPU trap handler unregistered ("Failed to get trap hanlder code", and on
// older firmware "No trap hanlder for mode 224"). Zero what the caller asked
// for; the PS4 path keeps the shorter fill.
static const ps5Sys *ps5Extra(u32 sid) {
  if (sid < kPs5Base)
    return nullptr;
  u32 i = sid - kPs5Base;
  return i < (sizeof(kPs5Extra) / sizeof(kPs5Extra[0])) ? &kPs5Extra[i]
                                                        : nullptr;
}

uintptr_t lv2_get_ps5(u32 sid) {
  const ps5Sys *ex = ps5Extra(sid);

  static std::set<u32> seen;
  if (kPs5SysTrace && seen.insert(sid).second) {
    const char *name = ex ? ex->name : syscall_getname(sid);
    BASE_LOGI("ps5sys", "{:4}  {}", sid, name ? name : "?");
  }

  if (ex) {
    const void *h = ex->kind == PK_PAGESIZE
                        ? reinterpret_cast<const void *>(&ps5_get_phys_page_size)
                    : ex->kind == PK_NOSYS
                        ? reinterpret_cast<const void *>(&ps5_nosys)
                        : reinterpret_cast<const void *>(&ps5_ok);
    return lv2_trampoline(h, sid);
  }
  // Beyond the mapped range (newer firmware additions): succeed silently.
  if (sid > kPs5Base + sizeof(kPs5Extra) / sizeof(kPs5Extra[0]))
    return lv2_trampoline(reinterpret_cast<const void *>(&ps5_ok), sid);

  // sysctl: PS5 widens the kern.proc.35 reply (see handler).
  if (sid == 202)
    return lv2_trampoline(reinterpret_cast<const void *>(&ps5_sysctl), sid);

  // cpuset_getaffinity: a PS5 grants seven cores, not the PS4's six.
  if (sid == 487)
    return lv2_trampoline(
        reinterpret_cast<const void *>(&ps5_cpuset_getaffinity), sid);

  // dynlib_get_obj_member: PS5 virtualizes the module param (see handler).
  if (sid == 649)
    return lv2_trampoline(
        reinterpret_cast<const void *>(&ps5_dynlib_get_obj_member), sid);

  // Base FreeBSD/Orbis syscall: reuse the shared handler + trampoline.
  return lv2_get(sid);
}
} // namespace krnl
