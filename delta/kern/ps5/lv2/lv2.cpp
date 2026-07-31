/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// PS5 (Prospero) syscall dispatch. A PS5 process routes here (never the PS4
// table). Prospero shares syscall numbers 0x000..0x2a4 with Orbis (both are
// Sony-customized FreeBSD), so those reuse the shared, proven handlers via
// lv2_get(). Prospero then diverges: 0x2a5 is sys_wait6 (Orbis had
// get_phys_page_size there, which PS5 moved to 0x2c1), followed by the new
// FreeBSD-11 / Sony additions 0x2a6..0x2d2. Those are named + stubbed below from
// the authoritative PS4/PS5 syscall map. Enumerate with DELTA_PS5_SYSTRACE.

#include <base.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>

#include "kern/ps4/lv2/error_table.h"
#include "lv2.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kPs5SysTrace, "DELTA_PS5_SYSTRACE", false);
}  // namespace

namespace krnl {
const char *syscall_getname(uint32_t idx);
uintptr_t lv2_get(uint32_t sid);                        // shared base handlers
uintptr_t lv2_trampoline(const void *handler, uint32_t sid);

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
constexpr uint32_t kPs5Base = 0x2a5; // 677
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

static const ps5Sys *ps5Extra(uint32_t sid) {
  if (sid < kPs5Base)
    return nullptr;
  uint32_t i = sid - kPs5Base;
  return i < (sizeof(kPs5Extra) / sizeof(kPs5Extra[0])) ? &kPs5Extra[i]
                                                        : nullptr;
}

uintptr_t lv2_get_ps5(uint32_t sid) {
  const ps5Sys *ex = ps5Extra(sid);

  static std::set<uint32_t> seen;
  if (kPs5SysTrace && seen.insert(sid).second) {
    const char *name = ex ? ex->name : syscall_getname(sid);
    std::fprintf(stderr, "[ps5sys] %4u  %s\n", sid, name ? name : "?");
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

  // Base FreeBSD/Orbis syscall: reuse the shared handler + trampoline.
  return lv2_get(sid);
}
} // namespace krnl
