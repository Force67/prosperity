/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// The Prospero syscall table, as the 13.42 kernel's native sysent lays it out
// (sv_size 0x2df). A PS5 process routes here, never to the Orbis table. Ids the
// two kernels implement alike reuse the Orbis handler through Lv2Get(); the
// rows below are where the PS5 kernel differs or the id is PS5-only. Enumerate
// what a title calls with DELTA_PS5_SYSTRACE.

#include "base/arch.h"
#include "base/logging.h"
#include "guest/session.h"
#include "guest_abi.h"

#include "base/containers/set.h"
#include "kern/lv2/dispatch.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/ps5/sys_info.h"
#include "kern/lv2/sys_aio.h"
#include "kern/lv2/sys_context.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_info.h"
#include "kern/lv2/sys_signal.h"
#include "kern/lv2/sys_vfs.h"
#include "kern/module.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kPs5SysTrace, "DELTA_PS5_SYSTRACE", false);
}  // namespace

namespace kern {

int PS4ABI Ps5Ok() {
  return 0;
}
// An id the kernel implements but a title cannot reach (or one whose handler is
// unknown): ENOSYS without nosys's SIGSYS.
int PS4ABI Ps5Nosys() {
  return -SysError::eNOSYS;
}
int PS4ABI Ps5Eperm() {
  return -SysError::ePERM;
}
int PS4ABI Ps5Eopnotsupp() {
  return -SysError::eOPNOTSUPP;
}
// A title runs chrooted into its sandbox, which is what the kernel tests.
static int PS4ABI Ps5InSandbox() {
  return 1;
}
static int PS4ABI Ps5Ebadf() {
  return -SysError::eBADF;
}
static int PS4ABI Ps5Einval() {
  return -SysError::eINVAL;
}

// dynlib_get_obj_member index 8 (module param): the PS5 kernel reports the
// param with the SDK dword at +0x10 rewritten to the process' compiled SDK
// version. libSceSysmodule's load probe XOR-compares that dword against
// sceKernelGetCompiledSdkVersion (top 16 bits must match) and unloads the
// module with 0x80020063 on mismatch, so handing out the raw firmware value
// (0x11090001 on 08.40) fails every probe: Demon's Souls then panics with
// "Required cell system module(s) could not be loaded".
static int PS4ABI Ps5DynlibGetObjMember(u32 handle, u8 index, void** value) {
  if (index != 8)
    return sys_dynlib_get_obj_member(handle, index, value);
  auto* proc = Process::GetActive();
  auto mod = proc->GetModule(handle);
  if (!mod)
    return -SysError::eSRCH;
  auto& info = mod->GetInfo();
  if (info.module_param && info.module_param_size >= 0x14)
    *reinterpret_cast<u32*>(info.module_param + 0x10) = proc->GetSdkVersion();
  *value = info.module_param;
  return 0;
}

constexpr u32 kPs5TableSize = 0x2df;

// Ids the kernel routes to nosys (and lkmnosys, which forwards there), which
// raises SIGSYS.
static const u16 kPs5Nosys[] = {
    0,   8,   11,  17,  18,  19,  38,  40,  46,  48,  51,  52,  62,  63,  64,
    67,  68,  71,  72,  76,  77,  84,  87,  88,  91,  94,  103, 107, 108, 109,
    110, 111, 112, 115, 119, 129, 130, 139, 142, 143, 144, 145, 146, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 184,
    185, 186, 187, 193, 197, 198, 199, 200, 201, 205, 208, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 244, 245, 246, 248, 249, 252, 257, 258, 259, 260, 261, 262,
    263, 264, 265, 266, 267, 268, 269, 270, 271, 273, 278, 279, 280, 281, 282,
    283, 284, 285, 286, 287, 288, 291, 292, 293, 294, 295, 296, 297, 298, 299,
    300, 301, 302, 303, 313, 318, 319, 320, 322, 323, 336, 338, 339, 342, 344,
    347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 364, 365, 366,
    367, 368, 369, 370, 371, 372, 373, 375, 376, 377, 380, 381, 382, 383, 398,
    399, 412, 413, 414, 418, 419, 420, 424, 425, 426, 427, 428, 434, 435, 436,
    437, 438, 439, 440, 445, 446, 447, 448, 449, 450, 451, 452, 453, 457, 458,
    459, 460, 461, 462, 463, 467, 468, 469, 470, 471, 472, 473, 474, 489, 492,
    500, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 518, 521, 523,
    524, 530, 531, 537, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584,
    644, 645, 665, 668, 695, 696, 697, 698, 699,
};

struct Ps5Row {
  u16 id;
  const char* name;
  const void* handler;
};

template <typename F>
constexpr const void* H(F* f) {
  return reinterpret_cast<const void*>(f);
}

// Rows that differ from the Orbis handler of the same id. Ps5Eperm marks calls
// the kernel gates on a privilege or a system-process identity a title never
// has; Ps5Nosys on an implemented id marks a handler that answers ENOSYS on a
// retail unit (Capsicum, MAC, rctl, ffclock, the devkit-only kld calls).
static const Ps5Row kPs5Rows[] = {
    {2, "fork", H(&Ps5Eperm)},
    {9, "link", H(&Ps5Eopnotsupp)},
    {14, "mknod", H(&Ps5Eperm)},
    {16, "chown", H(&sys_chmod)},
    {21, "mount", H(&Ps5Eperm)},
    {22, "unmount", H(&Ps5Eperm)},
    {26, "ptrace", H(&Ps5Eperm)},
    {30, "accept", H(&Ps5Accept)},
    {45, "ktrace", H(&Ps5Nosys)},
    {50, "setlogin", H(&Ps5Eperm)},
    {55, "reboot", H(&Ps5Eperm)},
    {56, "revoke", H(&Ps5Eperm)},
    {57, "symlink", H(&Ps5Eopnotsupp)},
    {61, "chroot", H(&Ps5Eperm)},
    {66, "vfork", H(&Ps5Eperm)},
    {69, "sbrk", H(&Ps5Eopnotsupp)},
    {70, "sstk", H(&Ps5Eopnotsupp)},
    {85, "swapon", H(&Ps5Eperm)},
    {101, "netabort", H(&Ps5Netabort)},
    {123, "fchown", H(&sys_fchmod)},
    {125, "netgetiflist", H(&Ps5Ok)},  // no interfaces: zero entries
    {132, "mkfifo", H(&Ps5Eopnotsupp)},
    {202, "__sysctl", H(&Ps5Sysctl)},
    {232, "clock_gettime", H(&Ps5ClockGettime)},
    {234, "clock_getres", H(&Ps5ClockGetres)},
    {241, "ffclock_getcounter", H(&Ps5Nosys)},
    {242, "ffclock_setestimate", H(&Ps5Nosys)},
    {243, "ffclock_getestimate", H(&Ps5Nosys)},
    {247, "clock_getcpuclockid2", H(&Ps5ClockGetcpuclockid2)},
    {254, "lchown", H(&sys_chmod)},
    {274, "lchmod", H(&sys_chmod)},
    {275, "netbsd_lchown", H(&sys_chmod)},
    {276, "lutimes", H(&sys_chmod)},
    {304, "kldload", H(&Ps5Nosys)},
    {305, "kldunload", H(&Ps5Nosys)},
    {306, "kldfind", H(&Ps5Nosys)},
    {307, "kldnext", H(&Ps5Nosys)},
    {308, "kldstat", H(&Ps5Nosys)},
    {309, "kldfirstmod", H(&Ps5Nosys)},
    {335, "utrace", H(&Ps5Ok)},
    {337, "kldsym", H(&Ps5Nosys)},
    {374, "__setugid", H(&Ps5Nosys)},
    {378, "nmount", H(&Ps5Eperm)},
    {379, "mtypeprotect", H(&Ps5Mtypeprotect)},
    {384, "__mac_get_proc", H(&Ps5Nosys)},
    {385, "__mac_set_proc", H(&Ps5Nosys)},
    {386, "__mac_get_fd", H(&Ps5Nosys)},
    {387, "__mac_get_file", H(&Ps5Nosys)},
    {388, "__mac_set_fd", H(&Ps5Nosys)},
    {389, "__mac_set_file", H(&Ps5Nosys)},
    {390, "kenv", H(&Ps5Eperm)},
    {391, "lchflags", H(&sys_chflags)},
    {394, "mac_syscall", H(&Ps5Nosys)},
    {395, "getfsstat", H(&Ps5Ok)},
    {409, "__mac_get_pid", H(&Ps5Nosys)},
    {410, "__mac_get_link", H(&Ps5Nosys)},
    {411, "__mac_set_link", H(&Ps5Nosys)},
    {415, "__mac_execve", H(&Ps5Nosys)},
    // Served through Lv2ContextHandler; the rows only name them.
    {417, "sigreturn", H(&Ps5Nosys)},
    {421, "getcontext", H(&Ps5Nosys)},
    {422, "setcontext", H(&Ps5Nosys)},
    {423, "swapcontext", H(&Ps5Nosys)},
    {430, "thr_create", H(&sys_thr_create)},
    {444, "kldunloadf", H(&Ps5Nosys)},
    {464, "thr_set_name", H(&Ps5ThrSetName)},
    {487, "cpuset_getaffinity", H(&Ps5CpusetGetaffinity)},
    {490, "fchmodat", H(&Ps5Fchmodat)},
    {491, "fchownat", H(&Ps5Fchownat)},
    {494, "futimesat", H(&Ps5Futimesat)},
    {495, "linkat", H(&Ps5Eopnotsupp)},
    {497, "mkfifoat", H(&Ps5Eopnotsupp)},
    {498, "mknodat", H(&Ps5Eperm)},
    {502, "symlinkat", H(&Ps5Eopnotsupp)},
    {515, "__cap_rights_get", H(&Ps5Nosys)},
    {516, "cap_enter", H(&Ps5Nosys)},
    {517, "cap_getmode", H(&Ps5Nosys)},
    {519, "pdkill", H(&Ps5Ebadf)},
    {520, "pdgetpid", H(&Ps5Ebadf)},
    {525, "rctl_get_racct", H(&Ps5Nosys)},
    {526, "rctl_get_rules", H(&Ps5Nosys)},
    {527, "rctl_get_limits", H(&Ps5Nosys)},
    {528, "rctl_add_rule", H(&Ps5Nosys)},
    {529, "rctl_remove_rule", H(&Ps5Nosys)},
    {568, "budget_create", H(&Ps5Nosys)},
    {569, "budget_delete", H(&Ps5Nosys)},
    {570, "budget_get", H(&Ps5Nosys)},
    {571, "budget_set", H(&Ps5Nosys)},
    {572, "virtual_query", H(&Ps5VirtualQuery)},
    {585, "is_in_sandbox", H(&Ps5InSandbox)},
    {588, "mname", H(&Ps5Mname)},
    {589, "dynlib_dlopen", H(&Ps5Nosys)},
    {590, "dynlib_dlclose", H(&Ps5Nosys)},
    {591, "dynlib_dlsym", H(&Ps5DynlibDlsym)},
    {605, "workaround8849", H(&Ps5Workaround8849)},
    {608, "dynlib_get_info_ex", H(&Ps5DynlibGetInfoEx)},
    {609, "budget_getid", H(&Ps5Nosys)},
    {610, "budget_get_ptype", H(&Ps5BudgetGetPtype)},
    {611, "get_paging_stats_of_all_threads", H(&Ps5Eperm)},
    {612, "get_proc_type_info", H(&Ps5GetProcTypeInfo)},
    {613, "get_resident_count", H(&Ps5Eperm)},
    {618, "get_paging_stats_of_all_objects", H(&Ps5Eperm)},
    {619, "test_debug_rwmem", H(&Ps5Nosys)},
    {623, "get_gpo", H(&Ps5Eperm)},
    {627, "get_cpu_usage_all", H(&Ps5Nosys)},
    {632, "thr_suspend_ucontext", H(&sys_thr_suspend_ucontext)},
    {633, "thr_resume_ucontext", H(&sys_thr_resume_ucontext)},
    {634, "thr_get_ucontext", H(&sys_thr_get_ucontext)},
    {635, "thr_set_ucontext", H(&Ps5Nosys)},
    {637, "set_phys_fmem_limit", H(&Ps5SetPhysFmemLimit)},
    {638, "utc_to_localtime", H(&Ps5UtcToLocaltime)},
    {639, "localtime_to_utc", H(&Ps5LocaltimeToUtc)},
    {641, "get_cpu_usage_proc", H(&Ps5Nosys)},
    {642, "get_map_statistics", H(&Ps5Eperm)},
    {646, "get_kernel_mem_statistics", H(&Ps5Eperm)},
    {647, "get_sdk_compiled_version", H(&Ps5Eperm)},
    {648, "app_state_change", H(&Ps5Eperm)},
    {649, "dynlib_get_obj_member", H(&Ps5DynlibGetObjMember)},
    {650, "budget_get_ptype_of_budget", H(&Ps5Nosys)},
    {652, "process_terminate", H(&Ps5Eperm)},
    {656, "dynlib_get_info_for_libdbg", H(&Ps5Eperm)},
    // The plain calls with the kernel's hidden-module filter off; we hide none.
    {659, "dynlib_get_list2", H(&sys_dynlib_get_list)},
    {660, "dynlib_get_info2", H(&sys_dynlib_get_info)},
    {667, "get_bio_usage_all", H(&Ps5Nosys)},
    {671, "get_page_table_stats", H(&Ps5Eperm)},
    {672, "dynlib_get_list_for_libdbg", H(&Ps5Eperm)},
    {674, "virtual_query_all", H(&Ps5Einval)},
    {675, "reserve_2mb_page", H(&Ps5Ok)},
    {676, "cpumode_yield", H(&Ps5Nosys)},  // PS4 processes only
    {677, "wait6", H(&Ps5Wait6)},
    {678, "cap_rights_limit", H(&Ps5Nosys)},
    {679, "cap_ioctls_limit", H(&Ps5Nosys)},
    {680, "cap_ioctls_get", H(&Ps5Nosys)},
    {681, "cap_fcntls_limit", H(&Ps5Nosys)},
    {682, "cap_fcntls_get", H(&Ps5Nosys)},
    {683, "bindat", H(&Ps5Bindat)},
    {684, "connectat", H(&Ps5Connectat)},
    {685, "chflagsat", H(&Ps5Chflagsat)},
    {686, "accept4", H(&Ps5Accept4)},
    {687, "pipe2", H(&Ps5Pipe2)},
    {688, "aio_mlock", H(&sys_aio_unsupported)},
    {689, "procctl", H(&Ps5Procctl)},
    {690, "ppoll", H(&Ps5Ppoll)},
    {691, "futimens", H(&Ps5Futimens)},
    {692, "utimensat", H(&Ps5Utimensat)},
    {693, "numa_getaffinity", H(&Ps5NumaGetaffinity)},
    {694, "numa_setaffinity", H(&Ps5NumaSetaffinity)},
    {700, "apr_submit", H(&Ps5Nosys)},
    {701, "apr_resolve", H(&Ps5Nosys)},
    {702, "apr_stat", H(&Ps5Nosys)},
    {703, "apr_wait", H(&Ps5Nosys)},
    {704, "apr_ctrl", H(&Ps5Eperm)},
    {705, "get_phys_page_size", H(&Ps5Eperm)},  // devkit debug setting only
    {706, "begin_app_mount", H(&Ps5Eperm)},
    {707, "end_app_mount", H(&Ps5Eperm)},
    {708, "fsc2h_ctrl", H(&Ps5Eperm)},
    {709, "streamwrite", H(&Ps5Ok)},
    {710, "app_save", H(&Ps5Eperm)},
    {711, "app_restore", H(&Ps5Eperm)},
    {712, "saved_app_delete", H(&Ps5Eperm)},
    {713, "get_ppr_sdk_compiled_version", H(&Ps5Eperm)},
    {714, "notify_app_event", H(&Ps5Eperm)},
    {715, "ioreq", H(&Ps5Ioreq)},
    {716, "openintr", H(&Ps5Openintr)},
    {717, "dl_get_info_2", H(&Ps5Eperm)},
    {718, "acinfo_add", H(&Ps5Eperm)},
    {719, "acinfo_delete", H(&Ps5Eperm)},
    {720, "acinfo_get_all_for_coredump", H(&Ps5Eperm)},
    {721, "ampr_ctrl_debug", H(&Ps5AmprCtrlDebug)},
    {722, "workspace_ctrl", H(&Ps5Eperm)},
    {723, "notify_a53_bind_progress", H(&Ps5Eperm)},
    {724, "dynlib_get_lid_mid_info", H(&Ps5Eperm)},
    {725, "mmap2", H(&Ps5Mmap2)},
    {726, "get_aio_debug_submit_info", H(&Ps5Eperm)},
    {727, "get_aio_debug_request_info", H(&Ps5Eperm)},
    {728, "virtual_query_by_pid", H(&Ps5Eperm)},
    {729, "fgcinfo_add", H(&Ps5Eperm)},
    {730, "fgcinfo_delete", H(&Ps5Eperm)},
    {731, "fgcinfo_get_all_for_coredump", H(&Ps5Eperm)},
    {732, "cffmgr_call", H(&Ps5Eperm)},
    {733, "get_resident_fmem_count2", H(&Ps5GetResidentFmemCount2)},
    {734, "trace_ctrl", H(&Ps5Nosys)},
};

static bool IsPs5Nosys(u32 sid) {
  if (sid >= kPs5TableSize)
    return true;
  for (u16 id : kPs5Nosys)
    if (id == sid)
      return true;
  return false;
}

static const Ps5Row* Ps5RowOf(u32 sid) {
  for (const auto& row : kPs5Rows)
    if (row.id == sid)
      return &row;
  return nullptr;
}

uintptr_t Lv2GetPs5(u32 sid) {
  const bool nosys = IsPs5Nosys(sid);
  const Ps5Row* row = nosys ? nullptr : Ps5RowOf(sid);

  static base::Set<u32> seen;
  static const guest::SessionReset reset_seen(
      [] { guest::ResetResource(seen); });
  if (kPs5SysTrace && seen.insert(sid).second) {
    const char* name = row ? row->name : SyscallGetname(sid);
    BASE_LOGI("ps5sys", "{:4}  {}{}", sid, name ? name : "?",
              nosys ? " (nosys)" : "");
  }

  if (nosys)
    return Lv2Trampoline(H(&sys_nosys), sid);
  if (row)
    return Lv2Trampoline(row->handler, sid);
  return Lv2Get(sid);
}
}  // namespace kern
