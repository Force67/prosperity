/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero process, thread and module syscalls the PS5 kernel answers
 * differently from the Orbis table, and the PS5-only ones among them.
 */

#include <cstdint>
#include <cstring>
#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_budget.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_thread.h"
#include "kern/module.h"
#include "kern/process.h"
#include "kern/system_settings.h"

namespace kern {
namespace {
// wait(2) option bits FreeBSD 11 accepts.
constexpr int kWaitOptions = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20;

enum ProcctlCmd {
  kSprotect = 1,
  kReapAcquire = 2,
  kReapRelease = 3,
  kReapStatus = 4,
  kReapGetpids = 5,
  kReapKill = 6,
  kTraceCtl = 7,
  kTraceStatus = 8,
};
constexpr int kTraceCtlEnable = 1;
constexpr int kTraceCtlDisable = 2;
constexpr int kTraceCtlDisableExec = 3;
bool g_trace_disabled = false;

// struct procctl_reaper_status: flags, children, descendants, reaper, pid.
struct ReaperStatus {
  u32 flags;
  u32 children;
  u32 descendants;
  i32 reaper;
  i32 pid;
  u32 pad[15];
};
constexpr u32 kReaperStatusOwned = 1;

thread_local u32 t_ioreq = 0;

// Thread names as thr_set_name stored them (the kernel keeps 31 characters);
// a thread never renamed reads back the name thr_new gave it.
constexpr u32 kMaxThreads = 4096;

bool IsOurThread(i64 tid) {
  // The main thread (tid 1) never passes through thr_new, so it is not in
  // the host map.
  return tid > 0 && tid < kMaxThreads &&
         (tid == 1 || tid == *CurrentGuestTidPtr() ||
          HostTidForGuest(static_cast<u32>(tid)));
}

struct ProcTypeInfo {
  u64 size;
  u32 type;
  u32 caps;
};
// JIT application | SCE program attribute, as sys_get_proc_type_info reports.
constexpr u32 kGameCaps = 0x42;

}  // namespace

// We host no child processes.
int PS4ABI Ps5Wait6(int /*idtype*/,
                    i64 /*id*/,
                    int* /*status*/,
                    int options,
                    void* /*wrusage*/,
                    void* /*info*/) {
  if (options & ~kWaitOptions)
    return -SysError::eINVAL;
  return -SysError::eCHILD;
}

int PS4ABI Ps5Procctl(int /*idtype*/, i64 id, int cmd, void* data) {
  if (cmd < kSprotect || cmd > kTraceStatus)
    return -SysError::eINVAL;
  if (id && id != sys_getpid())
    return -SysError::eSRCH;
  switch (cmd) {
    case kSprotect:
    case kReapAcquire:
    case kReapRelease:
      return 0;
    case kReapStatus: {
      if (!data)
        return -SysError::eFAULT;
      ReaperStatus rs{};
      rs.flags = kReaperStatusOwned;
      rs.reaper = sys_getpid();
      rs.pid = -1;
      std::memcpy(data, &rs, sizeof(rs));
      return 0;
    }
    case kReapGetpids:
      return data ? 0 : -SysError::eFAULT;  // no descendants to list
    case kReapKill:
      return -SysError::eSRCH;
    case kTraceCtl: {
      const int how = data ? *static_cast<const int*>(data) : 0;
      if (how == kTraceCtlEnable)
        g_trace_disabled = false;
      else if (how == kTraceCtlDisable || how == kTraceCtlDisableExec)
        g_trace_disabled = true;
      else
        return -SysError::eINVAL;
      return 0;
    }
    case kTraceStatus:
      if (!data)
        return -SysError::eFAULT;
      *static_cast<int*>(data) = g_trace_disabled ? -1 : 0;
      return 0;
  }
  return -SysError::eINVAL;
}

// which 1 is a thread, 2 the process. A PS5 has one memory domain.
int PS4ABI Ps5NumaGetaffinity(int which, i64 /*id*/, void* policy) {
  if (which != 1 && which != 2)
    return -SysError::eINVAL;
  if (!policy)
    return -SysError::eFAULT;
  const u64 domains = 1;
  std::memcpy(policy, &domains, sizeof(domains));
  return 0;
}

int PS4ABI Ps5NumaSetaffinity(int which, i64 /*id*/, const void* policy) {
  if (!policy)
    return -SysError::eFAULT;
  return which == 1 || which == 2 ? 0 : -SysError::eINVAL;
}

// The calling thread's I/O request class: op 0 reads it, op 1 sets it to one
// of the classes a title may pick (0xb0, 0xd0, 0xe0 in bits 4..7).
int PS4ABI Ps5Ioreq(int op, u32 value, u32* out) {
  if (op == 0) {
    if (!out)
      return -SysError::eFAULT;
    *out = t_ioreq;
    return 0;
  }
  if (op != 1 || !value)
    return -SysError::eINVAL;
  const u32 cls = value & 0xf0;
  if (cls != 0xb0 && cls != 0xd0 && cls != 0xe0)
    return -SysError::eINVAL;
  t_ioreq = value;
  return 0;
}

int PS4ABI Ps5ThrSetName(i64 tid, const char* name) {
  if (!IsOurThread(tid))
    return -SysError::eSRCH;
  if (!name)
    return -SysError::eFAULT;
  return sys_thr_set_name(static_cast<u32>(tid), name);
}

// The caller sets the size field, and the kernel only answers the 16-byte
// layout it knows.
int PS4ABI Ps5GetProcTypeInfo(void* info) {
  if (!info)
    return -SysError::eFAULT;
  auto* out = static_cast<ProcTypeInfo*>(info);
  if (out->size != sizeof(ProcTypeInfo))
    return -SysError::eINVAL;
  out->type = static_cast<u32>(sys_budget_get_ptype());
  out->caps = kGameCaps;
  return 0;
}

// An unknown handle and an unresolved symbol are both ESRCH to the kernel.
int PS4ABI Ps5DynlibDlsym(u32 handle, const char* name, void** sym) {
  const int err = sys_dynlib_dlsym(handle, name, sym);
  return err == -1 ? -SysError::eSRCH : err;
}

// The PS5 kernel lists every loaded segment with its own protection where the
// PS4 one reports a fixed text (r-x) and data (rw-) pair.
int PS4ABI Ps5DynlibGetInfoEx(u32 handle, i32 flags, void* info) {
  auto* out = static_cast<dynlib_info_ex*>(info);
  const int err = sys_dynlib_get_info_ex(handle, flags, out);
  if (err)
    return err;
  auto mod = Process::GetActive()->GetModule(handle);
  if (!mod)
    return err;
  const auto& mi = mod->GetInfo();
  std::memset(out->segs, 0, sizeof(out->segs));
  for (u32 i = 0; i < mi.load_seg_count; i++) {
    out->segs[i].addr = reinterpret_cast<uintptr_t>(mi.load_segs[i].addr);
    out->segs[i].size = mi.load_segs[i].size;
    out->segs[i].flags = mi.load_segs[i].prot;
  }
  out->seg_count = mi.load_seg_count;
  return 0;
}

// A title may only ask about itself (-1 or its pid); asking about another
// process is for system ones.
int PS4ABI Ps5BudgetGetPtype(int pid) {
  if (pid != -1 && pid != sys_getpid())
    return -SysError::eNOSYS;
  return sys_budget_get_ptype();
}

// The flexible-memory limit of the caller (pid 0 or its own); nothing here
// enforces one, so only the checks are real.
int PS4ABI Ps5SetPhysFmemLimit(int pid, i64 limit) {
  if (limit < 0x11)
    return -SysError::eINVAL;
  if (pid && pid != sys_getpid())
    return -SysError::ePERM;
  return 0;
}

// A registry integer, for the five keys the kernel lets a title read.
int PS4ABI Ps5Workaround8849(u32 key) {
  static constexpr u32 kKeys[] = {0x78026400, 0x78028a00, 0x78028300,
                                  0x19810000, 0x3cc80700};
  for (u32 k : kKeys)
    if (k == key) {
      i32 value = 0;
      GetRegistryInt(key, value);
      return value;
    }
  return -SysError::eINVAL;
}

// Debug controls for the AMPR engine; a retail unit refuses them.
int PS4ABI Ps5AmprCtrlDebug(int op, void* /*arg*/, size_t /*size*/) {
  if (op < 1 || op > 5)
    return -SysError::eINVAL;
  return -SysError::eOPNOTSUPP;
}

}  // namespace kern
