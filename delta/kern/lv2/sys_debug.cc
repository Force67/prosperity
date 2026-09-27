// Debugger, perf-counter and platform-control syscalls. None is modelled;
// each answers the way a retail console without a debugger attached would.

#include "kern/lv2/sys_debug.h"

#include "base/atomic.h"
#include "kern/lv2/error_table.h"

namespace kern {

// The kernel stores the debugger protocol version once and answers EBUSY to any
// repeat init, so a second sys_debug_init from the guest is already an error we
// mirror rather than fake away.
int PS4ABI sys_debug_init(const int* version) {
  static base::Atomic<bool> once{false};
  if (!version || once.exchange(true))
    return -SysError::eBUSY;
  return 0;
}

// Hardware performance counters: no PMU, so every op is a silent success and
// readers see idle counters.
int PS4ABI sys_opmc_enable() {
  return 0;
}
int PS4ABI sys_opmc_disable() {
  return 0;
}
int PS4ABI sys_opmc_set_ctl() {
  return 0;
}
int PS4ABI sys_opmc_set_ctr() {
  return 0;
}
int PS4ABI sys_opmc_get_ctr() {
  return 0;
}
int PS4ABI sys_opmc_set_hw() {
  return 0;
}
int PS4ABI sys_opmc_get_hw() {
  return 0;
}

int PS4ABI sys_mdbg_call() {
  return 0;
}

// General-purpose output (debug LEDs/pins). No hardware; accept and ignore.
int PS4ABI sys_set_gpo() {
  return 0;
}
int PS4ABI sys_get_gpo() {
  return 0;
}

int PS4ABI sys_test_debug_rwmem() {
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_get_cpu_usage_all() {
  return 0;
}
int PS4ABI sys_get_cpu_usage_proc() {
  return 0;
}

int PS4ABI sys_set_uevt() {
  return 0;
}

int PS4ABI sys_set_chicken_switches() {
  return 0;
}

int PS4ABI sys_unk645() {
  return 0;
}

// sys_namedobj_create (557): registers name -> data pointer in the process id
// table (24-byte object {name@0, data@8, flags@16}, name max 32 chars, flags OR
// 0x1000, id in rax). We return 0; the only known caller ignores the id.
int PS4ABI sys_namedobj_create(const char* name, void* arg2, u32 arg3) {
  (void)name;
  (void)arg2;
  (void)arg3;
  return 0;
}

int PS4ABI sys_namedobj_delete() {
  return 0;
}

// sys_workaround8849 (605): restricted registry-int getter; kernel validates
// the key against four whitelisted values and reads sceRegMgrGetInt. No
// registry here, so return 0 (an unprovisioned console's value).
int PS4ABI sys_workaround8849() {
  return 0;
}

}  // namespace kern
