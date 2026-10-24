// Debugger, perf-counter and platform-control syscalls. None is modelled;
// each answers the way a retail console without a debugger attached would.

#include "kern/lv2/sys_debug.h"

#include <cstring>
#include "base/atomic.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/error_table.h"
#include "kern/object.h"
#include "kern/process.h"

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

namespace {
// A name -> data pointer entry in the process id table, the table event flags
// and semaphores also live in. The kernel tags the type with 0x1000.
class NamedObject : public Object {
 public:
  NamedObject(ObjectTable& objects, const char* name, void* data, u32 type)
      : Object(objects, Type::kNamedObject), data_(data), type_(type) {
    SetName(name);
  }
  u32 tag() const { return type_; }

 private:
  void* data_;
  u32 type_;
};
constexpr u32 kNamedObjectTag = 0x1000;
constexpr size_t kNamedObjectNameMax = 32;
}  // namespace

int PS4ABI sys_namedobj_create(const char* name, void* data, u32 type) {
  if (!name)
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(name, 1))
    return -SysError::eFAULT;
  if (strnlen(name, kNamedObjectNameMax) == kNamedObjectNameMax)
    return -SysError::eNAMETOOLONG;
  auto* obj = new NamedObject(Process::GetActive()->GetObjTable(), name, data,
                              type | kNamedObjectTag);
  return static_cast<int>(obj->handle());
}

int PS4ABI sys_namedobj_delete(u32 id, u32 type) {
  auto& objects = Process::GetActive()->GetObjTable();
  auto obj = objects.Get(id);
  if (!obj || obj->type() != Object::Type::kNamedObject ||
      static_cast<NamedObject*>(obj.get())->tag() !=
          ((type & 0xffff) | kNamedObjectTag))
    return -SysError::eSRCH;
  obj->ReleaseHandle();
  return 0;
}

// sys_workaround8849 (605): restricted registry-int getter; kernel validates
// the key against four whitelisted values and reads sceRegMgrGetInt. No
// registry here, so return 0 (an unprovisioned console's value).
int PS4ABI sys_workaround8849() {
  return 0;
}

}  // namespace kern
