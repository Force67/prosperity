#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"

#include "kern/module.h"
#include "kern/object.h"
#include "kern/object_table.h"
#include "kern/ps4/dev/device.h"
#include "kern/vm_map.h"

namespace krnl {
struct ProcInfo {
  u32 rip_zone_size = 5 * 1024;
  u8* user_stack = nullptr;
  size_t user_stack_size = 20 * 1024 * 1024;
  void* fs_base = nullptr;
};

class Smodule;
class Object;

/*TODO: ModulePtr is misused in places; audit the refs*/
using ModulePtr = krnl::ObjectRef<Smodule>;

class Proc {
  friend class Smodule;

 public:
  using ModuleList = base::Vector<ModulePtr>;

  enum class Platform { kPs4, kPs5 };

  Proc();
  // Load the process. When fromVfs is set, path is a guest VFS path (e.g.
  // "/app0/eboot.bin") loaded through the mount table; otherwise a host file.
  bool Create(const base::String&, bool from_vfs = false);
  void Start();

  static Proc* GetActive();

  inline ModuleList& GetModuleList() { return modules_; }
  inline ObjectTable& GetObjTable() { return objects_; }

  ModulePtr LoadModule(base::StringRef);
  ModulePtr GetModule(base::StringRef);
  ModulePtr GetModule(u32);

  inline VmManager& GetVma() { return vmem_; }
  inline ProcInfo& GetEnv() { return env_; }

  Platform GetPlatform() const { return plat_; }
  void SetPlatform(Platform p) { plat_ = p; }

  // SDK version the title was built against, 0xMMmmpppp (PS5 titles carry it in
  // sce_sys/param.json). libkernel reads it back through sysctl kern.proc.36
  // and branches on it; 0 makes it take pre-1.70 code paths.
  u32 GetSdkVersion() const { return sdk_version_; }
  void SetSdkVersion(u32 v) { sdk_version_ = v; }

 private:
  VmManager vmem_;
  ProcInfo env_;
  Platform plat_ = Platform::kPs4;
  u32 sdk_version_ = 0;
  ModuleList modules_;
  ObjectTable objects_;
  u32 handle_counter_ = 1;
  u16 tls_counter_ = 1;

  // 1-based ELF TLS module index handed to each module that ships a PT_TLS.
  // libkernel uses this as the DTV slot; it must be unique and non-negative
  // (-1 corrupts DTPMOD relocations and the DTV).
  u16 NextFreeTls() { return tls_counter_++; }
};

}  // namespace krnl
