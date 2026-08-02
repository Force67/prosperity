#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base/containers/vector.h>
#include <base/strings/xstring.h>
#include <base/strings/string_ref.h>

#include "ps4/dev/device.h"
#include "module.h"
#include "object.h"
#include "util/object_table.h"
#include "vm_manager.h"

namespace krnl {
struct procInfo {
  uint32_t ripZoneSize = 5 * 1024;
  uint8_t *userStack = nullptr;
  size_t userStackSize = 20 * 1024 * 1024;
  void *fsBase = nullptr;
};

class smodule;
class kObject;

// Set the calling host thread's guest fs base (guest TLS pointer). Called for
// the main thread (sysarch 129) and for each guest thread we spawn.
void setThreadFsBase(uint64_t);
// The calling host thread's guest fs base, 0 when no guest thread runs on it.
uint64_t threadFsBase();
int32_t hostGuestFsOffset();
int32_t hostFsScratchOffset();

/*TODO: FIX MISUSE OF modulePtr*/
using modulePtr = utl::object_ref<smodule>;

class proc {
  friend class smodule;

public:
  using moduleList = base::Vector<modulePtr>;

  enum class platform { ps4, ps5 };

  proc();
  // Load the process. When fromVfs is set, path is a guest VFS path (e.g.
  // "/app0/eboot.bin") loaded through the mount table; otherwise a host file.
  bool create(const base::String &, bool fromVfs = false);
  void start();

  static proc *getActive();

  inline moduleList &getModuleList() { return modules; }
  inline objectTable &getObjTable() { return objects; }

  modulePtr loadModule(base::StringRef);
  modulePtr getModule(base::StringRef);
  modulePtr getModule(uint32_t);

  inline vmManager &getVma() { return vmem; }
  inline procInfo &getEnv() { return env; }

  platform getPlatform() const { return plat; }
  void setPlatform(platform p) { plat = p; }

  // SDK version the title was built against, 0xMMmmpppp (PS5 titles carry it in
  // sce_sys/param.json). libkernel reads it back through sysctl kern.proc.36 and
  // branches on it; 0 makes it take pre-1.70 code paths.
  uint32_t getSdkVersion() const { return sdkVersion; }
  void setSdkVersion(uint32_t v) { sdkVersion = v; }

private:
  vmManager vmem;
  procInfo env;
  platform plat = platform::ps4;
  uint32_t sdkVersion = 0;
  moduleList modules;
  objectTable objects;
  uint32_t handleCounter = 1;
  uint16_t tlsCounter = 1;

  // 1-based ELF TLS module index handed to each module that ships a PT_TLS.
  // libkernel uses this as the DTV slot; it must be unique and non-negative
  // (-1 corrupts DTPMOD relocations and the DTV).
  uint16_t nextFreeTLS() { return tlsCounter++; }
};

// DELTA_FIOS_TRACE hook: called from smodule::resolveImports for each PLT import.
// Returns a guest wrapper around `realAddr` that traces the libSceFios2 whole-file
// APIs (FHOpen/FHGetSize/FHRead/FHPread) when the env var is set and `nidName`
// (encoded "NID#lib#mod") matches; otherwise returns realAddr unchanged. Defined
// in proc.cpp. Off (identity) by default.
uintptr_t maybeWrapFiosImport(const char *nidName, uintptr_t realAddr);
}
