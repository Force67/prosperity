/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <sys/mman.h>
#include "base/arch.h"
#include <thread>
#include <chrono>
#include <base.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/file.h>
#include <utl/mem.h>
#include <utl/path.h>

#include "crash.h"
#include "module.h"
#include "proc.h"

#include "lv2/ps5/ctor_probe.h"
#include "lv2/ps5/initial_tcb.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include "ps4/hardware_mode.h"
#include "lv2/sys_dynlib.h"
#include "lv2/sys_mem.h"
#include "kern/probe/probe.h"
#include "runtime/vprx/vprx.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <utl/options.h>
#include <vector>

namespace {
DELTA_OPTION(const char *, kPs5Modules, "DELTA_PS5_MODULES", nullptr);
}  // namespace

namespace krnl {
const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid

static proc *g_activeProc{nullptr};

// The guest fs base (TLS) and how the guest entry is run are backend-specific
// (see delta/cpu): native uses a host thread_local + direct call, FEX uses the
// FEXCore CPUState + JIT. setThreadFsBase() is defined by the active backend.

proc::proc() : vmem(env) { g_activeProc = this; }

proc *proc::getActive() { return g_activeProc; }

bool proc::create(const base::String &path, bool fromVfs) {
  /*register HLE prx overrides*/
  runtime::vprx_init();

  /*init memory manager*/
  LOG_ASSERT(vmem.init());

  /*reserve slot for main module*/
  auto first = utl::make_ref<smodule>(this);
  first->getInfo().handle = 0;

  modules.emplace_back(first);

  const bool ps5 = plat == platform::ps5;
  if (ps5 && !kPs5Modules)
    LOG_WARNING("PS5 title but DELTA_PS5_MODULES is unset; system modules "
                "(libkernel etc.) won't be found");

  /*pre-load required modules
   (the kernel does it, so do we)*/
  if (!loadModule(base::StringRef("libkernel")) ||
      !loadModule(base::StringRef("libSceLibcInternal"))) {
    LOG_ERROR("unable to preload sys modules");
    return false;
  }

  // PS4 libkernel is a thin forwarder: a chunk of its exports (memory-pool
  // helpers) live in libkernel_sys. PS5 has no such split, so only preload it
  // for PS4. Tolerate absence on minimal module sets.
  if (!ps5)
    loadModule(base::StringRef("libkernel_sys"));

  bool loaded = fromVfs ? first->fromVfs(path) : first->fromFile(path);
  if (!loaded) {
    LOG_ERROR("unable to load main process module");
    return false;
  }
  // PS5 modules carry no DT_SCE_MODULEINFO, so name the main module ourselves.
  if (ps5 && first->getInfo().name.empty())
    first->getInfo().name = base::String("eboot");

  probe::onProcessCreated(*this, *first, ps5);


  return true;
}
modulePtr proc::getModule(base::StringRef name) {
  for (auto &mod : modules) {
    // module name is base::String, compare via c_str.
    if (name == base::StringRef(mod->getInfo().name))
      return mod;
  }
  return {nullptr};
}

modulePtr proc::getModule(u32 handle) {
  for (auto &mod : modules) {
    if (mod->getInfo().handle == handle)
      return mod;
  }
  return {nullptr};
}

/*does not expect an extension*/
// Isaac-specific surface setup. The game's global surface-name registry, a
// fixed-bucket hash map<string,surface> at rebirth+0x687a90, is constructed
// with a NULL bucket-array pointer (its ctor at +0x1e9bcd just zeroes it) and is
// meant to get its storage lazily when the renderer registers the base surfaces
// ("Floor Surface"/"Wall Surface"). That renderer path is gated on the Gnm->
modulePtr proc::loadModule(base::StringRef name) {
  const bool isPs4GnmDriver =
      plat == platform::ps4 &&
      (name == base::StringRef("libSceGnmDriver") ||
       name == base::StringRef("libSceGnmDriverForNeoMode"));
  if (isPs4GnmDriver)
    name = base::StringRef("libSceGnmDriver");

  auto mod = getModule(name);
  if (mod)
    return mod;

  auto lib = utl::make_ref<smodule>(this);
  lib->getInfo().handle = handleCounter;
  handleCounter++;

  modules.emplace_back(lib);

  base::String sname;
  sname.append(name.data(), name.length());

  // PS5 titles use a coherent Prospero module set: system modules from a
  // firmware dump (DELTA_PS5_MODULES, a ':'-separated list of dirs holding
  // <name>.sprx), the title's own SDK modules from its decrypted/ tree. A PS5
  // process never touches the PS4 modules/ dir - the ABIs are incompatible
  // (FreeBSD 11 vs 9 syscall numbers, different struct layouts).
  if (plat == platform::ps5) {
    lib->getInfo().name = sname; // PS5 modules have no DT_SCE_MODULEINFO
    bool ok = false;
    if (const char *env = kPs5Modules) {
      for (const char *p = env; *p && !ok;) {
        const char *sep = std::strchr(p, ':');
        size_t len = sep ? static_cast<size_t>(sep - p) : std::strlen(p);
        if (len) {
          base::String dir;
          dir.append(p, len);
          if (dir.back() != '/')
            dir += "/";
          // A PS5 firmware ships thirteen sysmodules twice: <name>.sprx is the
          // build a PS4 title runs under backwards compatibility, and
          // <name>.native.sprx is the Prospero one. Both carry the same
          // SONAME, so a NEEDED entry names the pair and the process ABI picks
          // -- and a native title given the BC build is missing exports the
          // rest of its native stack imports. Astro Bot's video decoder is
          // that: libSceVideodec2 imports four NIDs only libSceVdecCore.native
          // has, and its player parks forever without them. Three sysmodules
          // ship only as .native (fw 08.40's libSceShare), so the other order
          // is still a fallback, not an alternative.
          for (const char *ext : {".native.sprx", ".sprx"}) {
            base::String hp(dir);
            hp += sname.c_str();
            hp += ext;
            if (utl::File(hp, utl::fileMode::read).IsOpen() &&
                lib->fromFile(hp)) {
              ok = true;
              break;
            }
          }
        }
        p = sep ? sep + 1 : p + len;
      }
    }
    if (!ok) {
      const char *roots[] = {"/app0/decrypted/sce_module/", "/app0/decrypted/",
                             "/app0/sce_module/", "/app0/"};
      for (const char *root : roots) {
        base::String vp(root);
        vp += sname.c_str();
        vp += ".prx";
        if (vfs::openRead(vp.c_str()).Exists() && lib->fromVfs(vp)) {
          ok = true;
          break;
        }
      }
    }
    if (!ok) {
      // Drop the placeholder we emplaced above: a failed module left in the list
      // gets enumerated and libkernel calls its null module_start (crash). Its
      // imports fall through to the badcall stub instead, which is non-fatal.
      LOG_ERROR("unable to load ps5 module {}", sname.c_str());
      modules.pop_back();
      return nullptr;
    }
    return lib;
  }

  const bool isPackagedSdkModule =
      name == base::StringRef("libc") || name == base::StringRef("libSceFios2");
  auto loadPackagedModule = [&] {
    // Retail applications provide these SDK compatibility modules in
    // /app0/sce_module rather than using the firmware copies. The firmware-
    // derived classification used here is documented by shadPS4:
    // https://github.com/shadps4-emu/shadPS4/blob/2c9caf6bfbe7e1dc7a1b4565af8d84c56469dd56/src/core/libraries/sysmodule/sysmodule_table.h#L363-L372
    const char *roots[] = {"/app0/sce_module/", "/app0/"};
    for (const char *root : roots) {
      base::String vfsPath(root);
      vfsPath.append(name.data(), name.length());
      vfsPath += ".prx";
      if (!vfs::openRead(vfsPath.c_str()).Exists())
        continue;
      if (!lib->fromVfs(vfsPath))
        return false;
      probe::onModuleLoaded(*lib, name);
      return true;
    }
    return false;
  };

  if (isPackagedSdkModule && loadPackagedModule())
    return lib;

  // HLE/system modules ship with the emulator; prefer those for modules that
  // are not supplied by the application.
  base::String hostRel("modules/");
  if (isPs4GnmDriver)
    hostRel += ps4::gnmDriverModule();
  else
    hostRel.append(name.data(), name.length());
  hostRel += ".sprx";
  base::String hostPath = utl::make_abs_path(hostRel);
  if (utl::File(hostPath, utl::fileMode::read).IsOpen()) {
    if (lib->fromFile(hostPath)) {
      if (isPs4GnmDriver)
        lib->getInfo().name = sname;
      probe::onModuleLoaded(*lib, name);
      return lib;
    }
  } else {
    // The game's own modules live inside the pkg: SDK prx under
    // /app0/sce_module, the title's own prx at the app root.
    if (loadPackagedModule())
      return lib;
  }

  LOG_ERROR("unable to load module {}", sname.c_str());
  return nullptr;
}
void proc::start() {
  LOG_ASSERT(modules[1]->getInfo().name == "libkernel");

  installCrashHandler();
  probe::onBeforeStart(*this);

  auto &info = modules[0]->getInfo();
  auto &kinfo = modules[1]->getInfo();

  if (!info.entry) {
    LOG_WARNING("entry missing for {}", info.name.c_str());
    return;
  }

  union stack_entry {
    const void *ptr;
    u64 val;
  } stack[128];

  stack[0].val = 1 + 0; // argc
  auto s = reinterpret_cast<stack_entry *>(&stack[1]);
  (*s++).ptr = info.name.c_str();
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;
  (*s++).val = 9ull;
  (*s++).ptr = (const void *)(info.entry);
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;

  // Enter libkernel's entry with the PS4 convention (arg block in rdi). The
  // backend runs it natively (x86 host) or via the FEXCore JIT (aarch64 host).
  // PS5 starts with the TCB its kernel would have installed; libkernel reads
  // fs:0x10 before it gets around to setting up its own (see makeInitialTcb).
  const u64 fsbase = plat == platform::ps5 ? ps5::makeInitialTcb() : 0;
  cpu::backend().enterGuest(reinterpret_cast<uintptr_t>(kinfo.entry), stack,
                            fsbase);
}
}  // namespace krnl
