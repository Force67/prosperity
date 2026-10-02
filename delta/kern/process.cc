/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <sys/mman.h>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "host_memory/host_memory.h"
#include "io/file.h"
#include "io/path.h"

#include "kern/crash.h"
#include "kern/module.h"
#include "kern/process.h"

#include "cpu/backend.h"
#include "kern/lv2/ps5/ctor_probe.h"
#include "kern/lv2/ps5/initial_tcb.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_mem.h"
#include "kern/probe/probe.h"
#include "kern/ps4/hardware_mode.h"
#include "kern/vfs.h"
#include "runtime/vprx/vprx.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/strings/string_ref.h"
#include "kern/lv2/sys_thread.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kPs4Modules, "DELTA_PS4_MODULES", nullptr);
DELTA_OPTION(const char*, kPs5Modules, "DELTA_PS5_MODULES", nullptr);
}  // namespace

namespace kern {

static Process* g_active_proc{nullptr};

// The guest fs base (TLS) and how the guest entry is run are backend-specific
// (see delta/cpu): native uses a host thread_local + direct call, FEX uses the
// FEXCore CPUState + JIT. cpu::SetThreadFsBase() is defined by the active
// backend.

Process::Process() : vmem_(env_) {
  g_active_proc = this;
}

Process* Process::GetActive() {
  return g_active_proc;
}

bool Process::Create(const base::String& path, bool from_vfs) {
  /*register HLE prx overrides*/
  runtime::vprx::Init();

  /*init memory manager*/
  LOG_ASSERT(vmem_.Init());

  /*reserve slot for main module*/
  auto first = kern::MakeRef<Module>(this);
  first->GetInfo().handle = 0;

  modules_.emplace_back(first);

  const bool ps5 = plat_ == Platform::kPs5;
  if (ps5 && !kPs5Modules)
    LOG_WARNING(
        "PS5 title but DELTA_PS5_MODULES is unset; system modules "
        "(libkernel etc.) won't be found");

  /*pre-load required modules
   (the kernel does it, so do we)*/
  if (!LoadModule(base::StringRef("libkernel")) ||
      !LoadModule(base::StringRef("libSceLibcInternal"))) {
    LOG_ERROR("unable to preload sys modules");
    return false;
  }

  // PS4 libkernel is a thin forwarder: a chunk of its exports (memory-pool
  // helpers) live in libkernel_sys. PS5 has no such split, so only preload it
  // for PS4. Tolerate absence on minimal module sets.
  if (!ps5)
    LoadModule(base::StringRef("libkernel_sys"));

  bool loaded = from_vfs ? first->FromVfs(path) : first->FromFile(path);
  if (!loaded) {
    LOG_ERROR("unable to load main process module");
    return false;
  }
  // PS5 modules carry no DT_SCE_MODULEINFO, so name the main module ourselves.
  if (ps5 && first->GetInfo().name.empty())
    first->GetInfo().name = base::String("eboot");

  probe::OnProcessCreated(*this, *first, ps5);

  return true;
}
ModulePtr Process::GetModule(base::StringRef name) {
  for (auto& mod : modules_) {
    // module name is base::String, compare via c_str.
    if (name == base::StringRef(mod->GetInfo().name))
      return mod;
  }
  return {nullptr};
}

ModulePtr Process::GetModule(u32 handle) {
  for (auto& mod : modules_) {
    if (mod->GetInfo().handle == handle)
      return mod;
  }
  return {nullptr};
}

/*does not expect an extension*/
// Isaac-specific surface setup. The game's global surface-name registry, a
// fixed-bucket hash map<string,surface> at rebirth+0x687a90, is constructed
// with a NULL bucket-array pointer (its ctor at +0x1e9bcd just zeroes it) and
// is meant to get its storage lazily when the renderer registers the base
// surfaces
// ("Floor Surface"/"Wall Surface"). That renderer path is gated on the Gnm->
ModulePtr Process::LoadModule(base::StringRef name) {
  const bool is_ps4_gnm_driver =
      plat_ == Platform::kPs4 &&
      (name == base::StringRef("libSceGnmDriver") ||
       name == base::StringRef("libSceGnmDriverForNeoMode"));
  if (is_ps4_gnm_driver)
    name = base::StringRef("libSceGnmDriver");

  auto mod = GetModule(name);
  if (mod)
    return mod;

  auto lib = kern::MakeRef<Module>(this);
  lib->GetInfo().handle = handle_counter_;
  handle_counter_++;

  modules_.emplace_back(lib);

  base::String sname;
  sname.append(name.data(), name.length());

  // PS5 titles use a coherent Prospero module set: system modules from a
  // firmware dump (DELTA_PS5_MODULES, a ':'-separated list of dirs holding
  // <name>.sprx), the title's own SDK modules from its decrypted/ tree. A PS5
  // process never touches the PS4 modules/ dir - the ABIs are incompatible
  // (FreeBSD 11 vs 9 syscall numbers, different struct layouts).
  if (plat_ == Platform::kPs5) {
    lib->GetInfo().name = sname;  // PS5 modules have no DT_SCE_MODULEINFO
    bool ok = false;
    if (const char* env = kPs5Modules) {
      for (const char* p = env; *p && !ok;) {
        const char* sep = std::strchr(p, ':');
        size_t len = sep ? static_cast<size_t>(sep - p) : std::strlen(p);
        if (len) {
          base::String dir;
          dir.append(p, len);
          if (dir.back() != '/')
            dir += "/";
          // A PS5 firmware ships thirteen sysmodules twice: <name>.sprx is the
          // build a PS4 title runs under backwards compatibility, and
          // <name>.native.sprx is the Prospero one. Both carry the same SONAME,
          // so a NEEDED entry names the pair and the process ABI picks, and a
          // native title given the BC build is missing exports the rest of its
          // native stack imports (Astro Bot's libSceVideodec2 imports four NIDs
          // only libSceVdecCore.native has, and its player parks forever
          // without them). Three sysmodules ship only as .native (fw 08.40's
          // libSceShare), so the other order is still a fallback, not an
          // alternative.
          for (const char* ext : {".native.sprx", ".sprx"}) {
            base::String hp(dir);
            hp += sname.c_str();
            hp += ext;
            if (io::File(hp, io::FileMode::kRead).IsOpen() &&
                lib->FromFile(hp)) {
              ok = true;
              break;
            }
          }
        }
        p = sep ? sep + 1 : p + len;
      }
    }
    if (!ok) {
      const char* roots[] = {"/app0/decrypted/sce_module/", "/app0/decrypted/",
                             "/app0/sce_module/", "/app0/"};
      for (const char* root : roots) {
        base::String vp(root);
        vp += sname.c_str();
        vp += ".prx";
        if (vfs::OpenRead(vp.c_str()).Exists() && lib->FromVfs(vp)) {
          ok = true;
          break;
        }
      }
    }
    if (!ok) {
      // Drop the placeholder we emplaced above: a failed module left in the
      // list gets enumerated and libkernel calls its null module_start (crash).
      // Its imports fall through to the badcall stub instead, which is
      // non-fatal.
      LOG_ERROR("unable to load ps5 module {}", sname.c_str());
      modules_.pop_back();
      return nullptr;
    }
    return lib;
  }

  const bool is_packaged_sdk_module =
      name == base::StringRef("libc") || name == base::StringRef("libSceFios2");
  auto load_packaged_module = [&] {
    // Retail applications provide these SDK compatibility modules in
    // /app0/sce_module rather than using the firmware copies. The firmware-
    // derived classification used here is documented by shadPS4:
    // https://github.com/shadps4-emu/shadPS4/blob/2c9caf6bfbe7e1dc7a1b4565af8d84c56469dd56/src/core/libraries/sysmodule/sysmodule_table.h#L363-L372
    const char* roots[] = {"/app0/sce_module/", "/app0/"};
    for (const char* root : roots) {
      base::String vfs_path(root);
      vfs_path.append(name.data(), name.length());
      vfs_path += ".prx";
      if (!vfs::OpenRead(vfs_path.c_str()).Exists())
        continue;
      if (!lib->FromVfs(vfs_path))
        return false;
      probe::OnModuleLoaded(*lib, name);
      return true;
    }
    return false;
  };

  if (is_packaged_sdk_module && load_packaged_module())
    return lib;

  // HLE/system modules ship with the emulator; prefer those for modules that
  // are not supplied by the application.
  base::String host_rel("modules/");
  if (is_ps4_gnm_driver)
    host_rel += ps4::GnmDriverModule();
  else
    host_rel.append(name.data(), name.length());
  host_rel += ".sprx";
  base::String host_path;
  if (const char* directory = kPs4Modules; directory && *directory) {
    host_path = directory;
    if (host_path.back() != '/')
      host_path += "/";
    host_path += host_rel.substr(base::CountStringLength("modules/"));
  } else {
    host_path = io::MakeAbsPath(host_rel);
  }
  if (io::File(host_path, io::FileMode::kRead).IsOpen()) {
    if (lib->FromFile(host_path)) {
      if (is_ps4_gnm_driver)
        lib->GetInfo().name = sname;
      probe::OnModuleLoaded(*lib, name);
      return lib;
    }
  } else {
    // The game's own modules live inside the pkg: SDK prx under
    // /app0/sce_module, the title's own prx at the app root.
    if (load_packaged_module())
      return lib;
  }

  LOG_ERROR("unable to load module {}", sname.c_str());
  return nullptr;
}
void Process::Start() {
  LOG_ASSERT(modules_[1]->GetInfo().name == "libkernel");

  InstallCrashHandler();
  probe::OnBeforeStart(*this);

  auto& info = modules_[0]->GetInfo();
  auto& kinfo = modules_[1]->GetInfo();

  if (!info.entry) {
    LOG_WARNING("entry missing for {}", info.name.c_str());
    return;
  }

  union stack_entry {
    const void* ptr;
    u64 val;
  } stack[128];

  stack[0].val = 1 + 0;  // argc
  auto s = reinterpret_cast<stack_entry*>(&stack[1]);
  (*s++).ptr = info.name.c_str();
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;
  (*s++).val = 9ull;
  (*s++).ptr = (const void*)(info.entry);
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;

  // Enter libkernel's entry with the PS4 convention (arg block in rdi). The
  // backend runs it natively (x86 host) or via the FEXCore JIT (aarch64 host).
  // PS5 starts with the TCB its kernel would have installed; libkernel reads
  // fs:0x10 before it gets around to setting up its own (see MakeInitialTcb).
  const u64 fsbase = plat_ == Platform::kPs5 ? ps5::MakeInitialTcb() : 0;
  cpu::GetBackend().EnterGuest(reinterpret_cast<uintptr_t>(kinfo.entry), stack,
                               fsbase);
}
}  // namespace kern
