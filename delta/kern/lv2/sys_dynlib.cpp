
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include "kern/crash.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <logger/logger.h>

#include <utl/mem.h>
#include "cpu/cpu_backend.h"
#include "kern/module.h"
#include "kern/proc.h"

#include "error_table.h"
#include "sys_dynlib.h"

#include "sys_mem.h"
#include <runtime/vprx/vprx.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kVoInitOff, "DELTA_VO_INIT_OFF", nullptr);
DELTA_OPTION(bool, kModinitTrace, "DELTA_MODINIT_TRACE", false);
DELTA_OPTION(bool, kProcparamTrace, "DELTA_PROCPARAM_TRACE", false);
DELTA_OPTION(bool, kVoLleFix, "DELTA_VO_LLE_FIX", false);
}  // namespace

namespace krnl {
int PS4ABI sys_dynlib_dlopen(const char *) {
  /* devkit-only; retail titles never call it */
  return -SysError::eNOSYS;
}

int PS4ABI sys_dynlib_get_info(u32 handle, dynlib_info *dyn_info) {
  if (!dyn_info)
    return -SysError::eFAULT;
  if (dyn_info->size != sizeof(*dyn_info))
    return -SysError::eINVAL;

  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto &info = mod->getInfo();
  std::memset(dyn_info, 0, sizeof(dynlib_info));
  std::strncpy(dyn_info->name, info.name.c_str(), 256);

  auto &text = dyn_info->segs[0];
  text.addr = reinterpret_cast<uintptr_t>(info.textSeg.addr);
  text.size = info.textSeg.size;
  text.flags = 1 | 4;

  auto &data = dyn_info->segs[1];
  data.addr = reinterpret_cast<uintptr_t>(info.dataSeg.addr);
  data.size = info.dataSeg.size;
  data.flags = 1 | 2;

  dyn_info->seg_count = 2;

  std::memcpy(dyn_info->fingerprint, info.fingerprint, 20);
  return 0;
}

int PS4ABI sys_dynlib_get_info_ex(u32 handle, i32 ukn /*always 1*/,
                                  dynlib_info_ex *dyn_info) {
  if (!dyn_info)
    return -SysError::eFAULT;
  if (dyn_info->size != sizeof(*dyn_info))
    return -SysError::eINVAL;

  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto &info = mod->getInfo();
  std::memset(dyn_info, 0, sizeof(dynlib_info_ex));

  dyn_info->size = sizeof(dynlib_info_ex);
  dyn_info->handle = info.handle;
  std::strncpy(dyn_info->name, info.name.c_str(), 256);

  dyn_info->tls_index = info.tlsSlot;
  dyn_info->tls_align = info.tlsalign;
  dyn_info->tls_init_size = info.tlsSizeFile;
  dyn_info->tls_size = info.tlsSizeMem;
  dyn_info->tls_init_addr = reinterpret_cast<uintptr_t>(info.tlsAddr);

  dyn_info->init_proc_addr = reinterpret_cast<uintptr_t>(info.initAddr);
  dyn_info->fini_proc_addr = reinterpret_cast<uintptr_t>(info.finiAddr);

  // installEHFrame stores the hdr (PT_GNU_EH_FRAME) in ehFrameAddr and the unwind
  // data (.eh_frame) in ehFrameheaderAddr, i.e. named backwards; report as the
  // guest unwinder expects.
  dyn_info->eh_frame_addr =
      reinterpret_cast<uintptr_t>(info.ehFrameheaderAddr);
  dyn_info->eh_frame_hdr_addr = reinterpret_cast<uintptr_t>(info.ehFrameAddr);
  dyn_info->eh_frame_size = info.ehFrameheaderSize;
  dyn_info->eh_frame_hdr_size = info.ehFrameSize;
  BASE_LOGI("get_info_ex", "h={} {} eh_frame={:#x}+{:#x} hdr={:#x}+{:#x}",
            handle, info.name.c_str(), dyn_info->eh_frame_addr,
            dyn_info->eh_frame_size, dyn_info->eh_frame_hdr_addr,
            dyn_info->eh_frame_hdr_size);

  auto &text = dyn_info->segs[0];
  text.addr = reinterpret_cast<uintptr_t>(info.textSeg.addr);
  text.size = info.textSeg.size;
  text.flags = 1 | 4;

  auto &data = dyn_info->segs[1];
  data.addr = reinterpret_cast<uintptr_t>(info.dataSeg.addr);
  data.size = info.dataSeg.size;
  data.flags = 1 | 2;

  dyn_info->seg_count = 2;
  dyn_info->ref_count = 1;
  return 0;
}

int PS4ABI sys_dynlib_dlsym(u32 handle, const char *symName, void **sym) {
  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -1;

  char nameenc[12]{};
  runtime::encode_nid(symName, reinterpret_cast<u8 *>(&nameenc));

  auto &modName = mod->getInfo().name;

  // dlsym resolves an EXPORT of the target module by its NID. resolveObfSymbol
  // is for imports (it decodes "#libid#modid"); here we match the 11-char NID
  // directly against the module's export table.
  uintptr_t addrOut = mod->getSymbolByNid(nameenc);

  // Callers may pass an already-encoded 11-char NID (e.g. the SDK module-entry
  // symbol "BaOKcng8g88") instead of a plain name. encode_nid() would hash the
  // NID string itself and never match, so also try matching it verbatim.
  if (!addrOut && std::strlen(symName) == 11)
    addrOut = mod->getSymbolByNid(symName);

  if (!addrOut) {
    BASE_LOGI("dlsym", "{}!{} -> UNRESOLVED", modName.c_str(), symName);
    *sym = nullptr;
    return -1;
  }

  BASE_LOGI("dlsym", "{}!{} -> {:p} (+{:#x})", modName.c_str(), symName,
            reinterpret_cast<void *>(addrOut),
            addrOut - reinterpret_cast<uintptr_t>(mod->getInfo().base));

  *sym = reinterpret_cast<void *>(addrOut);

  return 0;
}

int PS4ABI sys_dynlib_get_obj_member(u32 handle, u8 index,
                                     void **value) {
  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto &info = mod->getInfo();
  switch (index) {
  case 1:  // module init proc
    *value = info.initAddr;
    if (kModinitTrace)
      BASE_LOGI("modinit", "h={} {} init={:p}", handle, info.name.c_str(),
                info.initAddr);
    return 0;
  case 2:  // module fini proc
    *value = info.finiAddr;
    return 0;
  case 8:  // SCE module param (libSceSysmodule reads the SDK version from it)
    *value = info.moduleParam;
    return 0;
  default:
    LOG_WARNING("get_obj_member: unhandled index {} for {}", index,
                info.name.c_str());
    return -SysError::eINVAL;
  }
}

int PS4ABI sys_dynlib_get_proc_param(void **data, size_t *size) {
  auto mod = proc::getActive()->getModuleList()[0];
  if (mod) {
    auto &info = mod->getInfo();

    *data = reinterpret_cast<void *>(info.procParam);
    *size = info.procParamSize;
    if (kProcparamTrace)
      BASE_LOGI("procparam", "get_proc_param -> data={:p} size={:#x}", *data,
                *size);
    return 0;
  }

  *data = nullptr;
  *size = 0;

  return 0;
}

int PS4ABI sys_dynlib_get_list(u32 *handles, size_t maxCount,
                               size_t *count) {
  auto *proc = proc::getActive();
  auto &list = proc->getModuleList();

  // The real kernel loads each PRX's needed modules first, so its list is
  // dependency-ordered and libkernel's init walker initializes a library before
  // its dependents. Our discovery order can invert that: SOTTR's NpManager PrxStart
  // ran before libSceHttp's init, got 0x80431001, left its NP context null. Emit
  // dependency-first (postorder over DT_SCE_NEEDED_MODULE), main module up front.
  base::Vector<smodule *> sorted;
  std::unordered_set<smodule *> visited;
  std::function<void(smodule *)> visit = [&](smodule *m) {
    if (!visited.insert(m).second)
      return;
    for (auto &dep : m->neededObjects()) {
      auto d = proc->getModule(base::StringRef(dep.c_str()));
      if (d && d.get() != m)
        visit(d.get());
    }
    sorted.push_back(m);
  };
  for (auto &mod : list)
    visit(mod.get());

  size_t listCount = 0;
  if (!list.empty() && !list[0]->getInfo().name.empty() && maxCount > 0) {
    *(handles++) = list[0]->getInfo().handle;  // main module first
    listCount++;
  }
  for (auto *mod : sorted) {
    if (!list.empty() && mod == list[0].get())
      continue;
    if (mod->getInfo().name.empty())
      continue;
    if (listCount >= maxCount)
      break;  // don't overflow the caller's buffer
    *(handles++) = mod->getInfo().handle;
    listCount++;
  }

  *count = listCount;
  return 0;
}

// syscall 594. libkernel's _sceKernelLoadStartModule path calls this with
// rdi=path, rsi=flags, rdx=&handle. It expects the loaded module's handle
// written to *pHandle and 0 returned on success.
int PS4ABI sys_dynlib_load_prx(const char *path, u64 flags, int *pHandle,
                               u64 arg4, const void *opt, i64 *pRes) {
  if (pRes)
    *pRes = 0;
  if (!path)
    return -SysError::eINVAL;

  // Derive the module name from the path: basename minus its extension. The
  // module's internal name (set from the SCE module info) matches this for the
  // system libs we preload, so getModule() can find an already-loaded one.
  const char *slash = std::strrchr(path, '/');
  const char *baseStart = slash ? slash + 1 : path;
  const char *dot = std::strrchr(baseStart, '.');
  size_t nameLen = dot ? static_cast<size_t>(dot - baseStart)
                       : std::strlen(baseStart);
  base::String name;
  name.append(baseStart, nameLen);

  BASE_LOGI("load_prx", "path='{}' -> '{}' flags={:#x}", path, name.c_str(),
            (unsigned long long)flags);
  // DELTA_LOADPRX_STACK: who asked. A sysmodule the guest load-starts and one
  // it never asks for want different answers, and only the caller says which.
  if (std::getenv("DELTA_LOADPRX_STACK"))
    guestStackTrace("load_prx", 10);

  // Modules whose LLE module_start needs a backend we don't emulate, by how the
  // guest reacts to a failed load-start. kLoadOk: the app load-starts them directly
  // and ASSERTS success (Doom64), so report success with a real handle and skip the
  // LLE module_start (a preloaded dup whose IPMI init is scout-patched).
  static const char *kLoadOk[] = {"libSceAppContent"};
  // kSkipNotFound: libkernel PRELOADS these and strictly verifies they started,
  // aborting (0x80020064) on a load we can't start (module_start unresolved, e.g.
  // libSceNet faults on a __thread errno with no DTV TLS). Report genuine not-found
  // so the preloader skips them.
  static const char *kSkipNotFound[] = {"libSceSsl2", "libSceHttp2",
                                        "libSceNpManager", "libSceNpWebApi2"};
  // The skip list is PS4-only: on PS5 the net stack is up and these DT_INITs run fine
  // (Demon's Souls' Crossgen init asserts 0x0112 load-start succeeds, chaining
  // libSceHttp2/NpManager/NpWebApi2; libSceSsl2 has no PS5 module, degrades to not-found).
  auto *proc = proc::getActive();
  const bool isPs5 = proc->getPlatform() == krnl::proc::platform::ps5;
  bool skipInit = false;
  for (auto *s : kLoadOk) {
    if (std::strcmp(name.c_str(), s) == 0) {
      BASE_LOGI("load_prx", "{}: load-start ok, skipping LLE module_start", s);
      skipInit = true;
      break;
    }
  }
  if (!isPs5) {
    for (auto *s : kSkipNotFound) {
      if (std::strcmp(name.c_str(), s) == 0) {
        BASE_LOGI("load_prx", "{}: reporting not-found (init unsupported)", s);
        return -SysError::eNOENT;
      }
    }
  }

  // already loaded (we preload the system module tree): hand back its handle.
  auto mod = proc->getModule(base::StringRef(name));
  // PS5 ships thirteen sysmodules twice under ONE SONAME and the guest names them
  // inconsistently: the eboot needs "libSceVdecCore.prx" (registered as
  // "libSceVdecCore") while libSceSysmodule asks for "libSceVdecCore.native".
  // Missing that match loaded a SECOND copy and libSceSysmodule abandoned the rest
  // of the dependency list (why Astro Bot's libSceAvPlayer never came up). Let
  // either spelling find the other.
  if (!mod && isPs5) {
    constexpr size_t kNat = 7;  // ".native"
    base::String alt;
    if (name.length() > kNat &&
        std::strcmp(name.c_str() + name.length() - kNat, ".native") == 0)
      alt = name.substr(0, name.length() - kNat);
    else
      alt = name + ".native";
    mod = proc->getModule(base::StringRef(alt));
    if (mod)
      BASE_LOGI("load_prx", "{} is already loaded as {}", name.c_str(),
                alt.c_str());
  }
  if (!mod) {
    mod = proc->loadModule(base::StringRef(name));
    if (!mod) {
      LOG_ERROR("load_prx: unable to load {}", name.c_str());
      return -SysError::eNOENT;
    }
  }

  // Always relocate: a DT_NEEDED dep added by loadModule is never relocated, so its
  // init_array holds raw offsets and module_start calls a bad pointer. Idempotent.
  if ((!mod->resolveImports() || !mod->applyRelocations()) && !skipInit) {
    LOG_ERROR("load_prx: relocate failed for {}", name.c_str());
    return -SysError::eNOEXEC;
  }

  // A module loaded later can export what an earlier one's imports missed: libSceFontFt
  // needs libSceFreeTypeFull (not shipped); libSceFreeTypeOt exports the same 24 NIDs.
  // The console binds PLT slots at first call, so rebinding badcall slots is the same,
  // just eager.
  for (auto &other : proc->getModuleList())
    if (other.get() != mod.get() && other->hasUnresolvedImports())
      other->resolveImports();

  // Run DT_INIT (module_start) at load-start, as the kernel does: system modules
  // self-register their service (libSceVideoOut registers its display driver; without
  // it sceVideoOutOpen errors and LLE titles crash in their renderer). Some inits
  // fault on backend paths we don't emulate, so scope to the verified ones.
  static const char *kRunInit[] = {"libSceVideoOut"};
  for (auto *s : kRunInit) {
    if (!skipInit && std::strcmp(name.c_str(), s) == 0 && !mod->getInfo().initRan) {
      mod->getInfo().initRan = true;
      auto baseAddr = reinterpret_cast<uintptr_t>(mod->getInfo().base);
      // PS4 PRX carry module_start separately from DT_INIT (which is often 0 with
      // an empty init_array). DELTA_VO_INIT_OFF lets us point at the entry(ies)
      // by module offset (comma-separated, run in order) while pinning them.
      const char *list = kVoInitOff;
      base::String offs(list ? list : "");
      for (const char *p = offs.c_str(); *p;) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        char *end = nullptr;
        uintptr_t a = baseAddr + std::strtoull(p, &end, 0);
        p = end;
        BASE_LOGI("modinit", "running {} init @ +{:#x}", s, a - baseAddr);
        cpu::backend().runGuestFunction(a, 0, 0, 0);
        BASE_LOGI("modinit", "{} init @ returned", s);
      }
      // DELTA_VO_LLE_FIX: run libSceVideoOut's lazy init now (0xd530 sets display-config
      // defaults and tail-calls 0x28f0, opening /dev/dce and registering the driver into
      // cfg[0]). The driver would mark the MAIN display connected off a /dev/dce report
      // we don't emulate, so synthesize it: copy cfg[0] into cfg[idx], f0=4, and set the
      // scePthreadOnce guard so the first Open skips the ctor and reads our slot.
      if (kVoLleFix) {
        u8 *base = mod->getInfo().base;
        BASE_LOGI("volle", "running libSceVideoOut ctor (+0xd530)");
        cpu::backend().runGuestFunction(baseAddr + 0xd530, 0, 0, 0);
        i32 idx = *reinterpret_cast<i32 *>(base + 0x1cb40);
        u32 f0c = *reinterpret_cast<u32 *>(base + 0x1cb50);
        BASE_LOGI("volle", "after ctor: idx={} cfg[0].f0={:#x}", idx, f0c);
        if (idx >= 1 && idx < 8) {
          u8 *cfg0 = base + 0x1cb50;
          u8 *cfgi = base + 0x1cb50 + (size_t)idx * 0x140;
          std::memcpy(cfgi, cfg0, 0x140);
          *reinterpret_cast<u32 *>(cfgi) = 4;  // main display, connected
          // scePthreadOnce control @ +0x1cb18: mark "already run" so Open skips it.
          *reinterpret_cast<u32 *>(base + 0x1cb18) = 1;
          u64 op = *reinterpret_cast<u64 *>(cfgi + 0x48);
          BASE_LOGI("volle",
                    "connected cfg[{}] (f0=4), once-guard set; "
                    "cfg[idx].op@+0x48 = {:#x} (base+{:#x})",
                    idx, (unsigned long)op, (unsigned long)(op - (uintptr_t)base));
          // TEST (DELTA_VO_PATCH_OP): force the driver open-op to return a type-4
          // handle (0x40), so Open's (ret>>4)==cfg[idx].f0(4) && ret>1 checks pass.
          // Confirms whether the op return is the last gate before Open succeeds.
          (void)op;
        }
      }
    }
  }

  if (pHandle)
    *pHandle = mod->getInfo().handle;
  return 0;
}

// We never reclaim a loaded module (its code/data stay mapped for the process
// lifetime), so unload is a success no-op: the guest's module_stop has already
// run and it just wants the bookkeeping to succeed.
int PS4ABI sys_dynlib_unload_prx(u32 handle) {
  BASE_LOGI("unload_prx", "handle={:#x} (no-op)", handle);
  return 0;
}

// HLE of libkernel's __tls_get_addr (NID vNe1w4diLCs): its own dynamic-TLS allocator
// leaves DTV entries null, so general-dynamic __thread access faults. Resolve by
// TLS module index, hand back an init-image-populated block (+ offset). Single
// boot thread => one block per module, cached.
struct tls_index {
  u64 module_id;
  u64 offset;
};

void *PS4ABI guest_tls_get_addr(tls_index *ti) {
  // per-thread dynamic TLS blocks (module index -> block). Each thread gets its
  // own copy of every module's __thread storage, like a real DTV.
  static thread_local std::unordered_map<u32, u8 *> t_blocks;

  auto it = t_blocks.find(ti->module_id);
  if (it != t_blocks.end())
    return it->second + ti->offset;

  auto *proc = proc::getActive();
  for (auto &mod : proc->getModuleList()) {
    auto &info = mod->getInfo();
    if (info.tlsSlot != ti->module_id)
      continue;

    size_t sz = info.tlsSizeMem ? info.tlsSizeMem : 1;
    auto *block = static_cast<u8 *>(std::calloc(1, sz));
    if (info.tlsAddr && info.tlsSizeFile)
      std::memcpy(block, info.tlsAddr, info.tlsSizeFile);
    t_blocks[ti->module_id] = block;
    return block + ti->offset;
  }

  BASE_LOGI("tls", "__tls_get_addr: no module for index {}",
            (unsigned long long)ti->module_id);
  return nullptr;
}

int PS4ABI sys_dynlib_process_needed_and_relocate() {
  auto &list = proc::getActive()->getModuleList();
  for (auto &mod : list) {
    LOG_ASSERT(mod);

    if (!mod->resolveImports() || !mod->applyRelocations()) {
      LOG_ERROR("failed to apply relocations for module {}",
                mod->getInfo().name.c_str());
      return -1;
    }
  }

  return 0;
}
} // namespace krnl
