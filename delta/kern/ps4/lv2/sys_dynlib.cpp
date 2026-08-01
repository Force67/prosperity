
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <logger/logger.h>

#include <utl/mem.h>
#include "cpu/cpu_backend.h"
#include "../../module.h"
#include "../../proc.h"

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
void ps5MaybeInterposePthreadAlloc();  // proc.cpp: libc-mutex bootstrap fix

int PS4ABI sys_dynlib_dlopen(const char *) {
  /*TODO: implement, however note that this function is only
  present in devkits*/
  return -SysError::eNOSYS;
}

int PS4ABI sys_dynlib_get_info(uint32_t handle, dynlib_info *dyn_info) {
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

int PS4ABI sys_dynlib_get_info_ex(uint32_t handle, int32_t ukn /*always 1*/,
                                  dynlib_info_ex *dyn_info) {
  ps5MaybeInterposePthreadAlloc();  // frequent main-thread init call (see load_prx)
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

  // installEHFrame stores the *hdr* (PT_GNU_EH_FRAME) in ehFrameAddr/Size and
  // the actual unwind data (.eh_frame) in ehFrameheaderAddr/Size, i.e. the
  // fields are named backwards. Report them the way the guest unwinder expects:
  // eh_frame_addr = .eh_frame, eh_frame_hdr_addr = .eh_frame_hdr.
  dyn_info->eh_frame_addr =
      reinterpret_cast<uintptr_t>(info.ehFrameheaderAddr);
  dyn_info->eh_frame_hdr_addr = reinterpret_cast<uintptr_t>(info.ehFrameAddr);
  dyn_info->eh_frame_size = info.ehFrameheaderSize;
  dyn_info->eh_frame_hdr_size = info.ehFrameSize;
  std::printf("[get_info_ex] h=%u %s eh_frame=%#lx+%#x hdr=%#lx+%#x\n",
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

int PS4ABI sys_dynlib_dlsym(uint32_t handle, const char *symName, void **sym) {
  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -1;

  char nameenc[12]{};
  runtime::encode_nid(symName, reinterpret_cast<uint8_t *>(&nameenc));

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
    std::printf("DLSYM %s!%s -> UNRESOLVED\n", modName.c_str(), symName);
    *sym = nullptr;
    return -1;
  }

  std::printf("DLSYM %s!%s -> %p (+%#lx)\n", modName.c_str(), symName,
              reinterpret_cast<void *>(addrOut),
              addrOut - reinterpret_cast<uintptr_t>(mod->getInfo().base));

  *sym = reinterpret_cast<void *>(addrOut);

  return 0;
}

int PS4ABI sys_dynlib_get_obj_member(uint32_t handle, uint8_t index,
                                     void **value) {
  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto &info = mod->getInfo();
  switch (index) {
  case 1:  // module init proc
    *value = info.initAddr;
    if (kModinitTrace)
      std::fprintf(stderr, "[modinit] h=%u %s init=%p\n", handle,
                   info.name.c_str(), info.initAddr);
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
      std::fprintf(stderr, "[procparam] get_proc_param -> data=%p size=%#zx\n",
                   *data, *size);
    return 0;
  }

  *data = nullptr;
  *size = 0;

  return 0;
}

int PS4ABI sys_dynlib_get_list(uint32_t *handles, size_t maxCount,
                               size_t *count) {
  auto *proc = proc::getActive();
  auto &list = proc->getModuleList();

  // The real kernel loads each PRX's needed modules before the PRX itself, so
  // the module list is dependency-ordered and libkernel's module-init walker
  // (which runs inits in list order) always initializes a library before its
  // dependents. Our list is discovery order, which can invert that: SOTTR's
  // eboot names libSceNpManager before libSceHttp, so NpManager's PrxStart ran
  // first, called sceHttpInit before libSceHttp's own init, got 0x80431001
  // (before-init), and left its NP context null (later deref'd by Matching2).
  // Emit the list dependency-first (postorder over DT_SCE_NEEDED_MODULE), with
  // the main module kept up front like the real list.
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
int PS4ABI sys_dynlib_load_prx(const char *path, uint64_t flags, int *pHandle,
                               uint64_t arg4, const void *opt, int64_t *pRes) {
  // Main-thread module loading runs after libc init (so libkernel's pthread-state
  // allocator pointer is populated) but before the multithreaded malloc-mutex
  // bootstrap that would recurse; interpose it here (idempotent, PS5+native).
  ps5MaybeInterposePthreadAlloc();
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

  std::printf("[load_prx] path='%s' -> '%s' flags=%#llx\n", path, name.c_str(),
              (unsigned long long)flags);

  // Modules whose LLE module_start needs a backend we don't emulate yet fall into
  // two groups by how the guest reacts to a failed load-start.
  //
  // kLoadOk: the application itself load-starts these directly and *asserts* that
  // it succeeded (Doom64: `sceSysmoduleLoadModule(SCE_SYSMODULE_APP_CONTENT) ==
  // SCE_OK`), aborting on any signalled failure. Report load-start SUCCESS with a
  // real handle and merely skip running the LLE module_start (it's a preloaded
  // dup of libSceAppContentUtil whose IPMI init is already scout-patched). These
  // "worked" before only because the FEX syscall bridge used to drop the BSD
  // carry flag so every errno read as success; now that the bridge signals carry
  // correctly (needed for genuine error returns) a -ENOENT here aborts the title.
  static const char *kLoadOk[] = {"libSceAppContent"};
  // kSkipNotFound: libkernel *preloads* these via sceSysmodulePreloadModuleFor-
  // Libkernel, which strictly verifies the module actually STARTED and aborts
  // ("cannot be loaded", 0x80020064) if we report a load we then can't start
  // (their module_start is unresolved, so we can't run it, e.g. libSceNet's init
  // faults on a __thread errno whose TLS isn't in the DTV). Report a genuine
  // "not found" so the preloader treats the sysmodule as absent and skips it.
  static const char *kSkipNotFound[] = {"libSceSsl2", "libSceHttp2",
                                        "libSceNpManager", "libSceNpWebApi2"};
  bool skipInit = false;
  for (auto *s : kLoadOk) {
    if (std::strcmp(name.c_str(), s) == 0) {
      std::printf("[load_prx] %s: load-start ok, skipping LLE module_start\n", s);
      skipInit = true;
      break;
    }
  }
  for (auto *s : kSkipNotFound) {
    if (std::strcmp(name.c_str(), s) == 0) {
      std::printf("[load_prx] %s: reporting not-found (init unsupported)\n", s);
      return -SysError::eNOENT;
    }
  }

  auto *proc = proc::getActive();

  // already loaded (we preload the system module tree): hand back its handle.
  auto mod = proc->getModule(base::StringRef(name));
  if (!mod) {
    mod = proc->loadModule(base::StringRef(name));
    if (!mod) {
      LOG_ERROR("load_prx: unable to load {}", name.c_str());
      return -SysError::eNOENT;
    }
  }

  // Always relocate: a module pulled in as another module's DT_NEEDED dep is
  // added to the list by loadModule but never relocated, so its init_array
  // holds raw offsets and module_start calls a bad pointer. applyRelocations
  // is idempotent (guarded), so re-running it on an already-relocated module
  // is a no-op.
  if ((!mod->resolveImports() || !mod->applyRelocations()) && !skipInit) {
    LOG_ERROR("load_prx: relocate failed for {}", name.c_str());
    return -SysError::eNOEXEC;
  }

  // Run the module's DT_INIT (module_start) now, as the real kernel does during
  // load-start. The system modules ship with constructors that self-register
  // their service with the kernel devices; e.g. the real libSceVideoOut registers
  // its display driver here, without which sceVideoOutOpen returns an error (its
  // internal display-config table stays empty) and titles that run it LLE crash
  // in their renderer. We don't run every module's init (some take backend paths
  // we don't emulate and fault); scope it to the ones we've verified.
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
        std::printf("[modinit] running %s init @ +%#lx\n", s, a - baseAddr);
        cpu::backend().runGuestFunction(a, 0, 0, 0);
        std::printf("[modinit] %s init @ returned\n", s);
      }
      // DELTA_VO_LLE_FIX: run libSceVideoOut's lazy init now (its 0xd530 ctor sets
      // the display-config defaults + tail-calls 0x28f0 which opens /dev/dce and
      // registers the driver into cfg[0]). The real driver would then mark the
      // MAIN display connected (cfg[idx].f0=4) off a /dev/dce report we don't yet
      // emulate, so synthesize it: copy the registered cfg[0] slot into cfg[idx]
      // and set f0=4. Finally set the scePthreadOnce guard so the title's first
      // sceVideoOutOpen skips re-running the ctor and reads our connected slot.
      if (kVoLleFix) {
        uint8_t *base = mod->getInfo().base;
        std::printf("[volle] running libSceVideoOut ctor (+0xd530)\n");
        cpu::backend().runGuestFunction(baseAddr + 0xd530, 0, 0, 0);
        int32_t idx = *reinterpret_cast<int32_t *>(base + 0x1cb40);
        uint32_t f0c = *reinterpret_cast<uint32_t *>(base + 0x1cb50);
        std::printf("[volle] after ctor: idx=%d cfg[0].f0=%#x\n", idx, f0c);
        if (idx >= 1 && idx < 8) {
          uint8_t *cfg0 = base + 0x1cb50;
          uint8_t *cfgi = base + 0x1cb50 + (size_t)idx * 0x140;
          std::memcpy(cfgi, cfg0, 0x140);
          *reinterpret_cast<uint32_t *>(cfgi) = 4;  // main display, connected
          // scePthreadOnce control @ +0x1cb18: mark "already run" so Open skips it.
          *reinterpret_cast<uint32_t *>(base + 0x1cb18) = 1;
          uint64_t op = *reinterpret_cast<uint64_t *>(cfgi + 0x48);
          std::printf("[volle] connected cfg[%d] (f0=4), once-guard set; "
                      "cfg[idx].op@+0x48 = %#lx (base+%#lx)\n",
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
int PS4ABI sys_dynlib_unload_prx(uint32_t handle) {
  std::printf("[unload_prx] handle=%#x (no-op)\n", handle);
  return 0;
}

// HLE of libkernel's __tls_get_addr (NID vNe1w4diLCs). libkernel's own dynamic
// TLS allocator leaves the per-thread DTV entries null, so general-dynamic
// __thread access (e.g. libc's malloc arena, module errno) reads address 0 and
// faults. We resolve the module by its TLS module index and hand back a real,
// init-image-populated block. tls_index = {module_id, offset}; the result is
// block + offset. Single boot thread => one block per module, cached.
struct tls_index {
  uint64_t module_id;
  uint64_t offset;
};

void *PS4ABI guest_tls_get_addr(tls_index *ti) {
  // per-thread dynamic TLS blocks (module index -> block). Each thread gets its
  // own copy of every module's __thread storage, like a real DTV.
  static thread_local std::unordered_map<uint32_t, uint8_t *> t_blocks;

  auto it = t_blocks.find(ti->module_id);
  if (it != t_blocks.end())
    return it->second + ti->offset;

  auto *proc = proc::getActive();
  for (auto &mod : proc->getModuleList()) {
    auto &info = mod->getInfo();
    if (info.tlsSlot != ti->module_id)
      continue;

    size_t sz = info.tlsSizeMem ? info.tlsSizeMem : 1;
    auto *block = static_cast<uint8_t *>(std::calloc(1, sz));
    if (info.tlsAddr && info.tlsSizeFile)
      std::memcpy(block, info.tlsAddr, info.tlsSizeFile);
    t_blocks[ti->module_id] = block;
    return block + ti->offset;
  }

  std::printf("[tls] __tls_get_addr: no module for index %llu\n",
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
