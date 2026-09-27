
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"
#include "guest_abi.h"
#include "kern/crash.h"
#include "logger/logger.h"

#include "cpu/backend.h"
#include "host_memory/host_memory.h"
#include "kern/module.h"
#include "kern/process.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_dynlib.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"
#include "kern/lv2/sys_mem.h"
#include "options/options.h"
#include "runtime/vprx/nid.h"
#include "runtime/vprx/vprx.h"

namespace {
DELTA_OPTION(const char*, kVoInitOff, "DELTA_VO_INIT_OFF", nullptr);
DELTA_OPTION(bool, kModinitTrace, "DELTA_MODINIT_TRACE", false);
DELTA_OPTION(bool, kProcparamTrace, "DELTA_PROCPARAM_TRACE", false);
DELTA_OPTION(bool, kVoLleFix, "DELTA_VO_LLE_FIX", false);
}  // namespace

namespace kern {
int PS4ABI sys_dynlib_dlopen(const char*) {
  /* devkit-only; retail titles never call it */
  return -SysError::eNOSYS;
}

int PS4ABI sys_dynlib_get_info(u32 handle, dynlib_info* dyn_info) {
  if (!dyn_info)
    return -SysError::eFAULT;
  if (dyn_info->size != sizeof(*dyn_info))
    return -SysError::eINVAL;

  auto mod = Process::GetActive()->GetModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto& info = mod->GetInfo();
  std::memset(dyn_info, 0, sizeof(dynlib_info));
  std::strncpy(dyn_info->name, info.name.c_str(), 256);

  auto& text = dyn_info->segs[0];
  text.addr = reinterpret_cast<uintptr_t>(info.text_seg.addr);
  text.size = info.text_seg.size;
  text.flags = 1 | 4;

  auto& data = dyn_info->segs[1];
  data.addr = reinterpret_cast<uintptr_t>(info.data_seg.addr);
  data.size = info.data_seg.size;
  data.flags = 1 | 2;

  dyn_info->seg_count = 2;

  std::memcpy(dyn_info->fingerprint, info.fingerprint, 20);
  return 0;
}

int PS4ABI sys_dynlib_get_info_ex(u32 handle,
                                  i32 ukn /*always 1*/,
                                  dynlib_info_ex* dyn_info) {
  if (!dyn_info)
    return -SysError::eFAULT;
  if (dyn_info->size != sizeof(*dyn_info))
    return -SysError::eINVAL;

  auto mod = Process::GetActive()->GetModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto& info = mod->GetInfo();
  std::memset(dyn_info, 0, sizeof(dynlib_info_ex));

  dyn_info->size = sizeof(dynlib_info_ex);
  dyn_info->handle = info.handle;
  std::strncpy(dyn_info->name, info.name.c_str(), 256);

  dyn_info->tls_index = info.tls_slot;
  dyn_info->tls_align = info.tlsalign;
  dyn_info->tls_init_size = info.tls_size_file;
  dyn_info->tls_size = info.tls_size_mem;
  dyn_info->tls_init_addr = reinterpret_cast<uintptr_t>(info.tls_addr);

  dyn_info->init_proc_addr = reinterpret_cast<uintptr_t>(info.init_addr);
  dyn_info->fini_proc_addr = reinterpret_cast<uintptr_t>(info.fini_addr);

  // InstallEhFrame stores the hdr (PT_GNU_EH_FRAME) in ehFrameAddr and the
  // unwind data (.eh_frame) in ehFrameheaderAddr, i.e. named backwards; report
  // as the guest unwinder expects.
  dyn_info->eh_frame_addr =
      reinterpret_cast<uintptr_t>(info.eh_frameheader_addr);
  dyn_info->eh_frame_hdr_addr = reinterpret_cast<uintptr_t>(info.eh_frame_addr);
  dyn_info->eh_frame_size = info.eh_frameheader_size;
  dyn_info->eh_frame_hdr_size = info.eh_frame_size;
  BASE_LOGI("get_info_ex", "h={} {} eh_frame={:#x}+{:#x} hdr={:#x}+{:#x}",
            handle, info.name.c_str(), dyn_info->eh_frame_addr,
            dyn_info->eh_frame_size, dyn_info->eh_frame_hdr_addr,
            dyn_info->eh_frame_hdr_size);

  auto& text = dyn_info->segs[0];
  text.addr = reinterpret_cast<uintptr_t>(info.text_seg.addr);
  text.size = info.text_seg.size;
  text.flags = 1 | 4;

  auto& data = dyn_info->segs[1];
  data.addr = reinterpret_cast<uintptr_t>(info.data_seg.addr);
  data.size = info.data_seg.size;
  data.flags = 1 | 2;

  dyn_info->seg_count = 2;
  dyn_info->ref_count = 1;
  return 0;
}

int PS4ABI sys_dynlib_dlsym(u32 handle, const char* sym_name, void** sym) {
  auto mod = Process::GetActive()->GetModule(handle);
  if (!mod)
    return -1;

  char nameenc[12]{};
  runtime::nid::Encode(sym_name, reinterpret_cast<u8*>(&nameenc));

  auto& mod_name = mod->GetInfo().name;

  // dlsym resolves an EXPORT of the target module by its NID. ResolveObfSymbol
  // is for imports (it decodes "#libid#modid"); here we match the 11-char NID
  // directly against the module's export table.
  uintptr_t addr_out = mod->GetSymbolByNid(nameenc);

  // Callers may pass an already-encoded 11-char NID (e.g. the SDK module-entry
  // symbol "BaOKcng8g88") instead of a plain name. nid::Encode() would hash the
  // NID string itself and never match, so also try matching it verbatim.
  if (!addr_out && std::strlen(sym_name) == 11)
    addr_out = mod->GetSymbolByNid(sym_name);

  if (!addr_out) {
    BASE_LOGI("dlsym", "{}!{} -> UNRESOLVED", mod_name.c_str(), sym_name);
    *sym = nullptr;
    return -1;
  }

  BASE_LOGI("dlsym", "{}!{} -> {:p} (+{:#x})", mod_name.c_str(), sym_name,
            reinterpret_cast<void*>(addr_out),
            addr_out - reinterpret_cast<uintptr_t>(mod->GetInfo().base));

  *sym = reinterpret_cast<void*>(addr_out);

  return 0;
}

int PS4ABI sys_dynlib_get_obj_member(u32 handle, u8 index, void** value) {
  auto mod = Process::GetActive()->GetModule(handle);
  if (!mod)
    return -SysError::eSRCH;

  auto& info = mod->GetInfo();
  switch (index) {
    case 1:  // module init proc
      *value = info.init_addr;
      if (kModinitTrace)
        BASE_LOGI("modinit", "h={} {} init={:p}", handle, info.name.c_str(),
                  info.init_addr);
      return 0;
    case 2:  // module fini proc
      *value = info.fini_addr;
      return 0;
    case 8:  // SCE module param (libSceSysmodule reads the SDK version from it)
      *value = info.module_param;
      return 0;
    default:
      LOG_WARNING("get_obj_member: unhandled index {} for {}", index,
                  info.name.c_str());
      return -SysError::eINVAL;
  }
}

int PS4ABI sys_dynlib_get_proc_param(void** data, size_t* size) {
  auto mod = Process::GetActive()->GetModuleList()[0];
  if (mod) {
    auto& info = mod->GetInfo();

    *data = reinterpret_cast<void*>(info.proc_param);
    *size = info.proc_param_size;
    if (kProcparamTrace)
      BASE_LOGI("procparam", "get_proc_param -> data={:p} size={:#x}", *data,
                *size);
    return 0;
  }

  *data = nullptr;
  *size = 0;

  return 0;
}

int PS4ABI sys_dynlib_get_list(u32* handles, size_t max_count, size_t* count) {
  auto* proc = Process::GetActive();
  auto& list = proc->GetModuleList();

  // The real kernel loads each PRX's needed modules first, so its list is
  // dependency-ordered and libkernel's init walker initializes a library before
  // its dependents. Our discovery order can invert that: SOTTR's NpManager
  // PrxStart ran before libSceHttp's init, got 0x80431001, left its NP context
  // null. Emit dependency-first (postorder over DT_SCE_NEEDED_MODULE), main
  // module up front.
  base::Vector<Module*> sorted;
  base::HashSet<Module*> visited;
  base::Function<void(Module*)> visit = [&](Module* m) {
    if (!visited.insert(m).second)
      return;
    for (auto& dep : m->NeededObjects()) {
      auto d = proc->GetModule(base::StringRef(dep.c_str()));
      if (d && d.get() != m)
        visit(d.get());
    }
    sorted.push_back(m);
  };
  for (auto& mod : list)
    visit(mod.get());

  size_t list_count = 0;
  if (!list.empty() && !list[0]->GetInfo().name.empty() && max_count > 0) {
    *(handles++) = list[0]->GetInfo().handle;  // main module first
    list_count++;
  }
  for (auto* mod : sorted) {
    if (!list.empty() && mod == list[0].get())
      continue;
    if (mod->GetInfo().name.empty())
      continue;
    if (list_count >= max_count)
      break;  // don't overflow the caller's buffer
    *(handles++) = mod->GetInfo().handle;
    list_count++;
  }

  *count = list_count;
  return 0;
}

// syscall 594. libkernel's _sceKernelLoadStartModule path calls this with
// rdi=path, rsi=flags, rdx=&handle. It expects the loaded module's handle
// written to *pHandle and 0 returned on success.
int PS4ABI sys_dynlib_load_prx(const char* path,
                               u64 flags,
                               int* p_handle,
                               u64 arg4,
                               const void* opt,
                               i64* p_res) {
  if (p_res)
    *p_res = 0;
  if (!path)
    return -SysError::eINVAL;

  // Derive the module name from the path: basename minus its extension. The
  // module's internal name (set from the SCE module info) matches this for the
  // system libs we preload, so GetModule() can find an already-loaded one.
  const char* slash = std::strrchr(path, '/');
  const char* base_start = slash ? slash + 1 : path;
  const char* dot = std::strrchr(base_start, '.');
  size_t name_len =
      dot ? static_cast<size_t>(dot - base_start) : std::strlen(base_start);
  base::String name;
  name.append(base_start, name_len);

  BASE_LOGI("load_prx", "path='{}' -> '{}' flags={:#x}", path, name.c_str(),
            (unsigned long long)flags);
  // DELTA_LOADPRX_STACK: who asked. A sysmodule the guest load-starts and one
  // it never asks for want different answers, and only the caller says which.
  if (std::getenv("DELTA_LOADPRX_STACK"))
    GuestStackTrace("load_prx", 10);

  // Modules whose LLE module_start needs a backend we don't emulate, by how the
  // guest reacts to a failed load-start. kLoadOk: the app load-starts them
  // directly and ASSERTS success (Doom64), so report success with a real handle
  // and skip the LLE module_start (a preloaded dup whose IPMI init is
  // scout-patched).
  static const char* kLoadOk[] = {"libSceAppContent"};
  // kSkipNotFound: libkernel PRELOADS these and strictly verifies they started,
  // aborting (0x80020064) on a load we can't start (module_start unresolved,
  // e.g. libSceNet faults on a __thread errno with no DTV TLS). Report genuine
  // not-found so the preloader skips them.
  static const char* kSkipNotFound[] = {"libSceSsl2", "libSceHttp2",
                                        "libSceNpManager", "libSceNpWebApi2"};
  // The skip list is PS4-only: on PS5 the net stack is up and these DT_INITs
  // run fine (Demon's Souls' Crossgen init asserts 0x0112 load-start succeeds,
  // chaining libSceHttp2/NpManager/NpWebApi2; libSceSsl2 has no PS5 module,
  // degrades to not-found).
  auto* proc = Process::GetActive();
  const bool is_ps5 = proc->GetPlatform() == kern::Process::Platform::kPs5;
  bool skip_init = false;
  for (auto* s : kLoadOk) {
    if (std::strcmp(name.c_str(), s) == 0) {
      BASE_LOGI("load_prx", "{}: load-start ok, skipping LLE module_start", s);
      skip_init = true;
      break;
    }
  }
  if (!is_ps5) {
    for (auto* s : kSkipNotFound) {
      if (std::strcmp(name.c_str(), s) == 0) {
        BASE_LOGI("load_prx", "{}: reporting not-found (init unsupported)", s);
        return -SysError::eNOENT;
      }
    }
  }

  // already loaded (we preload the system module tree): hand back its handle.
  auto mod = proc->GetModule(base::StringRef(name));
  // PS5 ships thirteen sysmodules twice under ONE SONAME and the guest names
  // them inconsistently: the eboot needs "libSceVdecCore.prx" (registered as
  // "libSceVdecCore") while libSceSysmodule asks for "libSceVdecCore.native".
  // Missing that match loaded a SECOND copy and libSceSysmodule abandoned the
  // rest of the dependency list (why Astro Bot's libSceAvPlayer never came up).
  // Let either spelling find the other.
  if (!mod && is_ps5) {
    constexpr size_t kNat = 7;  // ".native"
    base::String alt;
    if (name.length() > kNat &&
        std::strcmp(name.c_str() + name.length() - kNat, ".native") == 0)
      alt = name.substr(0, name.length() - kNat);
    else
      alt = name + ".native";
    mod = proc->GetModule(base::StringRef(alt));
    if (mod)
      BASE_LOGI("load_prx", "{} is already loaded as {}", name.c_str(),
                alt.c_str());
  }
  if (!mod) {
    mod = proc->LoadModule(base::StringRef(name));
    if (!mod) {
      LOG_ERROR("load_prx: unable to load {}", name.c_str());
      return -SysError::eNOENT;
    }
  }

  // Always relocate: a DT_NEEDED dep added by LoadModule is never relocated, so
  // its init_array holds raw offsets and module_start calls a bad pointer.
  // Idempotent.
  if ((!mod->ResolveImports() || !mod->ApplyRelocations()) && !skip_init) {
    LOG_ERROR("load_prx: relocate failed for {}", name.c_str());
    return -SysError::eNOEXEC;
  }

  // A module loaded later can export what an earlier one's imports missed:
  // libSceFontFt needs libSceFreeTypeFull (not shipped); libSceFreeTypeOt
  // exports the same 24 NIDs. The console binds PLT slots at first call, so
  // rebinding badcall slots is the same, just eager.
  for (auto& other : proc->GetModuleList())
    if (other.get() != mod.get() && other->HasUnresolvedImports())
      other->ResolveImports();

  // Run DT_INIT (module_start) at load-start, as the kernel does: system
  // modules self-register their service (libSceVideoOut registers its display
  // driver; without it sceVideoOutOpen errors and LLE titles crash in their
  // renderer). Some inits fault on backend paths we don't emulate, so scope to
  // the verified ones.
  static const char* kRunInit[] = {"libSceVideoOut"};
  for (auto* s : kRunInit) {
    if (!skip_init && std::strcmp(name.c_str(), s) == 0 &&
        !mod->GetInfo().init_ran) {
      mod->GetInfo().init_ran = true;
      auto base_addr = reinterpret_cast<uintptr_t>(mod->GetInfo().base);
      // PS4 PRX carry module_start separately from DT_INIT (which is often 0
      // with an empty init_array). DELTA_VO_INIT_OFF lets us point at the
      // entry(ies) by module offset (comma-separated, run in order) while
      // pinning them.
      const char* list = kVoInitOff;
      base::String offs(list ? list : "");
      for (const char* p = offs.c_str(); *p;) {
        while (*p == ',' || *p == ' ')
          p++;
        if (!*p)
          break;
        char* end = nullptr;
        uintptr_t a = base_addr + std::strtoull(p, &end, 0);
        p = end;
        BASE_LOGI("modinit", "running {} init @ +{:#x}", s, a - base_addr);
        cpu::GetBackend().RunGuestFunction(a, 0, 0, 0);
        BASE_LOGI("modinit", "{} init @ returned", s);
      }
      // DELTA_VO_LLE_FIX: run libSceVideoOut's lazy init now (0xd530 sets
      // display-config defaults and tail-calls 0x28f0, opening /dev/dce and
      // registering the driver into cfg[0]). The driver would mark the MAIN
      // display connected off a /dev/dce report we don't emulate, so synthesize
      // it: copy cfg[0] into cfg[idx], f0=4, and set the scePthreadOnce guard
      // so the first Open skips the ctor and reads our slot.
      if (kVoLleFix) {
        u8* base = mod->GetInfo().base;
        BASE_LOGI("volle", "running libSceVideoOut ctor (+0xd530)");
        cpu::GetBackend().RunGuestFunction(base_addr + 0xd530, 0, 0, 0);
        i32 idx = *reinterpret_cast<i32*>(base + 0x1cb40);
        u32 f0c = *reinterpret_cast<u32*>(base + 0x1cb50);
        BASE_LOGI("volle", "after ctor: idx={} cfg[0].f0={:#x}", idx, f0c);
        if (idx >= 1 && idx < 8) {
          u8* cfg0 = base + 0x1cb50;
          u8* cfgi = base + 0x1cb50 + (size_t)idx * 0x140;
          std::memcpy(cfgi, cfg0, 0x140);
          *reinterpret_cast<u32*>(cfgi) = 4;  // main display, connected
          // scePthreadOnce control @ +0x1cb18: mark "already run" so Open skips
          // it.
          *reinterpret_cast<u32*>(base + 0x1cb18) = 1;
          u64 op = *reinterpret_cast<u64*>(cfgi + 0x48);
          BASE_LOGI("volle",
                    "connected cfg[{}] (f0=4), once-guard set; "
                    "cfg[idx].op@+0x48 = {:#x} (base+{:#x})",
                    idx, (unsigned long)op,
                    (unsigned long)(op - (uintptr_t)base));
          // TEST (DELTA_VO_PATCH_OP): force the driver open-op to return a
          // type-4 handle (0x40), so Open's (ret>>4)==cfg[idx].f0(4) && ret>1
          // checks pass. Confirms whether the op return is the last gate before
          // Open succeeds.
          (void)op;
        }
      }
    }
  }

  if (p_handle)
    *p_handle = mod->GetInfo().handle;
  return 0;
}

// We never reclaim a loaded module (its code/data stay mapped for the process
// lifetime), so unload is a success no-op: the guest's module_stop has already
// run and it just wants the bookkeeping to succeed.
int PS4ABI sys_dynlib_unload_prx(u32 handle) {
  BASE_LOGI("unload_prx", "handle={:#x} (no-op)", handle);
  return 0;
}

// HLE of libkernel's __tls_get_addr (NID vNe1w4diLCs): its own dynamic-TLS
// allocator leaves DTV entries null, so general-dynamic __thread access faults.
// Resolve by TLS module index, hand back an init-image-populated block (+
// offset). Single boot thread => one block per module, cached.
struct tls_index {  // NOLINT(readability-identifier-naming): guest ABI name
  u64 module_id;
  u64 offset;
};

void* PS4ABI GuestTlsGetAddr(tls_index* ti) {
  // per-thread dynamic TLS blocks (module index -> block). Each thread gets its
  // own copy of every module's __thread storage, like a real DTV.
  static thread_local base::HashMap<u32, u8*> t_blocks;

  auto it = t_blocks.find(ti->module_id);
  if (it != t_blocks.end())
    return it->second + ti->offset;

  auto* proc = Process::GetActive();
  for (auto& mod : proc->GetModuleList()) {
    auto& info = mod->GetInfo();
    if (info.tls_slot != ti->module_id)
      continue;

    size_t sz = info.tls_size_mem ? info.tls_size_mem : 1;
    auto* block = static_cast<u8*>(std::calloc(1, sz));
    if (info.tls_addr && info.tls_size_file)
      std::memcpy(block, info.tls_addr, info.tls_size_file);
    t_blocks[ti->module_id] = block;
    return block + ti->offset;
  }

  BASE_LOGI("tls", "__tls_get_addr: no module for index {}",
            (unsigned long long)ti->module_id);
  return nullptr;
}

int PS4ABI sys_dynlib_process_needed_and_relocate() {
  auto& list = Process::GetActive()->GetModuleList();
  for (auto& mod : list) {
    LOG_ASSERT(mod);

    if (!mod->ResolveImports() || !mod->ApplyRelocations()) {
      LOG_ERROR("failed to apply relocations for module {}",
                mod->GetInfo().name.c_str());
      return -1;
    }
  }

  return 0;
}

// sys_dl_get_list/get_info: gated on a debugger/coredump/syscore process; a
// retail title gets EPERM (the dbglogger system process is the only legitimate
// enumerator). Arg block {pid@0, ids[]@8, max@16, count@24}.
int PS4ABI sys_dl_get_list() {
  return -SysError::ePERM;
}

// Kernel arg block: {pid@0, handle@8, info@16}; fills a 0xA50-byte module-info
// struct. Same debugger gate, same EPERM for a title.
int PS4ABI sys_dl_get_info() {
  return -SysError::ePERM;
}

// The kernel's sys_dl_notify_event returns ENOSYS unconditionally: dynlib
// event delivery to a debugger is not wired on the console either.
int PS4ABI sys_dl_notify_event() {
  return -SysError::eNOSYS;
}

int PS4ABI sys_dynlib_dlclose() {
  return 0;
}
int PS4ABI sys_dynlib_prepare_dlclose() {
  return 0;
}

// Same debugger gate as sys_dl_get_list/get_info; arg block is
// {pid@0, handle@8, meta@16, metasize@24, sizeOut@32}.
int PS4ABI sys_dl_get_metadata() {
  return -SysError::ePERM;
}

int PS4ABI sys_dynlib_get_info_for_libdbg() {
  return 0;
}
int PS4ABI sys_dynlib_get_list_for_libdbg() {
  return 0;
}
int PS4ABI sys_dynlib_get_list2() {
  return 0;
}
int PS4ABI sys_dynlib_get_info2() {
  return 0;
}

// sys_dynlib_do_copy_relocations (596): processes R_X86_64_COPY relocations for
// the main executable. Our loader already resolves data relocations when it
// maps each module, so there is nothing extra to copy here; return success.
int PS4ABI sys_dynlib_do_copy_relocations() {
  return 0;
}

}  // namespace kern
