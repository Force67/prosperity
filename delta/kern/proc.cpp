/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <utl/file.h>
#include <utl/mem.h>
#include <utl/path.h>

#include "crash.h"
#include "module.h"
#include "proc.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include "lv2/sys_dynlib.h"
#include "lv2/sys_mem.h"
#include "runtime/vprx/vprx.h"

#include <cstring>

namespace krnl {
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

  /*pre-load required modules
   (the kernel does it, so do we)*/
  if (!loadModule(base::StringRef("libkernel")) ||
      !loadModule(base::StringRef("libSceLibcInternal"))) {
    LOG_ERROR("unable to preload sys modules");
    return false;
  }

  // libkernel is a thin forwarder: a chunk of its exports (the memory-pool
  // helpers libSceSaveData & friends import as "libkernel") actually live in
  // libkernel_sys. Preload it so those NIDs resolve via the cross-module
  // fallback in resolveObfSymbol; tolerate absence on minimal module sets.
  loadModule(base::StringRef("libkernel_sys"));

  bool loaded = fromVfs ? first->fromVfs(path) : first->fromFile(path);
  if (!loaded) {
    LOG_ERROR("unable to load main process module");
    return false;
  }

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

modulePtr proc::getModule(uint32_t handle) {
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
// Vulkan graphics device, which isn't brought up yet, so the bucket array is
// never allocated and every registry find/insert dereferences null + idx*0x20
// (rebirth+0x1e8c8b on a worker insert, +0x1e7e09 on a main-thread find). Until
// the real gfx/renderer init exists, hand the registry valid empty storage up
// front: allocate the N=0x20 * 0x20-byte zeroed bucket array, point the registry
// at it, and NOP the ctor's null-write so it can't clobber the pointer when it
// later runs. The map then works as a valid empty registry independent of order
// or the missing gfx init. (Verified offsets via the decrypted rebirth.elf.)
static void bringUpRebirthSurfaceRegistry(smodule &m) {
  uint8_t *base = m.getInfo().base;
  constexpr uint32_t kRegistryOff = 0x687a90; // bucket-array base pointer
  constexpr uint32_t kCtorZeroOff = 0x1e9bcd; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  uint8_t *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth surface-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<uint64_t *>(base + kRegistryOff) =
      reinterpret_cast<uint64_t>(buckets);

  uint8_t *ctor = base + kCtorZeroOff;
  utl::protectMem(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ctor) & ~0xFFFull),
      0x1000, utl::pageProtection::rwx);
  std::memset(ctor, 0x90, 11); // NOP the ctor's null-write

  LOG_INFO("rebirth surface-registry: installed empty buckets@{} -> [+{:#x}]",
           (void *)buckets, kRegistryOff);
}

modulePtr proc::loadModule(base::StringRef name) {
  auto mod = getModule(name);
  if (mod)
    return mod;

  auto lib = utl::make_ref<smodule>(this);
  lib->getInfo().handle = handleCounter;
  handleCounter++;

  modules.emplace_back(lib);

  // HLE/system modules ship with the emulator; prefer those.
  base::String hostRel("modules/");
  hostRel.append(name.data(), name.length());
  hostRel += ".sprx";
  base::String hostPath = utl::make_abs_path(hostRel);
  const bool isRebirth = name == base::StringRef("rebirth");
  if (utl::File(hostPath, utl::fileMode::read).IsOpen()) {
    if (lib->fromFile(hostPath)) {
      if (isRebirth)
        bringUpRebirthSurfaceRegistry(*lib);
      return lib;
    }
  } else {
    // The game's own modules live inside the pkg: SDK prx under
    // /app0/sce_module, the title's own prx at the app root.
    const char *roots[] = {"/app0/sce_module/", "/app0/"};
    for (const char *root : roots) {
      base::String vfsPath(root);
      vfsPath.append(name.data(), name.length());
      vfsPath += ".prx";
      if (vfs::openRead(vfsPath.c_str()).Exists()) {
        if (!lib->fromVfs(vfsPath))
          return nullptr;
        if (isRebirth)
          bringUpRebirthSurfaceRegistry(*lib);
        return lib;
      }
    }
  }

  base::String sname;
  sname.append(name.data(), name.length());
  LOG_ERROR("unable to load module {}", sname.c_str());
  return nullptr;
}

// Patch a guest function to `xor eax,eax; ret`. Steps over libkernel-internal
// validation that rejects our externally-loaded module set (11.00 offsets).
static void forceReturn0(proc &p, const char *mod, uint32_t off) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  uint8_t *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0x31;  // xor eax, eax
  c[1] = 0xC0;
  c[2] = 0xC3;  // ret
}

// Boot patches applied before entering the guest, needed by every boot path
// (modexec and the real pkg boot), not just the modexec harness.
static void applyBootPatches(proc &p) {
  // Redirect libkernel's __tls_get_addr (NID vNe1w4diLCs) to our per-thread HLE;
  // libkernel's own dynamic-TLS allocator leaves DTV entries null. NID lookup is
  // firmware-independent.
  //
  // NATIVE ONLY: this patches the guest to `jmp` a *host* function pointer,
  // which only works when the host runs x86 directly. Under the FEXCore JIT
  // (aarch64) that address is ARM code and jumping to it as x86 faults wildly.
  // TODO(boot/fex): redirect __tls_get_addr via a FEXCore thunk trampoline, or
  // satisfy guest dynamic TLS another way.
#if defined(DELTA_BACKEND_NATIVE)
  if (auto k = p.getModule(base::StringRef("libkernel"))) {
    if (uintptr_t a = k->getSymbolByNid("vNe1w4diLCs")) {
      auto *c = reinterpret_cast<uint8_t *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0x48;  // movabs rax, imm64
      c[1] = 0xB8;
      *reinterpret_cast<uint64_t *>(c + 2) =
          reinterpret_cast<uint64_t>(&guest_tls_get_addr);
      c[10] = 0xFF;  // jmp rax
      c[11] = 0xE0;
      LOG_INFO("patched libkernel __tls_get_addr -> host HLE");
    }
  }
#else  // DELTA_BACKEND_FEX
  // FEX path: a host jump is invalid inside the x86 JIT, so patch the export to
  // a tiny `mov eax, <magic>; syscall; ret` stub that the FEX syscall handler
  // bridges to krnl::guest_tls_get_addr (tls_index ptr arrives in rdi).
  if (auto k = p.getModule(base::StringRef("libkernel"))) {
    if (uintptr_t a = k->getSymbolByNid("vNe1w4diLCs")) {
      auto *c = reinterpret_cast<uint8_t *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0xB8; // mov eax, imm32
      *reinterpret_cast<uint32_t *>(c + 1) = cpu::kTlsGetAddrSyscall;
      c[5] = 0x0F; // syscall
      c[6] = 0x05;
      c[7] = 0xC3; // ret
      LOG_INFO("patched libkernel __tls_get_addr -> magic syscall (FEX)");
    }
  }
#endif
  forceReturn0(p, "libkernel", 0x287e0);            // module-gen lib-id validator
  forceReturn0(p, "libSceAppContentUtil", 0x1a00);  // AppContent IPMI init
}

void proc::start() {
  LOG_ASSERT(modules[1]->getInfo().name == "libkernel");

  installCrashHandler();
  applyBootPatches(*this);

  auto &info = modules[0]->getInfo();
  auto &kinfo = modules[1]->getInfo();

  if (!info.entry) {
    LOG_WARNING("entry missing for {}", info.name.c_str());
    return;
  }

  union stack_entry {
    const void *ptr;
    uint64_t val;
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
  cpu::backend().enterGuest(reinterpret_cast<uintptr_t>(kinfo.entry), stack,
                            /*fsbase*/ 0);
}
}  // namespace krnl
