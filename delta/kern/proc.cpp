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
#include <sys/mman.h>

#include "crash.h"
#include "module.h"
#include "proc.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include "lv2/sys_dynlib.h"
#include "runtime/vprx/vprx.h"

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
  if (utl::File(hostPath, utl::fileMode::read).IsOpen()) {
    if (lib->fromFile(hostPath))
      return lib;
  } else {
    // The game's own modules live inside the pkg: SDK prx under
    // /app0/sce_module, the title's own prx at the app root.
    const char *roots[] = {"/app0/sce_module/", "/app0/"};
    for (const char *root : roots) {
      base::String vfsPath(root);
      vfsPath.append(name.data(), name.length());
      vfsPath += ".prx";
      if (vfs::openRead(vfsPath.c_str()).Exists())
        return lib->fromVfs(vfsPath) ? lib : nullptr;
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
  forceReturn0(p, "libkernel", 0x93dc);  // MutexattrInitForInternalLibc idempotent
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

  // Run libSceLibcInternal's malloc bootstrap (libc+0x29610) once, BEFORE the
  // eboot CRT, via a trampoline planted as AT_ENTRY: _start calls AT_ENTRY, the
  // trampoline runs the bootstrap then jmps the real eboot entry. Fixes the
  // savedata null-mspace crash (system libc arena otherwise never created).
  const void *guestEntry = (const void *)(info.entry);
  if (auto libc = getModule(base::StringRef("libSceLibcInternal"))) {
    uintptr_t boot = reinterpret_cast<uintptr_t>(libc->getInfo().base) + 0x29610;
    uintptr_t real = reinterpret_cast<uintptr_t>(info.entry);
    auto *t = static_cast<uint8_t *>(mmap(nullptr, 0x1000,
        PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    size_t n = 0;
    t[n++] = 0x57;                                              // push rdi
    t[n++] = 0x48; t[n++] = 0xB8;                               // movabs rax, boot
    *reinterpret_cast<uint64_t *>(t + n) = boot; n += 8;
    t[n++] = 0xFF; t[n++] = 0xD0;                               // call rax
    t[n++] = 0x5F;                                              // pop rdi
    t[n++] = 0x48; t[n++] = 0xB8;                               // movabs rax, real
    *reinterpret_cast<uint64_t *>(t + n) = real; n += 8;
    t[n++] = 0xFF; t[n++] = 0xE0;                               // jmp rax
    cpu::backend().registerExecRange(reinterpret_cast<uintptr_t>(t), 0x1000);
    guestEntry = t;
    LOG_INFO("boot: libc-bootstrap trampoline {:#x} -> entry {:#x}",
             (uintptr_t)t, real);
  }

  stack[0].val = 1 + 0; // argc
  auto s = reinterpret_cast<stack_entry *>(&stack[1]);
  (*s++).ptr = info.name.c_str();
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;
  (*s++).val = 9ull;
  (*s++).ptr = guestEntry;
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;

  // Enter libkernel's entry with the PS4 convention (arg block in rdi). The
  // backend runs it natively (x86 host) or via the FEXCore JIT (aarch64 host).
  cpu::backend().enterGuest(reinterpret_cast<uintptr_t>(kinfo.entry), stack,
                            /*fsbase*/ 0);
}
}  // namespace krnl
