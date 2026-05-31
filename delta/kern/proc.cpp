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
#include <xbyak.h>

#include "crash.h"
#include "module.h"
#include "proc.h"
#include "vfs.h"
#include "lv2/sys_dynlib.h"
#include "runtime/vprx/vprx.h"

namespace krnl {
static proc *g_activeProc{nullptr};

// Per-thread guest fs base (TLS). The lifter rewrites guest `fs:[disp]` reads to
// call krnl_current_fsbase() so each host thread running guest code reads its
// own thread's TLS. initial-exec model => the access is a single
// `mov rax, fs:[off]; ret` that clobbers only rax, which the lifter stub relies
// on (it preserves only rax around the call).
__attribute__((tls_model("initial-exec"))) static thread_local uint64_t t_fsbase =
    0;
extern "C" uint64_t krnl_current_fsbase() { return t_fsbase; }
void setThreadFsBase(uint64_t v) { t_fsbase = v; }

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

  auto func = (void *(PS4ABI *)(void *))kinfo.entry;

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
  func(stack);
}
}  // namespace krnl
