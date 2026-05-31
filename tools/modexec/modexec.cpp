// Runs a module through load -> relocate -> start, one stage at a time, so each
// layer can be brought up on its own. Usage: modexec <main-module.sprx> [run]
#include <cstdio>
#include <cstring>

#include <logger/logger.h>
#include <utl/mem.h>

#include "kern/lv2/sys_dynlib.h"
#include "kern/module.h"
#include "kern/proc.h"
#include "kern/vfs.h"

#include <string>

// SCOUT: patch a guest function to `xor eax,eax; ret` (return 0). Used to step
// over libkernel-internal validation that rejects our externally-loaded module
// set, so we can see how much further the boot gets.
static void forceReturn0(krnl::proc& proc, const char* mod, uint32_t off) {
  auto m = proc.getModule(base::StringRef(mod));
  if (!m)
    return;
  uint8_t* p = m->getInfo().base + off;
  // mprotect needs a page-aligned base.
  auto page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) & ~0xFFFull);
  utl::protectMem(page, 0x1000, utl::pageProtection::rwx);
  p[0] = 0x31;  // xor eax, eax
  p[1] = 0xC0;
  p[2] = 0xC3;  // ret
  std::printf("[modexec] SCOUT patched %s+%#x -> return 0\n", mod, off);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <main-module.sprx>\n", argv[0]);
    return 1;
  }

  utl::createLogger(true);

  // Mount /app0 onto the directory the main module lives in, so the game's
  // runtime file opens resolve to the extracted disc image.
  {
    std::string p(argv[1]);
    auto slash = p.find_last_of('/');
    std::string dir = slash == std::string::npos ? "." : p.substr(0, slash);
    krnl::vfs::mount("/app0", dir.c_str());
    std::printf("[modexec] mounted /app0 -> %s\n", dir.c_str());
  }

  krnl::proc proc;

  // stage 1: load. create() preloads libkernel + libSceLibcInternal, then the
  // main module and its DT_NEEDED tree.
  std::printf("[modexec] === stage 1: load ===\n");
  if (!proc.create(base::String(argv[1]))) {
    std::printf("[modexec] proc::create FAILED\n");
    return 2;
  }
  auto& mods = proc.getModuleList();
  std::printf("[modexec] loaded %zu modules\n", mods.size());

  // stage 2: resolve imports + relocate every module (what guest libkernel
  // triggers via syscall 599 at startup).
  std::printf("[modexec] === stage 2: relocate ===\n");
  int rc = krnl::sys_dynlib_process_needed_and_relocate();
  std::printf("[modexec] relocate -> %d\n", rc);
  if (rc != 0)
    return 3;

  // A real eboot carries its own SCE process param (PT_SCE_PROCPARAM). Only when
  // it's missing (e.g. running a bare lib as main) do we fake a minimal one so
  // libkernel's _start clears its size + "ORBI" magic check.
  auto& m0 = mods[0]->getInfo();
  if (!m0.procParam) {
    static uint8_t procParam[0x50] = {};
    *reinterpret_cast<uint64_t*>(procParam + 0x00) = sizeof(procParam);
    *reinterpret_cast<uint32_t*>(procParam + 0x08) = 0x4942524F;  // "ORBI"
    *reinterpret_cast<uint32_t*>(procParam + 0x0C) = 1;           // entry count (!= 0)
    *reinterpret_cast<uint32_t*>(procParam + 0x10) = 0x11000000;  // sdk version
    m0.procParam = procParam;
    m0.procParamSize = sizeof(procParam);
    std::printf("[modexec] (using synthetic proc param)\n");
  } else {
    std::printf("[modexec] using module's own proc param (%u bytes)\n", m0.procParamSize);
  }

  // stage 3 (opt-in): jump into the guest. proc::start enters libkernel's entry
  // with modules[0] as the main program.
  if (argc > 2 && std::strcmp(argv[2], "run") == 0) {
    // SCOUT patches for libkernel-internal module bookkeeping (11.00 offsets).
    forceReturn0(proc, "libkernel", 0x287e0);  // module-gen lib-id validator
    std::printf("[modexec] === stage 3: execute (jumping to guest entry) ===\n");
    std::fflush(stdout);
    proc.start();
    std::printf("[modexec] returned from guest entry\n");
  }

  return 0;
}
