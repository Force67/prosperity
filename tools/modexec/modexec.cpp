// Runs a module through load -> relocate -> start, one stage at a time, so each
// layer can be brought up on its own. Usage: modexec <main-module.sprx> [run]
#include <cstdio>
#include <cstring>

#include <logger/logger.h>

#include "kern/lv2/sys_dynlib.h"
#include "kern/module.h"
#include "kern/proc.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <main-module.sprx>\n", argv[0]);
    return 1;
  }

  utl::createLogger(true);

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

  // no eboot, so fake a minimal SCE process param. libkernel's _start fetches
  // it via sys_dynlib_get_proc_param and checks the size + "ORBI" magic.
  static uint8_t procParam[0x50] = {};
  *reinterpret_cast<uint64_t*>(procParam + 0x00) = sizeof(procParam);
  *reinterpret_cast<uint32_t*>(procParam + 0x08) = 0x4942524F;  // "ORBI"
  *reinterpret_cast<uint32_t*>(procParam + 0x0C) = 1;           // entry count (!= 0)
  *reinterpret_cast<uint32_t*>(procParam + 0x10) = 0x11000000;  // sdk version
  {
    auto& m0 = mods[0]->getInfo();
    m0.procParam = procParam;
    m0.procParamSize = sizeof(procParam);
  }

  // stage 3 (opt-in): jump into the guest. proc::start enters libkernel's entry
  // with modules[0] as the main program.
  if (argc > 2 && std::strcmp(argv[2], "run") == 0) {
    std::printf("[modexec] === stage 3: execute (jumping to guest entry) ===\n");
    std::fflush(stdout);
    proc.start();
    std::printf("[modexec] returned from guest entry\n");
  }

  return 0;
}
