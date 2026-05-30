// Staged execution harness. Drives the post-load pipeline one stage at a time
// so each layer (relocate -> tls -> entry) can be brought up and debugged in
// isolation. Usage: modexec <main-module.sprx>
#include <cstdio>

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

  // Load the module graph the way the kernel does: preload libkernel
  // + libSceLibcInternal, then the main module and its DT_NEEDED tree.
  std::printf("[modexec] === stage 1: load ===\n");
  if (!proc.create(base::String(argv[1]))) {
    std::printf("[modexec] proc::create FAILED\n");
    return 2;
  }
  auto& mods = proc.getModuleList();
  std::printf("[modexec] loaded %zu modules\n", mods.size());

  // Resolve imports and apply relocations across every module, the same work
  // guest libkernel triggers via syscall 599 at startup.
  std::printf("[modexec] === stage 2: relocate ===\n");
  int rc = krnl::sys_dynlib_process_needed_and_relocate();
  std::printf("[modexec] relocate -> %d\n", rc);

  return rc == 0 ? 0 : 3;
}
