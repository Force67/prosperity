// Loads one module (a decrypted SCE-dynamic ELF) through krnl::smodule and
// prints what came out. Usage: modload <module.sprx>
#include <cstdio>

#include <logger/logger.h>
#include <utl/object_ref.h>

#include "kern/module.h"
#include "kern/proc.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <module.(s)prx>\n", argv[0]);
    return 1;
  }

  utl::createLogger(true);

  krnl::proc proc;  // ctor registers itself as the active process
  if (!proc.getVma().init()) {
    std::printf("[modload] vma init failed\n");
    return 1;
  }

  auto mod = utl::make_ref<krnl::smodule>(&proc);
  mod->getInfo().handle = 0;

  std::printf("[modload] loading %s ...\n", argv[1]);
  bool ok = mod->fromFile(base::String(argv[1]));
  std::printf("[modload] fromFile -> %s\n", ok ? "OK" : "FAIL");

  if (ok) {
    auto& info = mod->getInfo();
    std::printf("  name:     %s\n", info.name.c_str());
    std::printf("  base:     %p\n", reinterpret_cast<void*>(info.base));
    std::printf("  entry:    %p\n", reinterpret_cast<void*>(info.entry));
    std::printf("  codeSize: %u bytes\n", info.codeSize);
  }
  return ok ? 0 : 2;
}
