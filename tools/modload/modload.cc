// Loads one module (a decrypted SCE-dynamic ELF) through kern::Module and
// prints what came out. Usage: modload <module.sprx>
#include <cstdio>

#include "kern/object_ref.h"
#include "logger/logger.h"

#include "base/strings/xstring.h"
#include "kern/module.h"
#include "kern/process.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <module.(s)prx>\n", argv[0]);
    return 1;
  }

  logger::CreateLogger(true);

  kern::Process proc;  // ctor registers itself as the active process
  if (!proc.GetVma().Init()) {
    std::printf("[modload] vma init failed\n");
    return 1;
  }

  auto mod = kern::MakeRef<kern::Module>(&proc);
  mod->GetInfo().handle = 0;

  std::printf("[modload] loading %s ...\n", argv[1]);
  bool ok = mod->FromFile(base::String(argv[1]));
  std::printf("[modload] fromFile -> %s\n", ok ? "OK" : "FAIL");

  if (ok) {
    auto& info = mod->GetInfo();
    std::printf("  name:     %s\n", info.name.c_str());
    std::printf("  base:     %p\n", reinterpret_cast<void*>(info.base));
    std::printf("  entry:    %p\n", reinterpret_cast<void*>(info.entry));
    std::printf("  codeSize: %u bytes\n", info.code_size);
  }
  return ok ? 0 : 2;
}
