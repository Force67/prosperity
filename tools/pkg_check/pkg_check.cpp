// Dumps a fake-signed .pkg through the native PkgFilesystem so it can be
// cross-checked against tools/pkg_extract.py: lists every file and rebuilds
// eboot.bin into an ELF. Usage: pkg_check <game.pkg> [out.elf]
#include "base/arch.h"
#include <cstdio>

#include <logger/logger.h>
#include <io/file.h>

#include <formats/fself.h>

#include "formats/pkg_filesystem.h"
#include <base/algorithm.h>
#include <base/containers/vector.h>
#include <base/strings/xstring.h>

int main(int argc, char **argv) {
  logger::CreateLogger(true);
  if (argc < 2) {
    std::printf("usage: pkg_check <game.pkg> [out.elf]\n");
    return 1;
  }

  vfs::PkgFilesystem fs((base::String(argv[1])));
  if (!fs.Valid()) {
    std::printf("invalid / unsupported pkg\n");
    return 1;
  }

  base::Vector<base::String> paths;
  fs.Paths(paths);
  base::Sort(paths.begin(), paths.end());
  std::printf("%zu files\n", paths.size());
  for (const auto &p : paths)
    std::printf("%s\n", p.c_str());

  const auto *node = fs.Find("/eboot.bin");
  if (node) {
    base::Vector<u8> buf(node->size);
    fs.Read(*node, buf.data(), 0, static_cast<i64>(node->size));
    auto elf = crypto::Self2elf(buf.data(), buf.size());
    if (!elf.empty()) {
      const char *out = argc > 2 ? argv[2] : "eboot_native.elf";
      io::File f(base::String(out), io::FileMode::kWrite);
      f.Write(elf.data(), elf.size());
      std::printf("wrote %s (%zu bytes)\n", out, elf.size());
    } else {
      std::printf("eboot.bin is not a SELF?\n");
    }
  }
  return 0;
}
