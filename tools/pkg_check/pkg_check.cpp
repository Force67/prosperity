// Dumps a fake-signed .pkg through the native PkgFilesystem so it can be
// cross-checked against tools/pkg_extract.py: lists every file and rebuilds
// eboot.bin into an ELF. Usage: pkg_check <game.pkg> [out.elf]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <logger/logger.h>
#include <utl/file.h>

#include <crypto/UnSELF.h>

#include "formats/pkg_object.h"

int main(int argc, char **argv) {
  utl::createLogger(true);
  if (argc < 2) {
    std::printf("usage: pkg_check <game.pkg> [out.elf]\n");
    return 1;
  }

  vfs::PkgFilesystem fs((base::String(argv[1])));
  if (!fs.valid()) {
    std::printf("invalid / unsupported pkg\n");
    return 1;
  }

  std::vector<std::string> paths;
  fs.paths(paths);
  std::sort(paths.begin(), paths.end());
  std::printf("%zu files\n", paths.size());
  for (const auto &p : paths)
    std::printf("%s\n", p.c_str());

  const auto *node = fs.find("/eboot.bin");
  if (node) {
    std::vector<uint8_t> buf(node->size);
    fs.read(*node, buf.data(), 0, static_cast<int64_t>(node->size));
    auto elf = crypto::self2elf(buf.data(), buf.size());
    if (!elf.empty()) {
      const char *out = argc > 2 ? argv[2] : "eboot_native.elf";
      utl::File f(base::String(out), utl::fileMode::write);
      f.Write(elf.data(), elf.size());
      std::printf("wrote %s (%zu bytes)\n", out, elf.size());
    } else {
      std::printf("eboot.bin is not a SELF?\n");
    }
  }
  return 0;
}
