// Lists / extracts files from a PS5 *.ffpkg (UFS2 image) through the native
// Ufs2Filesystem reader.
//   ffpkg_extract <game.ffpkg>                     list every file
//   ffpkg_extract <game.ffpkg> <relpath> <out>     extract one file
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <logger/logger.h>
#include <utl/file.h>

#include "formats/ufs2_object.h"

int main(int argc, char **argv) {
  utl::createLogger(true);
  if (argc < 2) {
    std::printf("usage: ffpkg_extract <game.ffpkg> [<relpath> <out>]\n");
    return 1;
  }

  vfs::Ufs2Filesystem fs((base::String(argv[1])));
  if (!fs.valid()) {
    std::printf("not a valid UFS2 image (bad superblock)\n");
    return 1;
  }

  if (argc >= 4) {
    const auto *node = fs.find(argv[2]);
    if (!node) {
      std::printf("%s: not found\n", argv[2]);
      return 1;
    }
    std::vector<uint8_t> buf(node->size);
    int64_t n = fs.read(*node, buf.data(), 0, static_cast<int64_t>(node->size));
    if (n < 0) {
      std::printf("read failed\n");
      return 1;
    }
    utl::File out(base::String(argv[3]), utl::fileMode::write);
    out.Write(buf.data(), static_cast<size_t>(n));
    std::printf("wrote %s (%lld bytes)\n", argv[3], (long long)n);
    return 0;
  }

  std::vector<std::string> paths;
  fs.paths(paths);
  std::sort(paths.begin(), paths.end());
  uint64_t total = 0;
  for (const auto &p : paths) {
    const auto *node = fs.find(p.c_str());
    uint64_t sz = node ? node->size : 0;
    total += sz;
    std::printf("%12llu  %s\n", (unsigned long long)sz, p.c_str());
  }
  std::printf("%zu files, %llu bytes total\n", paths.size(),
              (unsigned long long)total);
  return 0;
}
