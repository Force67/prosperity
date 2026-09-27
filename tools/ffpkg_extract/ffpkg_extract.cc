// Lists / extracts files from a PS5 *.ffpkg (UFS2 image) through the native
// Ufs2Filesystem reader.
//   ffpkg_extract <game.ffpkg>                     list every file
//   ffpkg_extract <game.ffpkg> <relpath> <out>     extract one file
#include <cstdio>
#include "base/arch.h"

#include "io/file.h"
#include "logger/logger.h"

#include "base/algorithm.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "formats/ufs2_filesystem.h"

int main(int argc, char** argv) {
  logger::CreateLogger(true);
  if (argc < 2) {
    std::printf("usage: ffpkg_extract <game.ffpkg> [<relpath> <out>]\n");
    return 1;
  }

  formats::Ufs2Filesystem fs((base::String(argv[1])));
  if (!fs.Valid()) {
    std::printf("not a valid UFS2 image (bad superblock)\n");
    return 1;
  }

  if (argc >= 4) {
    const auto* node = fs.Find(argv[2]);
    if (!node) {
      std::printf("%s: not found\n", argv[2]);
      return 1;
    }
    base::Vector<u8> buf(node->size);
    i64 n = fs.Read(*node, buf.data(), 0, static_cast<i64>(node->size));
    if (n < 0) {
      std::printf("read failed\n");
      return 1;
    }
    io::File out(base::String(argv[3]), io::FileMode::kWrite);
    out.Write(buf.data(), static_cast<size_t>(n));
    std::printf("wrote %s (%lld bytes)\n", argv[3], (long long)n);
    return 0;
  }

  base::Vector<base::String> paths;
  fs.Paths(paths);
  base::Sort(paths.begin(), paths.end());
  u64 total = 0;
  for (const auto& p : paths) {
    const auto* node = fs.Find(p.c_str());
    u64 sz = node ? node->size : 0;
    total += sz;
    std::printf("%12llu  %s\n", (unsigned long long)sz, p.c_str());
  }
  std::printf("%zu files, %llu bytes total\n", paths.size(),
              (unsigned long long)total);
  return 0;
}
