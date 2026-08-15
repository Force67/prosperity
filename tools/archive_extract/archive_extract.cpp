// Lists / extracts files from a game left inside its distribution container
// (.rar, .zip) through the native ArchiveFilesystem reader. Paths are the ones
// the guest sees, i.e. after the wrapper directory is stripped, so
// "/eboot.bin" and not "PPSA01342-app/eboot.bin".
//   archive_extract <game.rar>                     list every file
//   archive_extract <game.rar> <relpath> <out>     extract one file
//   archive_extract <game.rar> --bench <relpath>   time a streaming read
#include "base/arch.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <logger/logger.h>
#include <utl/file.h>

#include "formats/archive_object.h"

int main(int argc, char **argv) {
  utl::createLogger(true);
  if (argc < 2) {
    std::printf("usage: archive_extract <game.rar> [<relpath> <out>]\n"
                "       archive_extract <game.rar> --bench <relpath>\n");
    return 1;
  }

  const auto t0 = std::chrono::steady_clock::now();
  vfs::ArchiveFilesystem fs((base::String(argv[1])));
  const auto t1 = std::chrono::steady_clock::now();
  if (!fs.valid()) {
    std::printf("not a container we can read\n");
    return 1;
  }
  std::printf("opened via %s backend in %lld ms\n", fs.backendName(),
              (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                  t1 - t0)
                  .count());

  if (argc >= 4 && std::strcmp(argv[2], "--bench") == 0) {
    const auto *node = fs.find(argv[3]);
    if (!node) {
      std::printf("%s: not found\n", argv[3]);
      return 1;
    }
    // Read it the way a guest streams an asset: forward, one chunk at a time.
    // If the backend restarts its decoder per call this never finishes.
    constexpr i64 kChunk = 1 << 20;
    std::vector<u8> buf(kChunk);
    const auto start = std::chrono::steady_clock::now();
    i64 off = 0, total = 0;
    while (off < static_cast<i64>(node->size)) {
      const i64 n = fs.read(*node, buf.data(), off, kChunk);
      if (n <= 0)
        break;
      off += n;
      total += n;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf("streamed %lld / %llu bytes in %.2f s (%.1f MB/s)\n",
                (long long)total, (unsigned long long)node->size, secs,
                secs > 0 ? total / secs / (1 << 20) : 0.0);
    return total == static_cast<i64>(node->size) ? 0 : 1;
  }

  if (argc >= 4) {
    const auto *node = fs.find(argv[2]);
    if (!node) {
      std::printf("%s: not found\n", argv[2]);
      return 1;
    }
    std::vector<u8> buf(node->size);
    const i64 n = fs.read(*node, buf.data(), 0, static_cast<i64>(node->size));
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
  u64 total = 0;
  for (const auto &p : paths) {
    const auto *node = fs.find(p.c_str());
    const u64 sz = node ? node->size : 0;
    total += sz;
    std::printf("%12llu  %s\n", (unsigned long long)sz, p.c_str());
  }
  std::printf("%zu files, %llu bytes total\n", paths.size(),
              (unsigned long long)total);
  return 0;
}
