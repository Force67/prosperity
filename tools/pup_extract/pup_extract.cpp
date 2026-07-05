// Unpacks a decrypted PS4/PS5 firmware update (*.PUP / *.PUP.dec) through the
// native pupReader. Usage: pup_extract <fw.PUP.dec> <out_dir>
#include <cstdio>

#include <logger/logger.h>
#include <utl/file.h>

#include "formats/pup_object.h"

int main(int argc, char **argv) {
  utl::createLogger(true);
  if (argc < 3) {
    std::printf("usage: pup_extract <firmware.PUP[.dec]> <out_dir>\n");
    return 1;
  }

  vfs::pupReader r((base::String(argv[1])));
  if (!r.load()) {
    std::printf("not a recognized PUP container (encrypted or bad magic)\n");
    return 1;
  }

  std::printf("%s PUP, %d segment(s)\n", r.ps5() ? "PS5" : "PS4",
              r.segmentCount());
  bool encrypted = false;
  base::String summary = r.extractAll(base::String(argv[2]), encrypted);
  std::fputs(summary.c_str(), stdout);
  return 0;
}
