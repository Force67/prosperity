// Unpacks a decrypted PS4/PS5 firmware update (*.PUP / *.PUP.dec) through the
// native PupReader. Usage: pup_extract <fw.PUP.dec> <out_dir>
#include <cstdio>

#include <logger/logger.h>
#include <io/file.h>

#include "formats/pup_reader.h"
#include <base/strings/xstring.h>

int main(int argc, char **argv) {
  logger::CreateLogger(true);
  if (argc < 3) {
    std::printf("usage: pup_extract <firmware.PUP[.dec]> <out_dir>\n");
    return 1;
  }

  formats::PupReader r((base::String(argv[1])));
  if (!r.Load()) {
    std::printf("not a recognized PUP container (encrypted or bad magic)\n");
    return 1;
  }

  std::printf("%s PUP, %d segment(s)\n", r.ps5() ? "PS5" : "PS4",
              r.SegmentCount());
  bool encrypted = false;
  base::String summary = r.ExtractAll(base::String(argv[2]), encrypted);
  std::fputs(summary.c_str(), stdout);
  return 0;
}
