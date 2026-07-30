/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PlayGo daemon. libScePlayGo is a thin IPMI client: scePlayGoOpen invokes
 * open / get-chunk-count / get-loci and asserts FATAL when the count is 0. We
 * answer as a fully installed title, which is what a whole-pkg mount is.
 */

#include <cstdint>
#include <cstdlib>

#include <base.h>

#include "kern/vfs.h"
#include "services.h"

namespace krnl::ipmi {
namespace {

enum {
  kOpen = 0x30000,
  kGetLoci = 0x30008,
  kGetProgress = 0x3000d,
  kGetChunkCount = 0x3000f,
};

// Per-chunk availability.
enum { kLocusNotDownloaded = 0, kLocusLocalSlow = 2, kLocusLocalFast = 3 };

// The real per-title count lives in /app0/sce_sys/playgo-chunk.dat (magic
// "pgd\0", chunk_count is a u16 at 0x0A). A wrong count breaks multi-chunk
// titles: Shadow of the Tomb Raider enumerates chunk loci 0..N during boot and,
// when the count is too small, an unsigned `count - 0x50` underflows into a
// ~4-billion-iteration loop that smashes the stack. Default to 0x50, the value
// such titles treat as "the standard set, fully installed"; the pkgs we run
// mostly ship no playgo-chunk.dat. DELTA_PLAYGO_CHUNKS overrides.
uint32_t chunkCount() {
  static uint32_t cached = 0;
  if (cached)
    return cached;
  if (const char *ov = std::getenv("DELTA_PLAYGO_CHUNKS")) {
    int v = std::atoi(ov);
    cached = v > 0 ? static_cast<uint32_t>(v) : 1;
    return cached;
  }
  cached = 0x50;
  utl::File f = vfs::openRead("/app0/sce_sys/playgo-chunk.dat");
  if (f.Exists()) {
    uint8_t hdr[0x10] = {};
    if (f.Read(hdr, sizeof(hdr)) == sizeof(hdr) && hdr[0] == 'p' &&
        hdr[1] == 'g' && hdr[2] == 'd') {
      uint32_t cc = static_cast<uint32_t>(hdr[0x0a] | (hdr[0x0b] << 8));
      if (cc > 0)
        cached = cc;
    }
  }
  return cached;
}

struct PlayGo : Service {
  const char *name() const override { return "ScePlayGo"; }

  void invoke(Invocation &inv) override {
    switch (inv.method()) {
    case kOpen: // server-side handle; must be neither 0 nor -1
      inv.replyU32(0, 1);
      break;
    case kGetChunkCount: // 0 makes scePlayGoOpen fatal
      inv.replyU32(0, chunkCount());
      break;
    case kGetLoci: // byte array indexed by chunk id
      inv.replyFill(0, kLocusLocalFast);
      break;
    case kGetProgress: { // { uint64 progressSize; uint64 totalSize }
      const uint64_t done[2] = {1, 1}; // == 100%
      inv.reply(0, done, sizeof(done));
      break;
    }
    default:
      // Remaining getters (todo list, eta, install speed, language) and every
      // setter: a zeroed reply already reads as "installed, nothing pending".
      inv.replyEmpty();
      break;
    }
  }
};

PlayGo g_playGo;

} // namespace

Service &playGoService() { return g_playGo; }

} // namespace krnl::ipmi
