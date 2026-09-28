/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PlayGo daemon. libScePlayGo is a thin IPMI client on both platforms and
 * speaks the same 0x300xx method family: scePlayGoOpen invokes open /
 * get-chunk-ids / get-loci and fails when the count is 0. We answer as a fully
 * installed title, which is what a whole-pkg mount is.
 */

#include <cstdlib>
#include <cstring>
#include "base/arch.h"

#include "base/containers/vector.h"
#include "kern/ipmi/services.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
DELTA_OPTION(u32, kPlaygoChunks, "DELTA_PLAYGO_CHUNKS", 0);
}  // namespace

namespace kern::ipmi {
namespace {

enum {
  kOpen = 0x30000,
  kGetLoci = 0x30008,
  kGetProgress = 0x3000d,
  kGetChunkIds = 0x3000e,
  kGetChunkCount = 0x3000f,
};

// The PS5 client stores at most 1000 chunks (u16 ids at +0x24, loci at +0x7f4).
constexpr u32 kMaxChunks = 0x3e8;

// Per-chunk availability.
enum { kLocusNotDownloaded = 0, kLocusLocalSlow = 2, kLocusLocalFast = 3 };

bool ReadAll(const char* path, base::Vector<char>& out) {
  io::File f = vfs::OpenRead(path);
  if (!f.Exists())
    return false;
  const u64 size = f.GetSize();
  if (!size || size > 0x100000)
    return false;
  out.resize(size + 1);  // NUL so the scanners can strstr
  out[size] = '\0';
  return f.Read(out.data(), size) == size;
}

void ScanIds(const char* p, const char* end, u32& max_id, bool& any) {
  while (p < end) {
    if (*p >= '0' && *p <= '9') {
      u32 v = 0;
      while (p < end && *p >= '0' && *p <= '9')
        v = v * 10 + static_cast<u32>(*p++ - '0');
      if (v < kMaxChunks) {
        if (v > max_id)
          max_id = v;
        any = true;
      }
    } else {
      p++;
    }
  }
}

// PS5 pkgs ship no playgo-chunk.dat; the chunk id space lives in the scenario
// JSON
// ("chunks": ["0-15", "17"]) or, when omitted (Demon's Souls), a chunkdefs XML
// beside it. Ids 0..max: a dense superset of a sparse set is still "installed".
u32 Ps5ChunkCount() {
  u32 max_id = 0;
  bool any = false;
  base::Vector<char> text;
  if (ReadAll("/app0/sce_sys/playgo-scenario.json", text)) {
    const char* end = text.data() + text.size() - 1;
    for (const char* p = text.data(); (p = std::strstr(p, "\"chunks\""));) {
      p += 8;
      while (p < end && (*p == ':' || *p == ' ' || *p == '\t'))
        p++;
      const char* stop = p;
      if (p < end && *p == '[') {
        while (stop < end && *stop != ']')
          stop++;
      } else if (p < end && *p == '"') {
        stop++;
        while (stop < end && *stop != '"')
          stop++;
      } else {
        while (stop < end && *stop >= '0' && *stop <= '9')
          stop++;
      }
      ScanIds(p, stop, max_id, any);
      p = stop > p ? stop : p + 1;
    }
  }
  if (!any && ReadAll("/app0/playgo-chunkdefs.xml", text)) {
    const char* end = text.data() + text.size() - 1;
    for (const char* p = text.data(); (p = std::strstr(p, "<chunk "));) {
      const char* stop = p;
      while (stop < end && *stop != '>')
        stop++;
      if (const char* id = std::strstr(p, " id=\""); id && id < stop) {
        id += 5;
        const char* id_end = id;
        while (id_end < stop && *id_end >= '0' && *id_end <= '9')
          id_end++;
        ScanIds(id, id_end, max_id, any);
      }
      p = stop;
    }
  }
  return any ? max_id + 1 : 0;
}

// The real count lives in playgo-chunk.dat (magic "pgd\0", u16 at 0x0A). A
// wrong count breaks multi-chunk titles: SOTTR enumerates loci 0..N and an
// unsigned `count - 0x50` underflows into a ~4-billion-iteration stack-smashing
// loop when the count is too small. Default 0x50 ("standard set, fully
// installed"); DELTA_PLAYGO_CHUNKS overrides.
u32 ChunkCount() {
  static u32 cached = 0;
  if (cached)
    return cached;
  if (kPlaygoChunks > 0) {
    cached = kPlaygoChunks;
    return cached;
  }
  cached = 0x50;
  io::File f = vfs::OpenRead("/app0/sce_sys/playgo-chunk.dat");
  u8 hdr[0x10] = {};
  if (f.Exists() && f.Read(hdr, sizeof(hdr)) == sizeof(hdr) && hdr[0] == 'p' &&
      hdr[1] == 'g' && hdr[2] == 'd') {
    u32 cc = static_cast<u32>(hdr[0x0a] | (hdr[0x0b] << 8));
    if (cc > 0)
      cached = cc;
  } else if (u32 cc = Ps5ChunkCount()) {
    cached = cc;
  }
  if (cached > kMaxChunks)
    cached = kMaxChunks;
  return cached;
}

struct PlayGo : Service {
  const char* Name() const override { return "ScePlayGo"; }

  void Invoke(Invocation& inv) override {
    switch (inv.Method()) {
      case kOpen:  // server-side handle; must be neither 0 nor -1
        inv.ReplyU32(0, 1);
        break;
      case kGetChunkCount:  // 0 makes scePlayGoOpen fatal
        inv.ReplyU32(0, ChunkCount());
        break;
      case kGetLoci:  // byte array indexed by the requested chunk-id list
        inv.ReplyFill(0, kLocusLocalFast);
        break;
      case kGetChunkIds: {  // in {u32 handle, u32 max}; out u16 ids[], u32
                            // count.
        // scePlayGoOpen (PS5) fails 0x80b20001 on count 0 and scePlayGoGetLocus
        // rejects any requested id missing from this list, so it must cover
        // every chunk the title's own data names.
        u64 sz = 0;
        auto* in = static_cast<const u32*>(inv.Input(0, sz));
        u32 n = ChunkCount();
        if (in && sz >= 8 && in[1] < n)
          n = in[1];
        u16 ids[kMaxChunks];
        for (u32 i = 0; i < n; i++)
          ids[i] = static_cast<u16>(i);
        inv.Reply(0, ids, n * sizeof(u16));
        inv.ReplyU32(1, n);
        break;
      }
      case kGetProgress: {  // { uint64 progressSize; uint64 totalSize }
        const u64 done[2] = {1, 1};  // == 100%
        inv.Reply(0, done, sizeof(done));
        break;
      }
      default:
        // Remaining getters (todo list, eta, install speed, language) and every
        // setter: a zeroed reply already reads as "installed, nothing pending".
        inv.ReplyEmpty();
        break;
    }
  }
};

PlayGo g_play_go;

}  // namespace

Service& PlayGoService() {
  return g_play_go;
}

u32 PlayGoChunkCount() {
  return ChunkCount();
}

}  // namespace kern::ipmi
