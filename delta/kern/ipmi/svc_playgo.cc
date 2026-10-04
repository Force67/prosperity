/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PlayGo daemon. libScePlayGo is a thin IPMI client on both platforms and
 * speaks the same 0x300xx method family: scePlayGoOpen invokes open /
 * get-chunk-ids / get-loci and fails when the count is 0. We answer as a fully
 * installed title, which is what a whole-pkg mount is.
 */

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include "base/arch.h"

#include "base/containers/vector.h"
#include "guest/session.h"
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
  kClose = 0x30001,
  kSetSpeed = 0x30002,
  kGetSpeed = 0x30003,
  kSetLanguage = 0x30004,
  kGetLanguage = 0x30005,
  kGetLoci = 0x30008,
  kSetTodo = 0x30009,
  kGetTodo = 0x3000a,
  kPrefetch = 0x3000b,
  kGetEta = 0x3000c,
  kGetProgress = 0x3000d,
  kGetChunkIds = 0x3000e,
  kGetChunkCount = 0x3000f,
};

// The PS5 client stores at most 1000 chunks (u16 ids at +0x24, loci at +0x7f4).
constexpr u32 kMaxChunks = 0x3e8;

// Per-chunk availability.
constexpr u8 kLocusLocalFast = 3;
constexpr i32 kInvalidArgument = static_cast<i32>(0x80b20004u);
constexpr i32 kBadHandle = static_cast<i32>(0x80b20009u);
constexpr i32 kBadChunkId = static_cast<i32>(0x80b2000cu);
constexpr i32 kBadSpeed = static_cast<i32>(0x80b2000du);

std::mutex g_mutex;
std::unordered_set<u32> g_handles;
u32 g_next_handle = 1;
u32 g_chunk_count = 0;
u32 g_speed = 2;
u64 g_language_mask = ~u64(0);

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
        v = std::min(kMaxChunks, v * 10 + static_cast<u32>(*p++ - '0'));
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
  u32& cached = g_chunk_count;
  if (cached)
    return cached;
  if (kPlaygoChunks > 0) {
    cached = std::min(static_cast<u32>(kPlaygoChunks), kMaxChunks);
    return cached;
  }
  cached = 0x50;
  io::File f = vfs::OpenRead("/app0/sce_sys/playgo-chunk.dat");
  u8 hdr[0x10] = {};
  if (f.Exists() && f.Read(hdr, sizeof(hdr)) == sizeof(hdr) && hdr[0] == 'p' &&
      hdr[1] == 'g' && hdr[2] == 'd' && hdr[3] == 0) {
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

bool ReadInput(Invocation& inv, u32 slot, void* dest, u64 bytes) {
  u64 size = 0;
  const void* input = inv.Input(slot, size);
  if (!input || size < bytes) {
    inv.SetResult(kInvalidArgument);
    return false;
  }
  std::memcpy(dest, input, bytes);
  return true;
}

bool Output(Invocation& inv, u32 slot, u64 bytes) {
  if (inv.CanReply(slot, bytes))
    return true;
  inv.SetResult(kInvalidArgument);
  return false;
}

bool ValidateChunks(Invocation& inv, u32 count, bool todo = false) {
  if (count > kMaxChunks) {
    inv.SetResult(kInvalidArgument);
    return false;
  }
  if (!count)
    return true;
  u64 size = 0;
  const auto* ids = static_cast<const u8*>(inv.Input(1, size));
  const u32 stride = todo ? 4 : 2;
  if (!ids || size < count * stride) {
    inv.SetResult(kInvalidArgument);
    return false;
  }
  for (u32 i = 0; i < count; ++i) {
    u16 id;
    std::memcpy(&id, ids + i * stride, sizeof(id));
    if (id >= ChunkCount()) {
      inv.SetResult(kBadChunkId);
      return false;
    }
    if (todo && ids[i * stride + 2] > kLocusLocalFast) {
      inv.SetResult(kInvalidArgument);
      return false;
    }
  }
  return true;
}

struct PlayGo : Service {
  const char* Name() const override { return "ScePlayGo"; }

  void Invoke(Invocation& inv) override {
    std::lock_guard lock(g_mutex);
    const u32 method = inv.Method();
    if (method == kOpen) {
      if (!Output(inv, 0, sizeof(u32)))
        return;
      const u32 handle = g_next_handle++;
      g_handles.insert(handle);
      inv.ReplyU32(0, handle);
      return;
    }
    u32 handle;
    if (!ReadInput(inv, 0, &handle, sizeof(handle)))
      return;
    if (!g_handles.contains(handle)) {
      inv.SetResult(kBadHandle);
      return;
    }
    struct Request {
      u32 handle, count;
    } request{};
    switch (method) {
      case kClose:
        g_handles.erase(handle);
        break;
      case kSetSpeed:
        if (!ReadInput(inv, 0, &request, sizeof(request)))
          return;
        if (request.count > 2) {
          inv.SetResult(kBadSpeed);
          return;
        }
        g_speed = request.count;
        break;
      case kGetSpeed:
        if (Output(inv, 0, sizeof(g_speed)))
          inv.ReplyU32(0, g_speed);
        break;
      case kSetLanguage: {
        struct Language {
          u32 handle, pad;
          u64 mask;
        } language;
        if (ReadInput(inv, 0, &language, sizeof(language)))
          g_language_mask = language.mask;
        break;
      }
      case kGetLanguage:
        if (Output(inv, 0, sizeof(g_language_mask)))
          inv.Reply(0, &g_language_mask, sizeof(g_language_mask));
        break;
      case kGetChunkCount:
        if (Output(inv, 0, sizeof(u32)))
          inv.ReplyU32(0, ChunkCount());
        break;
      case kGetChunkIds: {
        if (!ReadInput(inv, 0, &request, sizeof(request)))
          return;
        const u32 count = std::min(ChunkCount(), request.count);
        if (!Output(inv, 0, count * sizeof(u16)) ||
            !Output(inv, 1, sizeof(u32)))
          return;
        std::array<u16, kMaxChunks> ids;
        for (u32 i = 0; i < count; ++i)
          ids[i] = static_cast<u16>(i);
        inv.Reply(0, ids.data(), count * sizeof(u16));
        inv.ReplyU32(1, count);
        break;
      }
      case kGetTodo:
        if (!ReadInput(inv, 0, &request, sizeof(request)) ||
            !Output(inv, 0, 0) || !Output(inv, 1, sizeof(u32)))
          return;
        inv.ReplyFill(0, 0);
        inv.ReplyU32(1, 0);
        break;
      case kGetLoci:
      case kGetProgress:
      case kGetEta:
      case kSetTodo:
      case kPrefetch: {
        if (!ReadInput(inv, 0, &request, sizeof(request)) ||
            !ValidateChunks(inv, request.count, method == kSetTodo))
          return;
        if (method == kPrefetch) {
          struct Prefetch {
            u32 handle, count;
            u8 locus, pad[3];
          } prefetch;
          if (!ReadInput(inv, 0, &prefetch, sizeof(prefetch)))
            return;
          if (prefetch.locus > kLocusLocalFast)
            inv.SetResult(kInvalidArgument);
        } else if (method == kGetLoci) {
          if (!Output(inv, 0, request.count))
            return;
          std::array<u8, kMaxChunks> loci;
          loci.fill(kLocusLocalFast);
          inv.Reply(0, loci.data(), request.count);
        } else if (method == kGetProgress) {
          const u64 done[2] = {1, 1};
          if (Output(inv, 0, sizeof(done)))
            inv.Reply(0, done, sizeof(done));
        } else if (method == kGetEta) {
          const i64 eta = 0;
          if (Output(inv, 0, sizeof(eta)))
            inv.Reply(0, &eta, sizeof(eta));
        }
        break;
      }
      default:
        inv.ReplyEmpty();
        break;
    }
  }
};

PlayGo g_play_go;
const guest::SessionReset g_session_reset([] {
  std::lock_guard lock(g_mutex);
  g_handles.clear();
  g_next_handle = 1;
  g_chunk_count = 0;
  g_speed = 2;
  g_language_mask = ~u64(0);
});

}  // namespace

Service& PlayGoService() {
  return g_play_go;
}

}  // namespace kern::ipmi
