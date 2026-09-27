/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// RAR backend over the vendored UnRAR sources. CmdExtract is bypassed (it
// insists on writing files); instead each decode drives Archive + ComprDataIO
// + Unpack the way CmdExtract::ExtractCurrentFile does, with RARDLL's
// UCM_PROCESSDATA callback landing the output in a ring buffer. A decode runs
// on its own worker thread so a guest can consume the stream incrementally:
// the callback blocks once the ring is full, which caps decode-ahead and lets
// a forward read resume where the last one stopped instead of restarting a
// multi-GB stream from byte 0.

#include "formats/archive_backend.h"

#include <cstring>
#include <string>

#include "base/logging.h"
#include "io/file.h"

// UnRAR's API speaks std::wstring and throws std::bad_alloc: both stay at
// its boundary. Its Min/Max macros would rewrite base::Min/Max.
#include <unrar/rar.hpp>
#undef Min
#undef Max
#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"

namespace vfs {
namespace {

// Decode-ahead per session. Must comfortably exceed the largest single
// UnpWrite burst, which is bounded by the dictionary flush granularity.
constexpr size_t kRingSize = 8 << 20;
constexpr size_t kMaxSessions = 4;

// DELTA_RAR_STAT: a decode always starts at entry byte 0, so every session the
// cache could not reuse re-inflates everything before the range asked for. The
// ratio of bytes inflated to bytes delivered is what says whether a slow boot
// is the archive thrashing or the title genuinely reading that much.
struct RarStats {
  base::Atomic<u64> sessions{0};   // decodes started
  base::Atomic<u64> reused{0};     // reads served by an existing session
  base::Atomic<u64> evicted{0};    // sessions dropped while still useful
  base::Atomic<u64> inflated{0};   // bytes the decoder produced
  base::Atomic<u64> delivered{0};  // bytes handed to the guest
  base::Atomic<u64> discarded{0};  // bytes skipped to reach the wanted offset
  base::Atomic<u64> whole{0};      // entries decoded in one shot, no session
};
RarStats& Stats() {
  static RarStats s;
  return s;
}
void ReportStats() {
  static base::Atomic<u64> last{0};
  const u64 n = Stats().delivered.load() >> 24;  // every 16 MiB delivered
  u64 prev = last.load();
  if (n == prev || !last.compare_exchange_strong(prev, n))
    return;
  const auto& st = Stats();
  BASE_LOGI("rarstat",
            "sessions={} reused={} evicted={} whole={} inflated={} MiB "
            "delivered={} MiB discarded={} MiB",
            st.sessions.load(), st.reused.load(), st.evicted.load(),
            st.whole.load(), st.inflated.load() >> 20,
            st.delivered.load() >> 20, st.discarded.load() >> 20);
}

// One in-flight decode of one entry: a worker thread unpacking from entry
// byte 0, and a cursor of how far the consumer has drained it. The worker
// borrows the session; whoever drops the last reference stops and joins it.
struct RarSession {
  u64 header_off = 0;

  base::Mutex m;
  base::ConditionVariable can_produce, can_consume;
  base::Vector<u8> ring;
  u64 produced = 0;  // total bytes the decoder pushed
  u64 consumed = 0;  // total bytes drained (delivered or discarded)
  bool done = false;
  bool ok = false;    // full decode finished and the stored hash matched
  bool stop = false;  // eviction: abort the decoder

  base::Mutex read_lock;  // serialises consumers of this session
  base::UniquePointer<base::Thread> worker;

  ~RarSession() {
    {
      base::LockGuard<base::Mutex> lk(m);
      stop = true;
    }
    can_produce.NotifyAll();
    if (worker)
      worker->Join();
  }

  // Decoder side: blocks while the ring is full. False aborts the decode.
  bool Push(const u8* data, size_t count) {
    base::UniqueLock<base::Mutex> lk(m);
    while (count) {
      can_produce.Wait(lk,
                       [&] { return stop || produced - consumed < kRingSize; });
      if (stop)
        return false;
      size_t space = kRingSize - size_t(produced - consumed);
      size_t wpos = size_t(produced % kRingSize);
      size_t n = base::Min(count, base::Min(space, kRingSize - wpos));
      std::memcpy(ring.data() + wpos, data, n);
      produced += n;
      data += n;
      count -= n;
      can_consume.NotifyAll();
    }
    return true;
  }

  void Finish(bool success) {
    {
      base::LockGuard<base::Mutex> lk(m);
      done = true;
      ok = success;
    }
    can_consume.NotifyAll();
  }

  // Consumer side: discards [consumed, off), then copies [off, off+len).
  // Caller must hold readLock and have verified consumed <= off.
  i64 Consume(u64 off, u8* out, u64 len) {
    base::UniqueLock<base::Mutex> lk(m);
    while (consumed < off + len) {
      u64 avail = produced - consumed;
      if (avail == 0) {
        if (done)
          return -1;  // stream ended short of the range: corrupt
        can_consume.Wait(lk, [&] { return done || produced > consumed; });
        continue;
      }
      if (done && !ok)
        return -1;  // full decode finished but the checksum failed
      if (consumed < off) {
        const u64 skip = base::Min(avail, off - consumed);
        Stats().discarded.fetch_add(skip, base::memory_order_relaxed);
        consumed += skip;
        can_produce.NotifyAll();
        continue;
      }
      size_t rpos = size_t(consumed % kRingSize);
      size_t n = size_t(base::Min(
          avail, base::Min(off + len - consumed, u64(kRingSize - rpos))));
      std::memcpy(out + (consumed - off), ring.data() + rpos, n);
      consumed += n;
      can_produce.NotifyAll();
    }
    return i64(len);
  }
};

extern "C" int RarSessionCallback(UINT msg,
                                  LPARAM user_data,
                                  LPARAM p1,
                                  LPARAM p2) {
  if (msg != UCM_PROCESSDATA)
    return 0;
  auto* s = reinterpret_cast<RarSession*>(user_data);
  Stats().inflated.fetch_add(size_t(p2), base::memory_order_relaxed);
  return s->Push(reinterpret_cast<const u8*>(p1), size_t(p2)) ? 1 : -1;
}

// Everything a decode needs that is worth keeping between entries. Unpack owns
// the LZ dictionary window, which for RAR5 is tens of megabytes, and
// Unpack::Init reuses it whenever the next entry's window is no larger
// (`WinSize <= AllocWinSize`). Constructing a fresh Unpack per entry threw that
// away and allocated + first-touched the whole window again: the average entry
// in a game archive is ~130 KB, so decoding it paid for a 32 MB window. Keeping
// the Archive open across entries saves re-opening and re-validating the
// container as well.
struct RarDecoder {
  CommandData cmd;
  Archive arc{&cmd};
  ComprDataIO data_io;
  Unpack unp{&data_io};
  bool opened = false;

  explicit RarDecoder(const std::wstring& arc_path) {
    cmd.DllOpMode = RAR_TEST;  // route UnpWrite into the callback
    cmd.FileArgs.AddString(L"*");
    try {
      opened = arc.Open(arc_path) && arc.IsArchive(false);
    } catch (RAR_EXIT) {
    } catch (std::bad_alloc&) {
    }
  }

  // Decode one non-solid entry whose header sits at headerOff, delivering its
  // bytes through cmd.Callback with `user` as the callback's userData.
  bool Decode(u64 header_off, UNRARCALLBACK cb, LPARAM user) {
    if (!opened)
      return false;
    cmd.Callback = cb;
    cmd.UserData = user;
    try {
      arc.Seek(header_off, SEEK_SET);
      if (arc.ReadHeader() <= 0 || arc.GetHeaderType() != HEAD_FILE ||
          arc.FileHead.Encrypted || arc.FileHead.Solid ||
          arc.FileHead.SplitBefore || arc.FileHead.SplitAfter)
        return false;
      FileHeader& hd = arc.FileHead;
      data_io.SetEncryption(false, CRYPT_NONE, nullptr, nullptr, nullptr, 0,
                            nullptr, nullptr);
      data_io.CurUnpRead = 0;
      data_io.CurUnpWrite = 0;
      data_io.UnpHash.Init(hd.FileHash.Type, 1);
      data_io.PackedDataHash.Init(hd.FileHash.Type, 1);
      data_io.SetPackedSizeToRead(hd.PackSize);
      data_io.SetFiles(&arc, nullptr);
      data_io.SetTestMode(true);
      data_io.SetSkipUnpCRC(false);

      if (hd.Method == 0) {
        base::Vector<byte> buf(0x40000);
        int64 left = hd.UnpSize;
        while (left > 0) {
          int r = data_io.UnpRead(buf.data(), buf.size());
          if (r <= 0)
            break;
          int w = int(base::Min(int64(r), left));
          data_io.UnpWrite(buf.data(), w);
          left -= w;
        }
      } else {
        unp.Init(hd.WinSize, false);
        unp.SetDestSize(hd.UnpSize);
        if (arc.Format != RARFMT50 && hd.UnpVer <= 15)
          unp.DoUnpack(15, false);
        else
          unp.DoUnpack(hd.UnpVer, false);
      }
      return data_io.CurUnpWrite == hd.UnpSize &&
             data_io.UnpHash.Cmp(&hd.FileHash,
                                 hd.UseHashKey ? hd.HashKey : nullptr);
    } catch (RAR_EXIT) {
    } catch (std::bad_alloc&) {
    }
    return false;
  }
};

// One decoder per decoding thread. A decode is single-threaded and a session's
// worker outlives none of its state, so thread-local ownership needs no lock
// and keeps each thread's window hot.
RarDecoder& ThreadDecoder(const std::wstring& arc_path) {
  static thread_local base::UniquePointer<RarDecoder> d;
  if (!d)
    d = base::MakeUnique<RarDecoder>(arc_path);
  return *d;
}

// Whole-entry decode straight into a caller-owned buffer. No ring, no worker:
// the guest's own thread does the inflate, so the thread-local decoder (and its
// LZ window) is reused across every entry that thread reads instead of being
// built and thrown away per file.
struct WholeSink {
  base::Vector<u8>* out;
};

extern "C" int RarWholeCallback(UINT msg,
                                LPARAM user_data,
                                LPARAM p1,
                                LPARAM p2) {
  if (msg != UCM_PROCESSDATA)
    return 0;
  auto* w = reinterpret_cast<WholeSink*>(user_data);
  const auto* src = reinterpret_cast<const u8*>(p1);
  w->out->insert(w->out->end(), src, src + size_t(p2));
  Stats().inflated.fetch_add(size_t(p2), base::memory_order_relaxed);
  return 1;
}

// Entries at or below this decode in one shot. Bigger ones keep the streaming
// session so a multi-GB read is not held in memory.
constexpr u64 kWholeEntryMax = 16u << 20;

void DecodeEntry(RarSession* s, const std::wstring& arc_path, u64 header_off) {
  s->Finish(ThreadDecoder(arc_path).Decode(header_off, RarSessionCallback,
                                           reinterpret_cast<LPARAM>(s)));
}

class RarBackend final : public ArchiveBackend {
 public:
  RarBackend(const base::String& path, std::wstring path_w)
      : pathW_(base::move(path_w)), rawFile_(path) {}

  bool Index(base::Vector<ArchiveEntry>& out) override;
  i64 ExtractRange(const ArchiveEntry& entry,
                   void* buf,
                   i64 off,
                   i64 len) override;
  const char* Name() const override { return "rar"; }

 private:
  base::SharedPointer<RarSession> AcquireSession(const ArchiveEntry& entry,
                                                 u64 off);

  base::SharedPointer<RarSession> StartSession(u64 header_off);

  std::wstring pathW_;
  io::File rawFile_;        // stored (method 0) entries are read directly
  base::Mutex rawMutex_;    // guards rawFile_'s seek+read
  base::Mutex cacheMutex_;  // guards sessions_ (MRU front)
  base::Vector<base::SharedPointer<RarSession>> sessions_;  // MRU first
};

bool RarBackend::Index(base::Vector<ArchiveEntry>& out) {
  CommandData cmd;
  cmd.FileArgs.AddString(L"*");
  Archive arc(&cmd);
  try {
    if (!arc.Open(pathW_) || !arc.IsArchive(false))
      return false;
    while (arc.ReadHeader() > 0) {
      if (arc.GetHeaderType() == HEAD_FILE && !arc.IsArcDir() &&
          arc.FileHead.RedirType == FSREDIR_NONE && !arc.FileHead.SplitBefore &&
          !arc.FileHead.UnknownUnpSize) {
        FileHeader& hd = arc.FileHead;
        ArchiveEntry e;
        std::string narrow;
        WideToUtf(hd.FileName, narrow);
        base::String utf(narrow.c_str(), narrow.size());
        for (char& c : utf)
          if (c == '\\')
            c = '/';
        while (!utf.empty() && utf.front() == '/')
          utf.erase(0, 1);
        e.path = base::move(utf);
        e.size = u64(hd.UnpSize);
        e.packed_size = u64(hd.PackSize);
        e.data_offset = u64(arc.NextBlockPos - hd.PackSize);
        e.extra = u64(arc.CurBlockPos);
        e.method = hd.Method;
        e.crc = hd.FileHash.Type == HASH_CRC32 ? hd.FileHash.CRC32 : 0;
        out.push_back(base::move(e));
      }
      arc.SeekToNext();
    }
  } catch (RAR_EXIT) {
    return false;
  }
  return true;
}

// Reuse the most advanced cached session that has not passed `off` yet, else
// start a fresh decode. The returned session may still be raced past `off` by
// a concurrent reader; the caller rechecks under readLock.
base::SharedPointer<RarSession> RarBackend::AcquireSession(
    const ArchiveEntry& entry,
    u64 off) {
  base::SharedPointer<RarSession> evicted, best;
  {
    base::LockGuard<base::Mutex> lk(cacheMutex_);
    u64 best_pos = 0;
    base::SharedPointer<RarSession>* best_it = nullptr;
    for (auto* it = sessions_.begin(); it != sessions_.end(); ++it) {
      auto& s = **it;
      if (s.header_off != entry.extra)
        continue;
      u64 pos;
      {
        base::LockGuard<base::Mutex> sl(s.m);
        pos = s.consumed;
      }
      if (pos <= off && (!best || pos >= best_pos)) {
        best = *it;
        best_pos = pos;
        best_it = it;
      }
    }
    if (best) {
      sessions_.erase(best_it);
      Stats().reused.fetch_add(1, base::memory_order_relaxed);
    } else {
      Stats().sessions.fetch_add(1, base::memory_order_relaxed);
      best = StartSession(entry.extra);
    }
    sessions_.insert(sessions_.begin(), best);
    if (sessions_.size() > kMaxSessions) {
      evicted = base::move(sessions_.back());
      sessions_.pop_back();
      Stats().evicted.fetch_add(1, base::memory_order_relaxed);
    }
  }
  // An evicted session no consumer holds is destroyed here, outside the cache
  // lock, which stops its decode; one still being drained stops when its
  // reader lets go.
  return best;
}

base::SharedPointer<RarSession> RarBackend::StartSession(u64 header_off) {
  auto s = base::MakeShared<RarSession>();
  s->header_off = header_off;
  s->ring.resize(kRingSize);
  RarSession* raw = s.get();
  s->worker = base::MakeUnique<base::Thread>(
      "rar-decode",
      [raw, path = pathW_, header_off] { DecodeEntry(raw, path, header_off); },
      /*start_now=*/true);
  return s;
}

i64 RarBackend::ExtractRange(const ArchiveEntry& entry,
                             void* buf,
                             i64 off,
                             i64 len) {
  if (off < 0 || len < 0)
    return -1;
  if (u64(off) >= entry.size)
    return 0;
  len = base::Min(len, i64(entry.size - u64(off)));
  if (len == 0)
    return 0;

  if (entry.method == 0) {
    base::LockGuard<base::Mutex> lk(rawMutex_);
    if (!rawFile_.Exists())
      return -1;
    rawFile_.Seek(entry.data_offset + u64(off), io::SeekMode::kSeekSet);
    return rawFile_.Read(buf, size_t(len)) == u64(len) ? len : -1;
  }

  if (entry.size <= kWholeEntryMax) {
    // Reused per thread so a run of reads does not reallocate; cleared, not
    // freed, so the capacity survives.
    static thread_local base::Vector<u8> whole;
    whole.clear();
    whole.reserve(size_t(entry.size));
    WholeSink sink{&whole};
    if (!ThreadDecoder(pathW_).Decode(entry.extra, RarWholeCallback,
                                      reinterpret_cast<LPARAM>(&sink)) ||
        whole.size() < u64(off) + u64(len))
      return -1;
    std::memcpy(buf, whole.data() + off, size_t(len));
    Stats().delivered.fetch_add(u64(len), base::memory_order_relaxed);
    Stats().whole.fetch_add(1, base::memory_order_relaxed);
    ReportStats();
    return len;
  }

  for (int attempt = 0; attempt < 4; attempt++) {
    auto s = AcquireSession(entry, u64(off));
    base::LockGuard<base::Mutex> rl(s->read_lock);
    {
      base::LockGuard<base::Mutex> sl(s->m);
      if (s->consumed > u64(off))
        continue;  // a concurrent reader advanced past us; retry fresh
    }
    const i64 got = s->Consume(u64(off), static_cast<u8*>(buf), u64(len));
    if (got > 0) {
      Stats().delivered.fetch_add(u64(got), base::memory_order_relaxed);
      ReportStats();
    }
    return got;
  }

  // Contended fallback: start a decode and take its read lock before other
  // threads can find it in the cache, so nobody can advance it past us.
  auto s = StartSession(entry.extra);
  base::LockGuard<base::Mutex> rl(s->read_lock);
  {
    base::SharedPointer<RarSession> evicted;
    {
      base::LockGuard<base::Mutex> lk(cacheMutex_);
      sessions_.insert(sessions_.begin(), s);
      if (sessions_.size() > kMaxSessions) {
        evicted = base::move(sessions_.back());
        sessions_.pop_back();
      }
    }
  }
  return s->Consume(u64(off), static_cast<u8*>(buf), u64(len));
}

}  // namespace

base::UniquePointer<ArchiveBackend> OpenRarBackend(const base::String& path) {
  io::File f(path);
  if (!f.Exists() || !f.IsOpen())
    return nullptr;
  u8 magic[8]{};
  if (f.Read(magic, sizeof(magic)) != sizeof(magic))
    return nullptr;
  if (std::memcmp(magic, "Rar!\x1a\x07", 6) != 0)
    return nullptr;
  std::wstring path_w;
  if (!UtfToWide(path.c_str(), path_w))
    return nullptr;
  return base::MakeUnique<RarBackend>(path, base::move(path_w));
}

}  // namespace vfs
