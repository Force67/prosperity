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

#include "archive_backend.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <list>
#include <mutex>
#include <thread>

#include <utl/file.h>

#include <unrar/rar.hpp>

namespace vfs {
namespace {

// Decode-ahead per session. Must comfortably exceed the largest single
// UnpWrite burst, which is bounded by the dictionary flush granularity.
constexpr size_t kRingSize = 8 << 20;
constexpr size_t kMaxSessions = 4;

// One in-flight decode of one entry: a worker thread unpacking from entry
// byte 0, and a cursor of how far the consumer has drained it.
struct RarSession {
  u64 headerOff = 0;

  std::mutex m;
  std::condition_variable canProduce, canConsume;
  std::vector<u8> ring;
  u64 produced = 0; // total bytes the decoder pushed
  u64 consumed = 0; // total bytes drained (delivered or discarded)
  bool done = false;
  bool ok = false;   // full decode finished and the stored hash matched
  bool stop = false; // eviction: abort the decoder

  std::mutex readLock; // serialises consumers of this session
  std::thread worker;

  // Abort the decode and wait for the worker to notice. Safe to call twice.
  void retire() {
    {
      std::lock_guard<std::mutex> lk(m);
      stop = true;
    }
    canProduce.notify_all();
    if (worker.joinable())
      worker.join();
  }

  ~RarSession() {
    {
      std::lock_guard<std::mutex> lk(m);
      stop = true;
    }
    canProduce.notify_all();
    if (!worker.joinable())
      return;
    // decodeEntry holds a shared_ptr to this session, so when a decode outlives
    // every consumer the last reference dies on the worker itself as that
    // parameter is destroyed. Joining there is joining ourselves, which throws
    // std::system_error(EDEADLK) out of a thread with no handler and aborts the
    // emulator. The worker is returning anyway, so let it go.
    if (worker.get_id() == std::this_thread::get_id())
      worker.detach();
    else
      worker.join();
  }

  // Decoder side: blocks while the ring is full. False aborts the decode.
  bool push(const u8 *data, size_t count) {
    std::unique_lock<std::mutex> lk(m);
    while (count) {
      canProduce.wait(lk, [&] { return stop || produced - consumed < kRingSize; });
      if (stop)
        return false;
      size_t space = kRingSize - size_t(produced - consumed);
      size_t wpos = size_t(produced % kRingSize);
      size_t n = std::min({count, space, kRingSize - wpos});
      std::memcpy(ring.data() + wpos, data, n);
      produced += n;
      data += n;
      count -= n;
      canConsume.notify_all();
    }
    return true;
  }

  void finish(bool success) {
    {
      std::lock_guard<std::mutex> lk(m);
      done = true;
      ok = success;
    }
    canConsume.notify_all();
  }

  // Consumer side: discards [consumed, off), then copies [off, off+len).
  // Caller must hold readLock and have verified consumed <= off.
  i64 consume(u64 off, u8 *out, u64 len) {
    std::unique_lock<std::mutex> lk(m);
    while (consumed < off + len) {
      u64 avail = produced - consumed;
      if (avail == 0) {
        if (done)
          return -1; // stream ended short of the range: corrupt
        canConsume.wait(lk, [&] { return done || produced > consumed; });
        continue;
      }
      if (done && !ok)
        return -1; // full decode finished but the checksum failed
      if (consumed < off) {
        consumed += std::min(avail, off - consumed);
        canProduce.notify_all();
        continue;
      }
      size_t rpos = size_t(consumed % kRingSize);
      size_t n = size_t(std::min({avail, off + len - consumed, u64(kRingSize - rpos)}));
      std::memcpy(out + (consumed - off), ring.data() + rpos, n);
      consumed += n;
      canProduce.notify_all();
    }
    return i64(len);
  }
};

extern "C" int rarSessionCallback(UINT msg, LPARAM userData, LPARAM p1, LPARAM p2) {
  if (msg != UCM_PROCESSDATA)
    return 0;
  auto *s = reinterpret_cast<RarSession *>(userData);
  return s->push(reinterpret_cast<const u8 *>(p1), size_t(p2)) ? 1 : -1;
}

// Mirrors CmdExtract::ExtractCurrentFile's DataIO/Unpack setup for one
// non-solid entry whose header sits at headerOff. Runs on the worker thread;
// UnRAR reports fatal errors by throwing RAR_EXIT.
void decodeEntry(std::shared_ptr<RarSession> s, std::wstring arcPath, u64 headerOff) {
  bool success = false;
  try {
    CommandData cmd;
    cmd.DllOpMode = RAR_TEST; // route UnpWrite into the callback
    cmd.Callback = rarSessionCallback;
    cmd.UserData = reinterpret_cast<LPARAM>(s.get());
    cmd.FileArgs.AddString(L"*");

    Archive arc(&cmd);
    if (arc.Open(arcPath) && arc.IsArchive(false)) {
      arc.Seek(headerOff, SEEK_SET);
      if (arc.ReadHeader() > 0 && arc.GetHeaderType() == HEAD_FILE &&
          !arc.FileHead.Encrypted && !arc.FileHead.Solid &&
          !arc.FileHead.SplitBefore && !arc.FileHead.SplitAfter) {
        FileHeader &hd = arc.FileHead;
        ComprDataIO dataIO;
        dataIO.SetEncryption(false, CRYPT_NONE, nullptr, nullptr, nullptr, 0,
                             nullptr, nullptr);
        dataIO.CurUnpRead = 0;
        dataIO.CurUnpWrite = 0;
        dataIO.UnpHash.Init(hd.FileHash.Type, 1);
        dataIO.PackedDataHash.Init(hd.FileHash.Type, 1);
        dataIO.SetPackedSizeToRead(hd.PackSize);
        dataIO.SetFiles(&arc, nullptr);
        dataIO.SetTestMode(true);
        dataIO.SetSkipUnpCRC(false);

        if (hd.Method == 0) {
          std::vector<byte> buf(0x40000);
          int64 left = hd.UnpSize;
          while (left > 0) {
            int r = dataIO.UnpRead(buf.data(), buf.size());
            if (r <= 0)
              break;
            int w = int(std::min(int64(r), left));
            dataIO.UnpWrite(buf.data(), w);
            left -= w;
          }
        } else {
          Unpack unp(&dataIO);
          unp.Init(hd.WinSize, false);
          unp.SetDestSize(hd.UnpSize);
          if (arc.Format != RARFMT50 && hd.UnpVer <= 15)
            unp.DoUnpack(15, false);
          else
            unp.DoUnpack(hd.UnpVer, false);
        }
        success = dataIO.CurUnpWrite == hd.UnpSize &&
                  dataIO.UnpHash.Cmp(&hd.FileHash,
                                     hd.UseHashKey ? hd.HashKey : nullptr);
      }
    }
  } catch (RAR_EXIT) {
  } catch (std::bad_alloc &) {
  }
  s->finish(success);
}

class RarBackend final : public ArchiveBackend {
public:
  RarBackend(const base::String &path, std::wstring pathW)
      : pathW_(std::move(pathW)), rawFile_(path) {}

  bool index(std::vector<ArchiveEntry> &out) override;
  i64 extractRange(const ArchiveEntry &entry, void *buf, i64 off, i64 len) override;
  const char *name() const override { return "rar"; }

private:
  std::shared_ptr<RarSession> acquireSession(const ArchiveEntry &entry, u64 off);

  std::wstring pathW_;
  utl::File rawFile_;     // stored (method 0) entries are read directly
  std::mutex rawMutex_;   // guards rawFile_'s seek+read
  std::mutex cacheMutex_; // guards sessions_ (MRU front)
  std::list<std::shared_ptr<RarSession>> sessions_;
};

bool RarBackend::index(std::vector<ArchiveEntry> &out) {
  CommandData cmd;
  cmd.FileArgs.AddString(L"*");
  Archive arc(&cmd);
  try {
    if (!arc.Open(pathW_) || !arc.IsArchive(false))
      return false;
    while (arc.ReadHeader() > 0) {
      if (arc.GetHeaderType() == HEAD_FILE && !arc.IsArcDir() &&
          arc.FileHead.RedirType == FSREDIR_NONE &&
          !arc.FileHead.SplitBefore && !arc.FileHead.UnknownUnpSize) {
        FileHeader &hd = arc.FileHead;
        ArchiveEntry e;
        std::string utf;
        WideToUtf(hd.FileName, utf);
        std::replace(utf.begin(), utf.end(), '\\', '/');
        while (!utf.empty() && utf.front() == '/')
          utf.erase(utf.begin());
        e.path = std::move(utf);
        e.size = u64(hd.UnpSize);
        e.packedSize = u64(hd.PackSize);
        e.dataOffset = u64(arc.NextBlockPos - hd.PackSize);
        e.extra = u64(arc.CurBlockPos);
        e.method = hd.Method;
        e.crc = hd.FileHash.Type == HASH_CRC32 ? hd.FileHash.CRC32 : 0;
        out.push_back(std::move(e));
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
std::shared_ptr<RarSession> RarBackend::acquireSession(const ArchiveEntry &entry,
                                                       u64 off) {
  std::shared_ptr<RarSession> evicted, best;
  bool evictedIdle = false;
  {
    std::lock_guard<std::mutex> lk(cacheMutex_);
    u64 bestPos = 0;
    std::list<std::shared_ptr<RarSession>>::iterator bestIt;
    for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
      auto &s = **it;
      if (s.headerOff != entry.extra)
        continue;
      u64 pos;
      {
        std::lock_guard<std::mutex> sl(s.m);
        pos = s.consumed;
      }
      if (pos <= off && (!best || pos >= bestPos)) {
        best = *it;
        bestPos = pos;
        bestIt = it;
      }
    }
    if (best) {
      sessions_.erase(bestIt);
    } else {
      best = std::make_shared<RarSession>();
      best->headerOff = entry.extra;
      best->ring.resize(kRingSize);
      best->worker = std::thread(decodeEntry, best, pathW_, entry.extra);
    }
    sessions_.push_front(best);
    if (sessions_.size() > kMaxSessions) {
      evicted = sessions_.back();
      sessions_.pop_back();
      // Once it is out of the list nobody new can find it, so the only refs
      // left are this local and the worker's own parameter. Anything more means
      // a consumer is still draining it, and stopping the decode would fail
      // that read.
      evictedIdle = evicted.use_count() <= 2;
    }
  }
  // An evicted session with no consumer left would otherwise sit in push()
  // waiting on a ring nobody drains, holding its worker thread forever: with
  // one worker per streamed asset the title runs out of threads and its loader
  // never completes. Retire it here, outside the cache lock.
  if (evicted && evictedIdle)
    evicted->retire();
  return best;
}

i64 RarBackend::extractRange(const ArchiveEntry &entry, void *buf, i64 off,
                             i64 len) {
  if (off < 0 || len < 0)
    return -1;
  if (u64(off) >= entry.size)
    return 0;
  len = std::min(len, i64(entry.size - u64(off)));
  if (len == 0)
    return 0;

  if (entry.method == 0) {
    std::lock_guard<std::mutex> lk(rawMutex_);
    if (!rawFile_.Exists())
      return -1;
    rawFile_.Seek(entry.dataOffset + u64(off), utl::seekMode::seek_set);
    return rawFile_.Read(buf, size_t(len)) == u64(len) ? len : -1;
  }

  for (int attempt = 0; attempt < 4; attempt++) {
    auto s = acquireSession(entry, u64(off));
    std::lock_guard<std::mutex> rl(s->readLock);
    {
      std::lock_guard<std::mutex> sl(s->m);
      if (s->consumed > u64(off))
        continue; // a concurrent reader advanced past us; retry fresh
    }
    return s->consume(u64(off), static_cast<u8 *>(buf), u64(len));
  }

  // Contended fallback: start a decode and take its read lock before other
  // threads can find it in the cache, so nobody can advance it past us.
  auto s = std::make_shared<RarSession>();
  s->headerOff = entry.extra;
  s->ring.resize(kRingSize);
  s->worker = std::thread(decodeEntry, s, pathW_, entry.extra);
  std::lock_guard<std::mutex> rl(s->readLock);
  {
    std::shared_ptr<RarSession> evicted;
    bool evictedIdle = false;
    {
      std::lock_guard<std::mutex> lk(cacheMutex_);
      sessions_.push_front(s);
      if (sessions_.size() > kMaxSessions) {
        evicted = sessions_.back();
        sessions_.pop_back();
        evictedIdle = evicted.use_count() <= 2;
      }
    }
    if (evicted && evictedIdle)
      evicted->retire();
  }
  return s->consume(u64(off), static_cast<u8 *>(buf), u64(len));
}

} // namespace

std::unique_ptr<ArchiveBackend> openRarBackend(const base::String &path) {
  utl::File f(path);
  if (!f.Exists() || !f.IsOpen())
    return nullptr;
  u8 magic[8]{};
  if (f.Read(magic, sizeof(magic)) != sizeof(magic))
    return nullptr;
  if (std::memcmp(magic, "Rar!\x1a\x07", 6) != 0)
    return nullptr;
  std::wstring pathW;
  if (!UtfToWide(path.c_str(), pathW))
    return nullptr;
  return std::make_unique<RarBackend>(path, std::move(pathW));
}

} // namespace vfs
