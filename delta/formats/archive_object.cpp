/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Generic container mount: sniffs a .rar/.zip, indexes its entries once and
// serves file bytes by decompressing on demand. The per-format decoding lives
// behind ArchiveBackend (archive_rar.cpp, archive_zip.cpp); everything here is
// format-agnostic bookkeeping: the index cache, the wrapper-directory strip and
// the decompressed-file cache.

#include "archive_object.h"
#include "archive_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <sys/stat.h>
#include <unordered_map>

#include <base/environment_variables.h>
#include <base/logging.h>
#include <utl/file.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(int, kCacheMb, "DELTA_ARCHIVE_CACHE_MB", 512,
             "budget for decompressed archive entries held in memory");
DELTA_OPTION(int, kSmallMb, "DELTA_ARCHIVE_SMALL_MB", 64,
             "entries up to this size are cached whole rather than streamed");
DELTA_OPTION(bool, kNoIndexCache, "DELTA_ARCHIVE_NO_INDEX_CACHE", false,
             "always re-walk the container instead of reading the index cache");
} // namespace

namespace vfs {
namespace {

constexpr char kIndexMagic[8] = {'D', 'A', 'R', 'I', 'D', 'X', '0', '1'};

// FNV-1a over the archive's full path. Combined with its size this names the
// index cache file; an archive edited in place without changing size would go
// stale, which is not a case a read-only game dump hits.
u64 hashPath(const char *s) {
  u64 h = 0xcbf29ce484222325ull;
  for (; *s; ++s) {
    h ^= static_cast<u8>(*s);
    h *= 0x100000001b3ull;
  }
  return h;
}

std::string indexCachePath(const char *archivePath, u64 archiveSize) {
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  std::string dir =
      std::string(home.empty() ? "." : (const char *)home.c_str()) +
      "/.prosperity/archive-index";
  char name[64];
  std::snprintf(name, sizeof(name), "/%016llx-%016llx.idx",
                (unsigned long long)hashPath(archivePath),
                (unsigned long long)archiveSize);
  return dir + name;
}

void put64(std::vector<u8> &v, u64 x) {
  for (int i = 0; i < 8; i++)
    v.push_back(static_cast<u8>(x >> (i * 8)));
}
void put32(std::vector<u8> &v, u32 x) {
  for (int i = 0; i < 4; i++)
    v.push_back(static_cast<u8>(x >> (i * 8)));
}

struct Reader {
  const u8 *p, *end;
  bool ok = true;
  u64 u64v() {
    if (p + 8 > end) {
      ok = false;
      return 0;
    }
    u64 x = 0;
    for (int i = 0; i < 8; i++)
      x |= u64(p[i]) << (i * 8);
    p += 8;
    return x;
  }
  u32 u32v() {
    if (p + 4 > end) {
      ok = false;
      return 0;
    }
    u32 x = 0;
    for (int i = 0; i < 4; i++)
      x |= u32(p[i]) << (i * 8);
    p += 4;
    return x;
  }
  std::string str(u32 n) {
    if (p + n > end) {
      ok = false;
      return {};
    }
    std::string s(reinterpret_cast<const char *>(p), n);
    p += n;
    return s;
  }
};

bool loadIndexCache(const std::string &path, const char *backend,
                    std::vector<ArchiveEntry> &out) {
  utl::File f(base::String(path.c_str()), utl::fileMode::read);
  if (!f.IsOpen())
    return false;
  const u64 size = f.GetSize();
  if (size < sizeof(kIndexMagic))
    return false;
  std::vector<u8> buf(static_cast<size_t>(size));
  if (f.Read(buf.data(), buf.size()) != size)
    return false;

  Reader r{buf.data(), buf.data() + buf.size()};
  if (std::memcmp(r.p, kIndexMagic, sizeof(kIndexMagic)) != 0)
    return false;
  r.p += sizeof(kIndexMagic);
  const u32 nameLen = r.u32v();
  if (!r.ok || r.str(nameLen) != backend)
    return false;
  const u64 count = r.u64v();
  if (!r.ok || count > (1ull << 26))
    return false;

  out.clear();
  out.reserve(static_cast<size_t>(count));
  for (u64 i = 0; i < count; i++) {
    ArchiveEntry e;
    const u32 pathLen = r.u32v();
    e.path = r.str(pathLen);
    e.size = r.u64v();
    e.packedSize = r.u64v();
    e.dataOffset = r.u64v();
    e.extra = r.u64v();
    e.method = r.u32v();
    e.crc = r.u32v();
    if (!r.ok)
      return false;
    out.push_back(std::move(e));
  }
  return true;
}

// mkdir -p for the index cache directory.
void makeDirs(std::string p) {
  for (size_t i = 1; i < p.size(); i++) {
    if (p[i] == '/') {
      p[i] = 0;
      ::mkdir(p.c_str(), 0755);
      p[i] = '/';
    }
  }
  ::mkdir(p.c_str(), 0755);
}

void saveIndexCache(const std::string &path, const char *backend,
                    const std::vector<ArchiveEntry> &entries) {
  const size_t slash = path.find_last_of('/');
  if (slash != std::string::npos)
    makeDirs(path.substr(0, slash));

  std::vector<u8> buf;
  buf.reserve(entries.size() * 96);
  buf.insert(buf.end(), kIndexMagic, kIndexMagic + sizeof(kIndexMagic));
  const u32 nameLen = static_cast<u32>(std::strlen(backend));
  put32(buf, nameLen);
  buf.insert(buf.end(), backend, backend + nameLen);
  put64(buf, entries.size());
  for (const auto &e : entries) {
    put32(buf, static_cast<u32>(e.path.size()));
    buf.insert(buf.end(), e.path.begin(), e.path.end());
    put64(buf, e.size);
    put64(buf, e.packedSize);
    put64(buf, e.dataOffset);
    put64(buf, e.extra);
    put32(buf, e.method);
    put32(buf, e.crc);
  }

  utl::File f(base::String(path.c_str()), utl::fileMode::write);
  if (!f.IsOpen()) {
    BASE_LOGW("archive", "could not write index cache {}", path.c_str());
    return;
  }
  f.Write(buf.data(), buf.size());
}

// The console's /app0 is case-insensitive, and titles rely on it: Demon's Souls
// is dumped all-lowercase but its engine opens mixed-case paths off its own
// command line ("$/CoreData/EngineSupport/DebugMenu/DebugRenderMenu.txt"), so an
// exact-match lookup loses files that are plainly there. Keys are folded, the
// entry keeps its real name for listings.
std::string foldCase(std::string s) {
  for (char &c : s)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return s;
}

// Length of the leading directory run every entry shares, e.g. 15 for a set of
// paths all under "PPSA01342-app/". Dumps are archived with the game wrapped in
// one (sometimes two) such directories, and the guest expects /app0 to be the
// game root, not the wrapper.
size_t commonWrapperPrefix(const std::vector<ArchiveEntry> &entries) {
  if (entries.empty())
    return 0;
  size_t prefix = 0;
  for (int depth = 0; depth < 4; depth++) {
    const std::string &first = entries[0].path;
    const size_t slash = first.find('/', prefix);
    if (slash == std::string::npos)
      break;
    const size_t next = slash + 1;
    bool shared = true;
    for (const auto &e : entries) {
      if (e.path.size() <= next ||
          e.path.compare(0, next, first, 0, next) != 0) {
        shared = false;
        break;
      }
    }
    if (!shared)
      break;
    prefix = next;
  }
  return prefix;
}

} // namespace

struct ArchiveImpl {
  std::unique_ptr<ArchiveBackend> backend;
  std::vector<ArchiveEntry> entries;
  std::unordered_map<std::string, ArchiveFilesystem::Node> files; // case-folded
  std::vector<std::string> guestPaths; // as stored, for listings
  bool ok = false;

  // Whole decompressed entries, most recently used first. Small files are read
  // over and over (the guest reopens headers and manifests constantly) and
  // re-inflating them each time is what makes a container mount feel slow.
  struct Slot {
    u32 index;
    std::vector<u8> data;
  };
  std::mutex lock;
  std::list<Slot> lru;
  std::unordered_map<u32, std::list<Slot>::iterator> lruIndex;
  u64 lruBytes = 0;

  explicit ArchiveImpl(const base::String &path) {
    backend = openRarBackend(path);
    if (!backend)
      backend = openZipBackend(path);
    if (!backend) {
      BASE_LOGE("archive", "{} is not a container we can read", path.c_str());
      return;
    }

    utl::File probe(path, utl::fileMode::read);
    const u64 archiveSize = probe.IsOpen() ? probe.GetSize() : 0;
    probe.Close();
    const std::string cache = indexCachePath(path.c_str(), archiveSize);

    bool fromCache = false;
    if (!kNoIndexCache && loadIndexCache(cache, backend->name(), entries)) {
      fromCache = true;
    } else if (!backend->index(entries)) {
      BASE_LOGE("archive", "failed to index {}", path.c_str());
      return;
    }
    if (entries.empty()) {
      BASE_LOGE("archive", "{} holds no files", path.c_str());
      return;
    }
    if (!fromCache && !kNoIndexCache)
      saveIndexCache(cache, backend->name(), entries);

    const size_t strip = commonWrapperPrefix(entries);
    files.reserve(entries.size() * 2);
    guestPaths.reserve(entries.size());
    for (u32 i = 0; i < entries.size(); i++) {
      std::string path = "/" + entries[i].path.substr(strip);
      files.emplace(foldCase(path), ArchiveFilesystem::Node{entries[i].size, i});
      guestPaths.push_back(std::move(path));
    }
    ok = true;

    const std::string root =
        strip ? ", root " + entries[0].path.substr(0, strip) : std::string();
    BASE_LOGI("archive", "{} ({}): {} files{}{}", path.c_str(),
              backend->name(), entries.size(),
              fromCache ? ", index cached" : "", root.c_str());
  }

  i64 read(const ArchiveFilesystem::Node &node, void *buf, i64 off, i64 len) {
    if (off < 0 || len < 0 || node.index >= entries.size())
      return -1;
    if (static_cast<u64>(off) >= node.size)
      return 0;
    len = std::min<i64>(len, static_cast<i64>(node.size - off));
    if (len == 0)
      return 0;

    const u64 smallMax = u64(std::max(0, (int)kSmallMb)) << 20;
    if (node.size > smallMax)
      return backend->extractRange(entries[node.index], buf, off, len);

    std::lock_guard<std::mutex> guard(lock);
    auto it = lruIndex.find(node.index);
    if (it == lruIndex.end()) {
      std::vector<u8> data(static_cast<size_t>(node.size));
      const i64 got = backend->extractRange(entries[node.index], data.data(), 0,
                                            static_cast<i64>(node.size));
      if (got < 0)
        return -1;
      data.resize(static_cast<size_t>(got));
      lru.push_front(Slot{node.index, std::move(data)});
      lruIndex[node.index] = lru.begin();
      lruBytes += lru.front().data.size();
      trim();
      it = lruIndex.find(node.index);
      if (it == lruIndex.end()) {
        // The entry alone blew the budget, so it was trimmed straight back out.
        return backend->extractRange(entries[node.index], buf, off, len);
      }
    } else if (it->second != lru.begin()) {
      lru.splice(lru.begin(), lru, it->second);
    }

    const std::vector<u8> &data = lru.front().data;
    if (static_cast<u64>(off) >= data.size())
      return 0;
    len = std::min<i64>(len, static_cast<i64>(data.size() - off));
    std::memcpy(buf, data.data() + off, static_cast<size_t>(len));
    return len;
  }

  // Directory path (folded, with trailing '/') -> immediate children. Built once
  // on the first listing: scanning all 223k paths per readdir made directory
  // enumeration quadratic.
  std::unordered_map<std::string, std::vector<ArchiveFilesystem::Child>> dirs;
  bool dirsBuilt = false;

  void buildDirs() {
    if (dirsBuilt)
      return;
    dirsBuilt = true;
    std::unordered_map<std::string, std::unordered_map<std::string, bool>> seen;
    for (const auto &p : guestPaths) {
      // Register the file under its parent, and every ancestor under its own.
      size_t pos = 0;
      while (true) {
        const size_t slash = p.find('/', pos + 1);
        const std::string parent = foldCase(p.substr(0, pos + 1));
        if (slash == std::string::npos) {
          seen[parent].emplace(p.substr(pos + 1), false);
          break;
        }
        seen[parent].emplace(p.substr(pos + 1, slash - pos - 1), true);
        pos = slash;
      }
    }
    for (auto &kv : seen) {
      auto &out = dirs[kv.first];
      out.reserve(kv.second.size());
      for (auto &child : kv.second)
        out.push_back({child.first, child.second});
    }
  }

  void trim() {
    const u64 budget = u64(std::max(0, (int)kCacheMb)) << 20;
    while (lruBytes > budget && !lru.empty()) {
      auto &back = lru.back();
      lruBytes -= back.data.size();
      lruIndex.erase(back.index);
      lru.pop_back();
    }
  }
};

ArchiveFilesystem::ArchiveFilesystem(const base::String &archivePath)
    : impl_(std::make_unique<ArchiveImpl>(archivePath)) {}
ArchiveFilesystem::~ArchiveFilesystem() = default;

bool ArchiveFilesystem::valid() const { return impl_ && impl_->ok; }

const ArchiveFilesystem::Node *ArchiveFilesystem::find(const char *rel) const {
  if (!impl_ || !impl_->ok || !rel)
    return nullptr;
  auto it = impl_->files.find(foldCase(rel));
  return it == impl_->files.end() ? nullptr : &it->second;
}

i64 ArchiveFilesystem::read(const Node &node, void *buf, i64 off, i64 len) {
  return impl_ && impl_->ok ? impl_->read(node, buf, off, len) : -1;
}

bool ArchiveFilesystem::list(const char *rel, std::vector<Child> &out) {
  if (!impl_ || !impl_->ok)
    return false;
  std::string prefix(rel ? rel : "");
  while (!prefix.empty() && prefix.back() == '/')
    prefix.pop_back();
  prefix += "/";
  if (prefix[0] != '/')
    prefix.insert(prefix.begin(), '/');

  std::lock_guard<std::mutex> guard(impl_->lock);
  impl_->buildDirs();
  auto it = impl_->dirs.find(foldCase(prefix));
  if (it == impl_->dirs.end())
    return false;
  out.insert(out.end(), it->second.begin(), it->second.end());
  return !out.empty();
}

void ArchiveFilesystem::paths(std::vector<std::string> &out) const {
  if (!impl_)
    return;
  out.insert(out.end(), impl_->guestPaths.begin(), impl_->guestPaths.end());
}

const char *ArchiveFilesystem::backendName() const {
  return impl_ && impl_->backend ? impl_->backend->name() : "";
}

bool isArchivePath(const char *path) {
  if (!path)
    return false;
  static const char *kExts[] = {".rar", ".zip"};
  const size_t n = std::strlen(path);
  for (const char *ext : kExts) {
    const size_t e = std::strlen(ext);
    if (n < e)
      continue;
    bool match = true;
    for (size_t i = 0; i < e; i++) {
      char a = path[n - e + i];
      if (a >= 'A' && a <= 'Z')
        a += 'a' - 'A';
      if (a != ext[i]) {
        match = false;
        break;
      }
    }
    if (match)
      return true;
  }
  return false;
}

} // namespace vfs
