/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Generic container mount: sniff .rar/.zip, index once, serve bytes by
// on-demand decompression. Per-format decoding lives behind ArchiveBackend;
// everything here is format-agnostic: index cache, wrapper-dir strip,
// decompressed-file cache.

#include "formats/archive_filesystem.h"
#include "formats/archive_backend.h"

#include <sys/stat.h>
#include <cstdio>
#include <cstring>

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/environment_variables.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "io/file.h"
#include "options/options.h"

namespace {
DELTA_OPTION(int,
             kCacheMb,
             "DELTA_ARCHIVE_CACHE_MB",
             512,
             "budget for decompressed archive entries held in memory");
DELTA_OPTION(int,
             kSmallMb,
             "DELTA_ARCHIVE_SMALL_MB",
             64,
             "entries up to this size are cached whole rather than streamed");
DELTA_OPTION(bool,
             kNoIndexCache,
             "DELTA_ARCHIVE_NO_INDEX_CACHE",
             false,
             "always re-walk the container instead of reading the index cache");
}  // namespace

namespace formats {
namespace {

constexpr char kIndexMagic[8] = {'D', 'A', 'R', 'I', 'D', 'X', '0', '1'};

// FNV-1a over the archive's full path. Combined with its size this names the
// index cache file; an archive edited in place without changing size would go
// stale, which is not a case a read-only game dump hits.
u64 HashPath(const char* s) {
  u64 h = 0xcbf29ce484222325ull;
  for (; *s; ++s) {
    h ^= static_cast<u8>(*s);
    h *= 0x100000001b3ull;
  }
  return h;
}

base::String IndexCachePath(const char* archive_path, u64 archive_size) {
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  base::String dir =
      base::String(home.empty() ? "." : (const char*)home.c_str()) +
      "/.prosperity/archive-index";
  char name[64];
  std::snprintf(name, sizeof(name), "/%016llx-%016llx.idx",
                (unsigned long long)HashPath(archive_path),
                (unsigned long long)archive_size);
  return dir + name;
}

void Put64(base::Vector<u8>& v, u64 x) {
  for (int i = 0; i < 8; i++)
    v.push_back(static_cast<u8>(x >> (i * 8)));
}
void Put32(base::Vector<u8>& v, u32 x) {
  for (int i = 0; i < 4; i++)
    v.push_back(static_cast<u8>(x >> (i * 8)));
}

struct Reader {
  const u8 *p, *end;
  bool ok = true;
  u64 U64v() {
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
  u32 U32v() {
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
  base::String Str(u32 n) {
    if (p + n > end) {
      ok = false;
      return {};
    }
    base::String s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
  }
};

bool LoadIndexCache(const base::String& path,
                    const char* backend,
                    base::Vector<ArchiveEntry>& out) {
  io::File f(base::String(path.c_str()), io::FileMode::kRead);
  if (!f.IsOpen())
    return false;
  const u64 size = f.GetSize();
  if (size < sizeof(kIndexMagic))
    return false;
  base::Vector<u8> buf(static_cast<size_t>(size));
  if (f.Read(buf.data(), buf.size()) != size)
    return false;

  Reader r{buf.data(), buf.data() + buf.size()};
  if (std::memcmp(r.p, kIndexMagic, sizeof(kIndexMagic)) != 0)
    return false;
  r.p += sizeof(kIndexMagic);
  const u32 name_len = r.U32v();
  if (!r.ok || r.Str(name_len) != backend)
    return false;
  const u64 count = r.U64v();
  if (!r.ok || count > (1ull << 26))
    return false;

  out.clear();
  out.reserve(static_cast<size_t>(count));
  for (u64 i = 0; i < count; i++) {
    ArchiveEntry e;
    const u32 path_len = r.U32v();
    e.path = r.Str(path_len);
    e.size = r.U64v();
    e.packed_size = r.U64v();
    e.data_offset = r.U64v();
    e.extra = r.U64v();
    e.method = r.U32v();
    e.crc = r.U32v();
    if (!r.ok)
      return false;
    out.push_back(base::move(e));
  }
  return true;
}

// mkdir -p for the index cache directory.
void MakeDirs(base::String p) {
  for (size_t i = 1; i < p.size(); i++) {
    if (p[i] == '/') {
      p[i] = 0;
      ::mkdir(p.c_str(), 0755);
      p[i] = '/';
    }
  }
  ::mkdir(p.c_str(), 0755);
}

void SaveIndexCache(const base::String& path,
                    const char* backend,
                    const base::Vector<ArchiveEntry>& entries) {
  const size_t slash = path.find_last_of('/');
  if (slash != base::String::npos)
    MakeDirs(path.substr(0, slash));

  base::Vector<u8> buf;
  buf.reserve(entries.size() * 96);
  buf.insert(buf.end(), kIndexMagic, kIndexMagic + sizeof(kIndexMagic));
  const u32 name_len = static_cast<u32>(std::strlen(backend));
  Put32(buf, name_len);
  buf.insert(buf.end(), backend, backend + name_len);
  Put64(buf, entries.size());
  for (const auto& e : entries) {
    Put32(buf, static_cast<u32>(e.path.size()));
    buf.insert(buf.end(), e.path.begin(), e.path.end());
    Put64(buf, e.size);
    Put64(buf, e.packed_size);
    Put64(buf, e.data_offset);
    Put64(buf, e.extra);
    Put32(buf, e.method);
    Put32(buf, e.crc);
  }

  io::File f(base::String(path.c_str()), io::FileMode::kWrite);
  if (!f.IsOpen()) {
    BASE_LOGW("archive", "could not write index cache {}", path.c_str());
    return;
  }
  f.Write(buf.data(), buf.size());
}

// The console's /app0 is case-insensitive and titles rely on it: Demon's Souls
// dumps all-lowercase but opens mixed-case paths off its own command line, so
// exact-match loses files that are plainly there. Keys fold; the entry keeps
// its real name.
base::String FoldCase(base::String s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return s;
}

// Length of the leading directory run every entry shares, e.g. 15 for a set of
// paths all under "PPSA01342-app/". Dumps are archived with the game wrapped in
// one (sometimes two) such directories, and the guest expects /app0 to be the
// game root, not the wrapper.
size_t CommonWrapperPrefix(const base::Vector<ArchiveEntry>& entries) {
  if (entries.empty())
    return 0;
  size_t prefix = 0;
  for (int depth = 0; depth < 4; depth++) {
    const base::String& first = entries[0].path;
    const size_t slash = first.find('/', prefix);
    if (slash == base::String::npos)
      break;
    const size_t next = slash + 1;
    bool shared = true;
    for (const auto& e : entries) {
      if (e.path.size() <= next ||
          std::memcmp(e.path.data(), first.data(), next) != 0) {
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

}  // namespace

struct ArchiveImpl {
  base::UniquePointer<ArchiveBackend> backend;
  base::Vector<ArchiveEntry> entries;
  base::HashMap<base::String, ArchiveFilesystem::Node> files;  // case-folded
  base::Vector<base::String> guest_paths;  // as stored, for listings
  bool ok = false;

  // Whole decompressed entries, most recently used first. Small files are read
  // over and over (the guest reopens headers and manifests constantly) and
  // re-inflating them each time is what makes a container mount feel slow.
  struct Slot {
    u32 index;
    base::Vector<u8> data;
  };
  base::Mutex lock;
  base::Vector<Slot> lru;  // at most a few hundred: a scan beats an index
  u64 lru_bytes = 0;

  explicit ArchiveImpl(const base::String& path) {
    backend = OpenRarBackend(path);
    if (!backend)
      backend = OpenZipBackend(path);
    if (!backend) {
      BASE_LOGE("archive", "{} is not a container we can read", path.c_str());
      return;
    }

    io::File probe(path, io::FileMode::kRead);
    const u64 archive_size = probe.IsOpen() ? probe.GetSize() : 0;
    probe.Close();
    const base::String cache = IndexCachePath(path.c_str(), archive_size);

    bool from_cache = false;
    if (!kNoIndexCache && LoadIndexCache(cache, backend->Name(), entries)) {
      from_cache = true;
    } else if (!backend->Index(entries)) {
      BASE_LOGE("archive", "failed to index {}", path.c_str());
      return;
    }
    if (entries.empty()) {
      BASE_LOGE("archive", "{} holds no files", path.c_str());
      return;
    }
    if (!from_cache && !kNoIndexCache)
      SaveIndexCache(cache, backend->Name(), entries);

    const size_t strip = CommonWrapperPrefix(entries);
    guest_paths.reserve(entries.size());
    for (u32 i = 0; i < entries.size(); i++) {
      base::String path = "/" + entries[i].path.substr(strip);
      files.emplace(FoldCase(path),
                    ArchiveFilesystem::Node{entries[i].size, i});
      guest_paths.push_back(base::move(path));
    }
    ok = true;

    const base::String root =
        strip ? ", root " + entries[0].path.substr(0, strip) : base::String();
    BASE_LOGI("archive", "{} ({}): {} files{}{}", path.c_str(), backend->Name(),
              entries.size(), from_cache ? ", index cached" : "", root.c_str());
  }

  i64 Read(const ArchiveFilesystem::Node& node, void* buf, i64 off, i64 len) {
    if (off < 0 || len < 0 || node.index >= entries.size())
      return -1;
    if (static_cast<u64>(off) >= node.size)
      return 0;
    len = base::Min<i64>(len, static_cast<i64>(node.size - off));
    if (len == 0)
      return 0;

    const u64 small_max = u64(base::Max(0, (int)kSmallMb)) << 20;
    if (node.size > small_max)
      return backend->ExtractRange(entries[node.index], buf, off, len);

    base::LockGuard<base::Mutex> guard(lock);
    Slot* hit =
        lru.FindIf([&](const Slot& s) { return s.index == node.index; });
    if (!hit) {
      base::Vector<u8> data(static_cast<size_t>(node.size));
      const i64 got = backend->ExtractRange(entries[node.index], data.data(), 0,
                                            static_cast<i64>(node.size));
      if (got < 0)
        return -1;
      data.resize(static_cast<size_t>(got));
      lru_bytes += data.size();
      lru.insert(lru.begin(), Slot{node.index, base::move(data)});
      Trim();
      if (lru.empty() || lru.front().index != node.index) {
        // The entry alone blew the budget, so it was trimmed straight back out.
        return backend->ExtractRange(entries[node.index], buf, off, len);
      }
    } else if (hit != lru.begin()) {
      Slot slot = base::move(*hit);
      lru.erase(hit);
      lru.insert(lru.begin(), base::move(slot));
    }

    const base::Vector<u8>& data = lru.front().data;
    if (static_cast<u64>(off) >= data.size())
      return 0;
    len = base::Min<i64>(len, static_cast<i64>(data.size() - off));
    std::memcpy(buf, data.data() + off, static_cast<size_t>(len));
    return len;
  }

  // Directory path (folded, with trailing '/') -> immediate children. Built
  // once on the first listing: scanning all 223k paths per readdir made
  // directory enumeration quadratic.
  base::HashMap<base::String, base::Vector<ArchiveFilesystem::Child>> dirs;
  bool dirs_built = false;

  void BuildDirs() {
    if (dirs_built)
      return;
    dirs_built = true;
    base::HashMap<base::String, base::HashMap<base::String, bool>> seen;
    for (const auto& p : guest_paths) {
      // Register the file under its parent, and every ancestor under its own.
      size_t pos = 0;
      while (true) {
        const size_t slash = p.find('/', pos + 1);
        const base::String parent = FoldCase(p.substr(0, pos + 1));
        if (slash == base::String::npos) {
          seen[parent].emplace(p.substr(pos + 1), false);
          break;
        }
        seen[parent].emplace(p.substr(pos + 1, slash - pos - 1), true);
        pos = slash;
      }
    }
    for (auto& kv : seen) {
      auto& out = dirs[kv.first];
      out.reserve(kv.second.size());
      for (auto& child : kv.second)
        out.push_back({child.first, child.second});
    }
  }

  void Trim() {
    const u64 budget = u64(base::Max(0, (int)kCacheMb)) << 20;
    while (lru_bytes > budget && !lru.empty()) {
      lru_bytes -= lru.back().data.size();
      lru.pop_back();
    }
  }
};

ArchiveFilesystem::ArchiveFilesystem(const base::String& archive_path)
    : impl_(base::MakeUnique<ArchiveImpl>(archive_path)) {}
ArchiveFilesystem::~ArchiveFilesystem() = default;

bool ArchiveFilesystem::Valid() const {
  return impl_ && impl_->ok;
}

const ArchiveFilesystem::Node* ArchiveFilesystem::Find(const char* rel) const {
  if (!impl_ || !impl_->ok || !rel)
    return nullptr;
  auto it = impl_->files.find(FoldCase(rel));
  return it == impl_->files.end() ? nullptr : &it->second;
}

i64 ArchiveFilesystem::Read(const Node& node, void* buf, i64 off, i64 len) {
  return impl_ && impl_->ok ? impl_->Read(node, buf, off, len) : -1;
}

bool ArchiveFilesystem::List(const char* rel, base::Vector<Child>& out) {
  if (!impl_ || !impl_->ok)
    return false;
  base::String prefix(rel ? rel : "");
  while (!prefix.empty() && prefix.back() == '/')
    prefix.pop_back();
  prefix += "/";
  if (prefix[0] != '/')
    prefix.insert(0, 1, '/');

  base::LockGuard<base::Mutex> guard(impl_->lock);
  impl_->BuildDirs();
  auto it = impl_->dirs.find(FoldCase(prefix));
  if (it == impl_->dirs.end())
    return false;
  out.insert(out.end(), it->second.begin(), it->second.end());
  return !out.empty();
}

void ArchiveFilesystem::Paths(base::Vector<base::String>& out) const {
  if (!impl_)
    return;
  out.insert(out.end(), impl_->guest_paths.begin(), impl_->guest_paths.end());
}

const char* ArchiveFilesystem::BackendName() const {
  return impl_ && impl_->backend ? impl_->backend->Name() : "";
}

bool IsArchivePath(const char* path) {
  if (!path)
    return false;
  static const char* kExts[] = {".rar", ".zip"};
  const size_t n = std::strlen(path);
  for (const char* ext : kExts) {
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

}  // namespace formats
