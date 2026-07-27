/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <base/containers/vector.h>

#include "vfs.h"

namespace krnl::vfs {
struct mountPoint {
  base::String guest;
  base::String host;                        // host mount
  std::shared_ptr<VirtualProvider> provider; // virtual mount (else null)
  bool writable = false;                     // host mount opened for writing
};

static base::Vector<mountPoint> g_mounts;
static std::mutex g_mountsMutex;

void mount(const char *guest, const char *host) {
  std::lock_guard<std::mutex> lock(g_mountsMutex);
  g_mounts.push_back({base::String(guest), base::String(host), nullptr, false});
}

// Create hostDir and each parent (mkdir -p).
static void makeHostDirs(const char *hostDir) {
  std::string p(hostDir);
  for (size_t i = 1; i < p.size(); i++) {
    if (p[i] == '/') {
      p[i] = 0;
      ::mkdir(p.c_str(), 0755);
      p[i] = '/';
    }
  }
  ::mkdir(p.c_str(), 0755);
}

void mountWritable(const char *guest, const char *host) {
  makeHostDirs(host);
  std::lock_guard<std::mutex> lock(g_mountsMutex);
  g_mounts.push_back({base::String(guest), base::String(host), nullptr, true});
}

void unmount(const char *guest) {
  std::lock_guard<std::mutex> lock(g_mountsMutex);
  for (mem_size i = g_mounts.size(); i > 0; --i) {
    if (g_mounts[i - 1].guest == guest) {
      g_mounts.erase(g_mounts.begin() + i - 1);
      return;
    }
  }
}

void mountVirtual(const char *guest, std::shared_ptr<VirtualProvider> provider) {
  std::lock_guard<std::mutex> lock(g_mountsMutex);
  g_mounts.push_back(
      {base::String(guest), base::String(), std::move(provider)});
}

// The process working directory is /app0 (sys_getcwd reports it) and the guest
// separator is '/'. Bethesda's engine opens plain relative paths ("Settings")
// and Windows-style ones, so normalise both before the mount lookup. relPrefix
// picks where a bare relative path lands: /app0 for reads (the game image),
// /download0 for writes (relative creates against a read-only mount could never
// succeed -- see resolveWritable).
static base::String normalizePath(const char *path, const char *relPrefix = "/app0/") {
  base::String out;
  if (path && path[0] != '/')
    out += relPrefix;
  for (const char *p = path; p && *p; p++)
    out += (*p == '\\') ? '/' : *p;
  return out;
}

// Longest matching mount. prefixLen is the matched length.
static bool findMount(const char *path, bool hostOnly, mountPoint &result,
                      size_t &prefixLen) {
  std::lock_guard<std::mutex> lock(g_mountsMutex);
  const mountPoint *best = nullptr;
  size_t bestLen = 0;
  for (auto &m : g_mounts) {
    if (hostOnly && m.provider)
      continue;
    size_t len = m.guest.length();
    if (std::strncmp(path, m.guest.c_str(), len) == 0 && len >= bestLen) {
      best = &m;
      bestLen = len;
    }
  }
  prefixLen = bestLen;
  if (!best)
    return false;
  result.guest = best->guest;
  result.host = best->host;
  result.provider = best->provider;
  result.writable = best->writable;
  return true;
}

base::String resolve(const char *path) {
  if (!path)
    return {};

  size_t bestLen = 0;
  mountPoint best;
  if (!findMount(path, true, best, bestLen))
    return {};

  base::String out(best.host);
  const char *rest = path + bestLen;
  if (*rest && *rest != '/')
    out += "/";
  out += rest;
  return out;
}

namespace {
// Adapts a VirtualFile to utl::fileBase so it can flow through fileDevice and
// the rest of the file machinery like a real file. Read-only.
struct PfsFileStream final : utl::fileBase {
  std::unique_ptr<VirtualFile> vf;
  uint64_t pos = 0;

  explicit PfsFileStream(std::unique_ptr<VirtualFile> v) : vf(std::move(v)) {}

  uint64_t Read(void *buf, size_t size) override {
    int64_t n = vf->read(buf, static_cast<int64_t>(pos),
                         static_cast<int64_t>(size));
    if (n <= 0)
      return 0;
    // DELTA_PREAD_ZEROPAD: return the FULL requested length even when the read
    // was clamped short at EOF. The provider's read chain already zero-fills the
    // whole destination buffer, so the tail bytes are valid (zeros). SotC's world
    // container is loaded by libSceFios2's whole-file sceFiosFHRead as ONE async
    // op; FIOS2 block-pads its final chunk past the member's EOF and treats the
    // resulting short read as an OP FAILURE -> actualCount 0 -> the loader (which
    // ignores the read error) commits payload=0,err=0 and RETRIES FOREVER (the
    // post-LoadInitialWorld freeze). Reporting the full length lets that op
    // complete. Off by default so other titles keep true short-at-EOF semantics.
    static const bool zeroPad = std::getenv("DELTA_PREAD_ZEROPAD") != nullptr;
    // DELTA_SHORTREAD: log every clamped-short provider read (raw n < requested),
    // which is exactly the FIOS2-op-failure trigger, regardless of zeroPad. Cheap:
    // only fires on the anomaly, not on full reads.
    if (static_cast<uint64_t>(n) < size && std::getenv("DELTA_SHORTREAD"))
      std::fprintf(stderr, "[shortread] pos=%llu req=%zu got=%lld%s\n",
                   (unsigned long long)pos, size, (long long)n,
                   zeroPad ? " (zeropadded->full)" : "");
    uint64_t reported = zeroPad ? size : static_cast<uint64_t>(n);
    pos += reported;
    return reported;
  }
  uint64_t Write(const void *, size_t) override { return 0; }
  uint64_t Seek(int64_t off, utl::seekMode whence) override {
    int64_t np = whence == utl::seekMode::seek_set
                     ? off
                     : whence == utl::seekMode::seek_cur
                           ? static_cast<int64_t>(pos) + off
                           : static_cast<int64_t>(vf->size()) + off;
    if (np < 0)
      return static_cast<uint64_t>(-1);
    pos = static_cast<uint64_t>(np);
    return pos;
  }
  uint64_t Tell() override { return pos; }
  uint64_t GetSize() override { return static_cast<uint64_t>(vf->size()); }
  utl::native_handle GetNativeHandle() override { return nullptr; }
  bool IsOpen() override { return true; }
};

// Append the post-prefix remainder onto a host directory, like resolve().
base::String joinHost(const base::String &host, const char *rest) {
  base::String out(host);
  if (*rest && *rest != '/')
    out += "/";
  out += rest;
  return out;
}

// /app0 is case-insensitive on the console: Skyrim's disc image holds "data/"
// and "Skyrim_de.ini", and the engine opens "/app0/Data/..." and "Skyrim.INI".
// When the exact spelling misses, walk the path a component at a time and take
// the unique case-insensitive match. Directory listings are memoised: a title
// streaming thousands of assets would otherwise rescan the same directory on
// every open.
std::mutex g_caseMutex;
std::map<std::string, std::map<std::string, std::string>> g_caseIndex;

// Returns the on-disc spelling of `name` in `dir`, or null. Caller holds no
// lock; the index is shared across the title's streaming threads.
const std::string *lookupCaseInsensitive(const std::string &dir,
                                         const std::string &name) {
  std::lock_guard<std::mutex> lk(g_caseMutex);
  auto it = g_caseIndex.find(dir);
  if (it == g_caseIndex.end()) {
    std::map<std::string, std::string> index;
    if (DIR *d = opendir(dir.c_str())) {
      while (dirent *e = readdir(d)) {
        std::string lower(e->d_name);
        for (auto &c : lower)
          c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        index.emplace(std::move(lower), e->d_name);
      }
      closedir(d);
    }
    it = g_caseIndex.emplace(dir, std::move(index)).first;
  }
  std::string lower(name);
  for (auto &c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  auto hit = it->second.find(lower);
  return hit == it->second.end() ? nullptr : &hit->second;
}

base::String fixHostCase(const base::String &host) {
  struct stat st;
  if (::stat(host.c_str(), &st) == 0)
    return host;

  std::string path(host.c_str());
  size_t pos = path.find('/', 1);
  std::string built = pos == std::string::npos ? path : path.substr(0, pos);
  while (pos != std::string::npos) {
    size_t next = path.find('/', pos + 1);
    std::string comp = path.substr(pos + 1, next == std::string::npos
                                                ? std::string::npos
                                                : next - pos - 1);
    std::string cand = built + "/" + comp;
    if (::stat(cand.c_str(), &st) != 0) {
      const std::string *real = lookupCaseInsensitive(built, comp);
      if (!real)
        return host;  // no match: let the caller report the original miss
      cand = built + "/" + *real;
    }
    built = std::move(cand);
    pos = next;
  }
  return base::String(built.c_str());
}
} // namespace

utl::File openRead(const char *path) {
  if (!path)
    return utl::File();

  if (std::getenv("DELTA_OPEN_TRACE"))
    std::fprintf(stderr, "[open] %s\n", path);

  // DELTA_VFS_HIDE=<substr>[,<substr>]: report a matching path as missing, to
  // test whether an optional asset (an intro movie, a DLC list) is what a boot
  // path chokes on.
  if (const char *hide = std::getenv("DELTA_VFS_HIDE")) {
    for (const char *p = hide; *p;) {
      const char *sep = std::strchr(p, ',');
      std::string pat(p, sep ? size_t(sep - p) : std::strlen(p));
      if (!pat.empty() && std::strstr(path, pat.c_str()))
        return utl::File();
      p = sep ? sep + 1 : p + pat.size();
    }
  }

  base::String norm = normalizePath(path);
  path = norm.c_str();

  size_t len = 0;
  mountPoint m;
  if (!findMount(path, false, m, len))
    return utl::File();

  const char *rest = path + len;
  if (m.provider) {
    auto vf = m.provider->open(rest);
    if (!vf)
      return utl::File();
    utl::File out(base::MakeUnique<PfsFileStream>(std::move(vf)));
    // DELTA_OPEN_TRACE: also report the size we hand back. A file that opens OK
    // but reports size 0 (e.g. a >4 GiB member whose size truncated) makes the
    // resource loader hang forever with {payload=0, err=0} (SotC world container).
    if (std::getenv("DELTA_OPEN_TRACE"))
      std::fprintf(stderr, "[opensz] %s -> size=%llu (provider)\n", path,
                   (unsigned long long)out.GetSize());
    return out;
  }

  // A raw console app dump keeps each executable twice: the encrypted SELF under
  // its real name and the decrypted ELF beside it as "<name>.esbak". Prefer the
  // decrypted one; we have no SELF crypto.
  base::String host = fixHostCase(joinHost(m.host, rest));
  utl::File esbak(host + ".esbak", utl::fileMode::read);
  if (esbak.Exists() && esbak.IsOpen())
    return esbak;

  utl::File f(host, utl::fileMode::read);
  if (!f.Exists() || !f.IsOpen())
    return utl::File();
  if (std::getenv("DELTA_OPEN_TRACE"))
    std::fprintf(stderr, "[opensz] %s -> size=%llu (host)\n", path,
                 (unsigned long long)f.GetSize());
  return f;
}

base::String resolveWritable(const char *path) {
  if (!path)
    return {};
  // A bare relative create ("Settings") can't mean the read-only game image:
  // route it to the title's writable data volume instead of /app0, where it
  // could never succeed.
  base::String norm = normalizePath(path, "/download0/");
  path = norm.c_str();
  size_t len = 0;
  mountPoint m;
  if (!findMount(path, false, m, len) || m.provider || !m.writable)
    return {};
  return joinHost(m.host, path + len);
}

bool makeDir(const char *path) {
  base::String host = resolveWritable(path);
  if (host.empty())
    return false;
  makeHostDirs(host.c_str());
  return true;
}

bool removeFile(const char *path) {
  base::String host = resolveWritable(path);
  if (host.empty())
    return false;
  return std::remove(host.c_str()) == 0;
}

bool stat(const char *path, int64_t &size, bool &isDir) {
  isDir = false;
  if (!path)
    return false;

  if (path[0] == '/' && path[1] == '\0') {
    size = 0;
    isDir = true;
    return true;
  }

  base::String norm = normalizePath(path);
  path = norm.c_str();

  size_t len = 0;
  mountPoint m;
  if (!findMount(path, false, m, len))
    return false;

  const char *rest = path + len;
  if (m.provider) {
    bool ok = m.provider->stat(rest, size);
    if (std::getenv("DELTA_OPEN_TRACE"))
      std::fprintf(stderr, "[stat] %s -> %s size=%lld\n", path,
                   ok ? "ok" : "MISS", (long long)size);
    return ok;
  }

  base::String host = fixHostCase(joinHost(m.host, rest));
  struct ::stat st;
  if (::stat(host.c_str(), &st) != 0)
    return false;
  isDir = S_ISDIR(st.st_mode);
  size = static_cast<int64_t>(st.st_size);
  return true;
}

static std::string g_titleId;
void setTitleId(const std::string &id) { g_titleId = id; }
const std::string &titleId() { return g_titleId; }

static std::map<std::string, std::vector<uint8_t>> g_fileCache;
void cacheFile(const std::string &key, std::vector<uint8_t> data) {
  g_fileCache[key] = std::move(data);
}
const std::vector<uint8_t> *getCachedFile(const char *key) {
  auto it = g_fileCache.find(key);
  return it == g_fileCache.end() ? nullptr : &it->second;
}

bool listDir(const char *path, std::vector<DirEntry> &out) {
  if (!path)
    return false;

  base::String norm = normalizePath(path);
  path = norm.c_str();
  size_t len = 0;
  mountPoint m;
  if (!findMount(path, false, m, len))
    return false;

  const char *rest = path + len;
  if (m.provider)
    return m.provider->list(rest, out);

  // Host mount: enumerate the host directory.
  base::String hostDir = fixHostCase(joinHost(m.host, rest));
  DIR *d = opendir(hostDir.c_str());
  if (!d)
    return false;
  while (dirent *e = readdir(d)) {
    if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0)
      continue;
    out.push_back({e->d_name, e->d_type == DT_DIR});
  }
  closedir(d);
  return true;
}
} // namespace krnl::vfs
