/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"
#include "guest/session.h"

#include "base/containers/vector.h"
#include "base/logging.h"

#include "base/containers/map.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kShortRead, "DELTA_SHORTREAD", false);
DELTA_OPTION(const char*, kVfsHide, "DELTA_VFS_HIDE", nullptr);
DELTA_OPTION(bool, kOpenTrace, "DELTA_OPEN_TRACE", false);
DELTA_OPTION(bool, kPreadZeropad, "DELTA_PREAD_ZEROPAD", false);
DELTA_OPTION(const char*, kVfsOverlay, "DELTA_VFS_OVERLAY", nullptr);
}  // namespace

namespace kern::vfs {
struct MountPoint {
  base::String guest;
  base::String host;                              // host mount
  base::SharedPointer<VirtualProvider> provider;  // virtual mount (else null)
  bool writable = false;  // host mount opened for writing
};

static base::Vector<MountPoint> g_mounts;
static base::Mutex g_mounts_mutex;

void Mount(const char* guest, const char* host) {
  base::LockGuard<base::Mutex> lock(g_mounts_mutex);
  g_mounts.push_back({base::String(guest), base::String(host), nullptr, false});
}

// Create hostDir and each parent (mkdir -p).
static void MakeHostDirs(const char* host_dir) {
  base::String p(host_dir);
  for (size_t i = 1; i < p.size(); i++) {
    if (p[i] == '/') {
      p[i] = 0;
      ::mkdir(p.c_str(), 0755);
      p[i] = '/';
    }
  }
  ::mkdir(p.c_str(), 0755);
}

void MountWritable(const char* guest, const char* host) {
  MakeHostDirs(host);
  base::LockGuard<base::Mutex> lock(g_mounts_mutex);
  g_mounts.push_back({base::String(guest), base::String(host), nullptr, true});
}

void Unmount(const char* guest) {
  base::LockGuard<base::Mutex> lock(g_mounts_mutex);
  for (mem_size i = g_mounts.size(); i > 0; --i) {
    if (g_mounts[i - 1].guest == guest) {
      g_mounts.erase(g_mounts.begin() + i - 1);
      return;
    }
  }
}

void MountVirtual(const char* guest,
                  base::SharedPointer<VirtualProvider> provider) {
  base::LockGuard<base::Mutex> lock(g_mounts_mutex);
  g_mounts.push_back(
      {base::String(guest), base::String(), base::move(provider)});
}

// The process working directory is /app0 (sys_getcwd reports it) and the guest
// separator is '/'. Bethesda's engine opens plain relative paths ("Settings")
// and Windows-style ones, so normalise both before the mount lookup. relPrefix
// picks where a bare relative path lands: /app0 for reads (the game image),
// /download0 for writes (relative creates against a read-only mount could never
// succeed; see ResolveWritable).
static base::String NormalizePath(const char* path,
                                  const char* rel_prefix = "/app0/") {
  base::String out;
  if (path && path[0] != '/')
    out += rel_prefix;
  for (const char* p = path; p && *p; p++)
    out += (*p == '\\') ? '/' : *p;
  return out;
}

// Longest matching mount. prefixLen is the matched length.
static bool FindMount(const char* path,
                      bool host_only,
                      MountPoint& result,
                      size_t& prefix_len) {
  base::LockGuard<base::Mutex> lock(g_mounts_mutex);
  const MountPoint* best = nullptr;
  size_t best_len = 0;
  for (auto& m : g_mounts) {
    if (host_only && m.provider)
      continue;
    size_t len = m.guest.length();
    if (std::strncmp(path, m.guest.c_str(), len) == 0 && len >= best_len) {
      best = &m;
      best_len = len;
    }
  }
  prefix_len = best_len;
  if (!best)
    return false;
  result.guest = best->guest;
  result.host = best->host;
  result.provider = best->provider;
  result.writable = best->writable;
  return true;
}

base::String Resolve(const char* path) {
  if (!path)
    return {};

  size_t best_len = 0;
  MountPoint best;
  if (!FindMount(path, true, best, best_len))
    return {};

  base::String out(best.host);
  const char* rest = path + best_len;
  if (*rest && *rest != '/')
    out += "/";
  out += rest;
  return out;
}

namespace {
// Adapts a VirtualFile to io::FileBase so it can flow through FileDevice and
// the rest of the file machinery like a real file. Read-only.
struct PfsFileStream final : io::FileBase {
  base::UniquePointer<VirtualFile> vf;
  u64 pos = 0;

  explicit PfsFileStream(base::UniquePointer<VirtualFile> v)
      : vf(base::move(v)) {}

  u64 Read(void* buf, size_t size) override {
    i64 n = vf->Read(buf, static_cast<i64>(pos), static_cast<i64>(size));
    if (n <= 0)
      return 0;
    // DELTA_PREAD_ZEROPAD: return the FULL requested length even when the read
    // was clamped short at EOF. The provider's read chain already zero-fills
    // the whole destination buffer, so the tail bytes are valid (zeros). SotC's
    // world container is loaded by libSceFios2's whole-file sceFiosFHRead as
    // ONE async op; FIOS2 block-pads its final chunk past the member's EOF and
    // treats the resulting short read as an OP FAILURE, which makes the loader
    // commit payload=0,err=0 and RETRIES FOREVER (the post-LoadInitialWorld
    // freeze). Reporting the full length lets that op complete. Off by default
    // so other titles keep true short-at-EOF semantics. DELTA_SHORTREAD: log
    // every clamped-short provider read (raw n < requested), which is exactly
    // the FIOS2-op-failure trigger, regardless of zeroPad. Cheap: only fires on
    // the anomaly, not on full reads.
    if (static_cast<u64>(n) < size && kShortRead)
      BASE_LOGI("shortread", "pos={} req={} got={}{}", (unsigned long long)pos,
                size, (long long)n, kPreadZeropad ? " (zeropadded->full)" : "");
    u64 reported = kPreadZeropad ? size : static_cast<u64>(n);
    pos += reported;
    return reported;
  }
  u64 Write(const void*, size_t) override { return 0; }
  u64 Seek(i64 off, io::SeekMode whence) override {
    i64 np = whence == io::SeekMode::kSeekSet ? off
             : whence == io::SeekMode::kSeekCur
                 ? static_cast<i64>(pos) + off
                 : static_cast<i64>(vf->Size()) + off;
    if (np < 0)
      return static_cast<u64>(-1);
    pos = static_cast<u64>(np);
    return pos;
  }
  u64 Tell() override { return pos; }
  u64 GetSize() override { return static_cast<u64>(vf->Size()); }
  io::NativeHandle GetNativeHandle() override { return nullptr; }
  bool IsOpen() override { return true; }
};

// Append the post-prefix remainder onto a host directory, like resolve().
base::String JoinHost(const base::String& host, const char* rest) {
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
base::Mutex g_case_mutex;
base::Map<base::String, base::Map<base::String, base::String>> g_case_index;

// Returns the on-disc spelling of `name` in `dir`, or null. Caller holds no
// lock; the index is shared across the title's streaming threads.
const base::String* LookupCaseInsensitive(const base::String& dir,
                                          const base::String& name) {
  base::LockGuard<base::Mutex> lk(g_case_mutex);
  auto it = g_case_index.find(dir);
  if (it == g_case_index.end()) {
    base::Map<base::String, base::String> index;
    if (DIR* d = opendir(dir.c_str())) {
      while (dirent* e = readdir(d)) {
        base::String lower(e->d_name);
        for (auto& c : lower)
          c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        index.emplace(base::move(lower), e->d_name);
      }
      closedir(d);
    }
    it = g_case_index.emplace(dir, base::move(index)).first;
  }
  base::String lower(name);
  for (auto& c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  auto hit = it->second.find(lower);
  return hit == it->second.end() ? nullptr : &hit->second;
}

base::String FixHostCase(const base::String& host) {
  struct stat st;
  if (::stat(host.c_str(), &st) == 0)
    return host;

  base::String path(host.c_str());
  size_t pos = path.find('/', 1);
  base::String built = pos == base::String::npos ? path : path.substr(0, pos);
  while (pos != base::String::npos) {
    size_t next = path.find('/', pos + 1);
    base::String comp =
        path.substr(pos + 1, next == base::String::npos ? base::String::npos
                                                        : next - pos - 1);
    base::String cand = built + "/" + comp;
    if (::stat(cand.c_str(), &st) != 0) {
      const base::String* real = LookupCaseInsensitive(built, comp);
      if (!real)
        return host;  // no match: let the caller report the original miss
      cand = built + "/" + *real;
    }
    built = base::move(cand);
    pos = next;
  }
  return base::String(built.c_str());
}
}  // namespace

// DELTA_VFS_OVERLAY=<hostdir>: a host tree searched before the real mounts, so
// a single file can be substituted or supplied without repacking the pkg (the
// title's own config hooks: SotC reads /app0/savedcmdargs.txt at boot, and they
// live inside a read-only PFS image otherwise).
static base::String OverlayPath(const char* guest_path) {
  const char* dir = kVfsOverlay;
  if (!dir || !*dir || !guest_path || guest_path[0] != '/')
    return base::String();
  base::String p(dir);
  if (!p.empty() && p[p.size() - 1] == '/')
    p = p.substr(0, p.size() - 1);
  p += guest_path;
  return p;
}

io::File OpenRead(const char* path) {
  if (!path)
    return io::File();

  if (kOpenTrace)
    BASE_LOGI("open", "{}", path);

  // DELTA_VFS_HIDE=<substr>[,<substr>]: report a matching path as missing, to
  // test whether an optional asset (an intro movie, a DLC list) is what a boot
  // path chokes on.
  if (const char* hide = kVfsHide) {
    for (const char* p = hide; *p;) {
      const char* sep = std::strchr(p, ',');
      base::String pat(p, sep ? size_t(sep - p) : std::strlen(p));
      if (!pat.empty() && std::strstr(path, pat.c_str()))
        return io::File();
      p = sep ? sep + 1 : p + pat.size();
    }
  }

  base::String norm = NormalizePath(path);
  path = norm.c_str();

  if (base::String ov = OverlayPath(path); !ov.empty()) {
    io::File f(ov);
    if (f.IsOpen()) {
      if (kOpenTrace)
        BASE_LOGI("open", "  -> overlay {}", ov.c_str());
      return f;
    }
  }

  size_t len = 0;
  MountPoint m;
  if (!FindMount(path, false, m, len))
    return io::File();

  const char* rest = path + len;
  if (m.provider) {
    auto vf = m.provider->Open(rest);
    if (!vf)
      return io::File();
    io::File out(base::MakeUnique<PfsFileStream>(base::move(vf)));
    // DELTA_OPEN_TRACE: also report the size we hand back. A file that opens OK
    // but reports size 0 (e.g. a >4 GiB member whose size truncated) makes the
    // resource loader hang forever with {payload=0, err=0} (SotC world
    // container).
    if (kOpenTrace)
      BASE_LOGI("opensz", "{} -> size={} (provider)", path,
                (unsigned long long)out.GetSize());
    return out;
  }

  // A raw console app dump keeps each executable twice: the encrypted SELF
  // under its real name and the decrypted ELF beside it as "<name>.esbak".
  // Prefer the decrypted one; we have no SELF crypto.
  base::String host = FixHostCase(JoinHost(m.host, rest));
  io::File esbak(host + ".esbak", io::FileMode::kRead);
  if (esbak.Exists() && esbak.IsOpen())
    return esbak;

  io::File f(host, io::FileMode::kRead);
  if (!f.Exists() || !f.IsOpen())
    return io::File();
  if (kOpenTrace)
    BASE_LOGI("opensz", "{} -> size={} (host)", path,
              (unsigned long long)f.GetSize());
  return f;
}

base::String ResolveWritable(const char* path) {
  if (!path)
    return {};
  // A bare relative create ("Settings") can't mean the read-only game image:
  // route it to the title's writable data volume instead of /app0, where it
  // could never succeed.
  base::String norm = NormalizePath(path, "/download0/");
  path = norm.c_str();
  size_t len = 0;
  MountPoint m;
  if (!FindMount(path, false, m, len) || m.provider || !m.writable)
    return {};
  return JoinHost(m.host, path + len);
}

bool MakeDir(const char* path) {
  base::String host = ResolveWritable(path);
  if (host.empty())
    return false;
  MakeHostDirs(host.c_str());
  return true;
}

bool RemoveFile(const char* path) {
  base::String host = ResolveWritable(path);
  if (host.empty())
    return false;
  return std::remove(host.c_str()) == 0;
}

bool Stat(const char* path, i64& size, bool& is_dir) {
  is_dir = false;
  if (!path)
    return false;

  if (path[0] == '/' && path[1] == '\0') {
    size = 0;
    is_dir = true;
    return true;
  }

  base::String norm = NormalizePath(path);
  path = norm.c_str();

  if (base::String ov = OverlayPath(path); !ov.empty()) {
    io::File f(ov);
    if (f.IsOpen()) {
      size = static_cast<i64>(f.GetSize());
      return true;
    }
  }

  size_t len = 0;
  MountPoint m;
  if (!FindMount(path, false, m, len))
    return false;

  const char* rest = path + len;
  if (m.provider) {
    bool ok = m.provider->Stat(rest, size);
    if (!ok && m.provider->IsDir(rest)) {
      ok = is_dir = true;
      size = 0;
    }
    if (kOpenTrace)
      BASE_LOGI("stat", "{} -> {} size={} dir={}", path, ok ? "ok" : "MISS",
                (long long)size, (int)is_dir);
    return ok;
  }

  base::String host = FixHostCase(JoinHost(m.host, rest));
  struct ::stat st;
  if (::stat(host.c_str(), &st) != 0)
    return false;
  is_dir = S_ISDIR(st.st_mode);
  size = static_cast<i64>(st.st_size);
  return true;
}

static base::String g_title_id;
void SetTitleId(const base::String& id) {
  g_title_id = id;
}
const base::String& TitleId() {
  return g_title_id;
}

static base::Map<base::String, base::Vector<u8>> g_file_cache;
void CacheFile(const base::String& key, base::Vector<u8> data) {
  g_file_cache[key] = base::move(data);
}
const base::Vector<u8>* GetCachedFile(const char* key) {
  auto it = g_file_cache.find(key);
  return it == g_file_cache.end() ? nullptr : &it->second;
}

bool ListDir(const char* path, base::Vector<DirEntry>& out) {
  if (!path)
    return false;

  base::String norm = NormalizePath(path);
  path = norm.c_str();

  // The sandbox root is not backed by a host directory: it is the mount table.
  // libkernel opens "/" and walks its entries by d_reclen looking for a name,
  // so an empty/failed listing leaves it spinning on a zero-length record.
  if (std::strcmp(path, "/") == 0) {
    base::LockGuard<base::Mutex> lock(g_mounts_mutex);
    for (auto& mp : g_mounts) {
      const char* g = mp.guest.c_str();
      if (*g != '/')
        continue;
      const char* end = std::strchr(g + 1, '/');
      base::String top(g + 1, end ? end - (g + 1) : std::strlen(g + 1));
      if (top.empty())
        continue;
      bool dup = false;
      for (auto& e : out)
        dup = dup || e.name == top;
      if (!dup)
        out.push_back({top, true});
    }
    out.push_back({"dev", true});
    return true;
  }

  size_t len = 0;
  MountPoint m;
  if (!FindMount(path, false, m, len))
    return false;

  const char* rest = path + len;
  if (m.provider)
    return m.provider->List(rest, out);

  // Host mount: enumerate the host directory.
  base::String host_dir = FixHostCase(JoinHost(m.host, rest));
  DIR* d = opendir(host_dir.c_str());
  if (!d)
    return false;
  while (dirent* e = readdir(d)) {
    if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0)
      continue;
    out.push_back({e->d_name, e->d_type == DT_DIR});
  }
  closedir(d);
  return true;
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_mounts);
  guest::ResetResource(g_case_index);
  guest::ResetResource(g_file_cache);
  guest::ResetResource(g_title_id);
});
}  // namespace

}  // namespace kern::vfs
