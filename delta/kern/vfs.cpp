/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstring>
#include <dirent.h>

#include <base/containers/vector.h>

#include "vfs.h"

namespace krnl::vfs {
struct mountPoint {
  base::String guest;
  base::String host;                        // host mount
  std::shared_ptr<VirtualProvider> provider; // virtual mount (else null)
};

static base::Vector<mountPoint> g_mounts;

void mount(const char *guest, const char *host) {
  g_mounts.push_back({base::String(guest), base::String(host), nullptr});
}

void mountVirtual(const char *guest, std::shared_ptr<VirtualProvider> provider) {
  g_mounts.push_back(
      {base::String(guest), base::String(), std::move(provider)});
}

// Longest matching mount (host or virtual). prefixLen is the matched length.
static const mountPoint *findMount(const char *path, size_t &prefixLen) {
  const mountPoint *best = nullptr;
  size_t bestLen = 0;
  for (auto &m : g_mounts) {
    size_t len = m.guest.length();
    if (std::strncmp(path, m.guest.c_str(), len) == 0 && len >= bestLen) {
      best = &m;
      bestLen = len;
    }
  }
  prefixLen = bestLen;
  return best;
}

base::String resolve(const char *path) {
  if (!path)
    return {};

  const mountPoint *best = nullptr;
  size_t bestLen = 0;
  for (auto &m : g_mounts) {
    if (m.provider)
      continue; // host-only
    size_t len = m.guest.length();
    if (std::strncmp(path, m.guest.c_str(), len) == 0 && len >= bestLen) {
      best = &m;
      bestLen = len;
    }
  }
  if (!best)
    return {};

  base::String out(best->host);
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
    pos += static_cast<uint64_t>(n);
    return static_cast<uint64_t>(n);
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
} // namespace

utl::File openRead(const char *path) {
  if (!path)
    return utl::File();

  size_t len = 0;
  const mountPoint *m = findMount(path, len);
  if (!m)
    return utl::File();

  const char *rest = path + len;
  if (m->provider) {
    auto vf = m->provider->open(rest);
    if (!vf)
      return utl::File();
    return utl::File(base::MakeUnique<PfsFileStream>(std::move(vf)));
  }

  utl::File f(joinHost(m->host, rest), utl::fileMode::read);
  if (!f.Exists() || !f.IsOpen())
    return utl::File();
  return f;
}

bool stat(const char *path, int64_t &size, bool &isDir) {
  isDir = false;
  if (!path)
    return false;

  size_t len = 0;
  const mountPoint *m = findMount(path, len);
  if (!m)
    return false;

  const char *rest = path + len;
  if (m->provider)
    return m->provider->stat(rest, size);

  utl::File f(joinHost(m->host, rest), utl::fileMode::read);
  if (!f.Exists() || !f.IsOpen())
    return false;
  size = static_cast<int64_t>(f.GetSize());
  return true;
}

bool listDir(const char *path, std::vector<DirEntry> &out) {
  if (!path)
    return false;

  size_t len = 0;
  const mountPoint *m = findMount(path, len);
  if (!m)
    return false;

  const char *rest = path + len;
  if (m->provider)
    return m->provider->list(rest, out);

  // Host mount: enumerate the host directory.
  base::String hostDir = joinHost(m->host, rest);
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
