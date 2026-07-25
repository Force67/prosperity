/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "dcore.h"
#include <logger/logger.h>
#include <utl/file.h>

#include <gfx/gfx.h>
#include <kern/vfs.h>

#include "formats/pkg_object.h"
#include "formats/pup_object.h"
#include "formats/ufs2_object.h"

deltaCore::deltaCore() = default;
deltaCore::~deltaCore() = default;

bool deltaCore::init() {
  LOG_INFO("Initializing deltaCore " rsc_copyright);
  return true;
}

namespace {
// Minimal param.sfo reader: return the string value of `key` (e.g. "TITLE_ID"),
// or "" if absent. The SFO is a small flat table; all offsets are bounds-checked.
std::string sfoGet(const uint8_t *d, size_t n, const char *key) {
  if (n < 20)
    return {};
  auto rd16 = [&](size_t o) -> uint16_t {
    return o + 2 <= n ? uint16_t(d[o] | (d[o + 1] << 8)) : 0;
  };
  auto rd32 = [&](size_t o) -> uint32_t {
    return o + 4 <= n ? uint32_t(d[o]) | (uint32_t(d[o + 1]) << 8) |
                            (uint32_t(d[o + 2]) << 16) | (uint32_t(d[o + 3]) << 24)
                      : 0;
  };
  if (rd32(0) != 0x46535000u) // "\0PSF"
    return {};
  uint32_t keyStart = rd32(8), dataStart = rd32(12), count = rd32(16);
  size_t klen = std::strlen(key);
  for (uint32_t i = 0, idx = 20; i < count; i++, idx += 16) {
    if (idx + 16 > n)
      break;
    size_t kpos = size_t(keyStart) + rd16(idx);
    if (kpos + klen + 1 > n)
      continue;
    if (std::memcmp(d + kpos, key, klen) != 0 || d[kpos + klen] != '\0')
      continue;
    size_t dpos = size_t(dataStart) + rd32(idx + 12);
    if (dpos >= n)
      return {};
    size_t avail = n - dpos, len = rd32(idx + 4);
    std::string s(reinterpret_cast<const char *>(d + dpos),
                  len < avail ? len : avail);
    while (!s.empty() && s.back() == '\0')
      s.pop_back();
    return s;
  }
  return {};
}

// PS5 titles carry sce_sys/param.json instead of the PS4 param.sfo. Pull one
// top-level string value out of it (flat file, no nesting on the keys we want).
std::string jsonGetString(const std::string &js, const char *key) {
  std::string pat = std::string("\"") + key + "\"";
  size_t k = js.find(pat);
  if (k == std::string::npos)
    return {};
  size_t colon = js.find(':', k + pat.size());
  if (colon == std::string::npos)
    return {};
  size_t open = js.find('"', colon);
  size_t close = open == std::string::npos ? open : js.find('"', open + 1);
  if (close == std::string::npos)
    return {};
  return js.substr(open + 1, close - open - 1);
}

// Bridges a PkgFilesystem into the kernel VFS as an on-demand virtual mount.
class PkgProvider : public krnl::vfs::VirtualProvider {
public:
  explicit PkgProvider(const base::String &path) : fs_(path) {
    if (const char *sub = std::getenv("DELTA_PKG_LS")) {
      std::vector<std::string> all;
      fs_.paths(all);
      for (const auto &p : all)
        if (sub[0] == '1' || p.find(sub) != std::string::npos) {
          const auto *n = fs_.find(p.c_str());
          std::fprintf(stderr, "[pkg] %12lld  %s\n",
                       n ? (long long)n->size : -1LL, p.c_str());
        }
    }
    if (const char *wantEnv = std::getenv("DELTA_PKG_DUMP")) {
      std::string list(wantEnv);
      size_t pos = 0;
      while (pos <= list.size()) {
        size_t comma = list.find(',', pos);
        std::string want = list.substr(pos, comma == std::string::npos
                                                ? std::string::npos
                                                : comma - pos);
        pos = comma == std::string::npos ? list.size() + 1 : comma + 1;
        if (want.empty())
          continue;
        if (const auto *node = fs_.find(want.c_str())) {
          std::vector<uint8_t> buf(node->size);
          int64_t n = fs_.read(*node, buf.data(), 0, node->size);
          const char *base = std::strrchr(want.c_str(), '/');
          std::string out =
              std::string("/tmp/") + (base ? base + 1 : want.c_str());
          if (FILE *f = std::fopen(out.c_str(), "wb")) {
            std::fwrite(buf.data(), 1, n > 0 ? n : 0, f);
            std::fclose(f);
            std::fprintf(stderr, "[pkg] dumped %s -> %s (%lld bytes)\n",
                         want.c_str(), out.c_str(), (long long)n);
          }
        } else {
          std::fprintf(stderr, "[pkg] DUMP: %s not found\n", want.c_str());
        }
      }
    }
  }
  bool valid() const { return fs_.valid(); }

  // The title's TITLE_ID from the outer-PKG param.sfo (entry 0x1000). That entry
  // lives in the PKG header, outside the encrypted PFS, so it reads even for
  // titles (e.g. Isaac) whose only param.sfo copy is there and never appears at
  // /app0/sce_sys. Returns "" when unavailable.
  std::string titleId() {
    std::vector<uint8_t> sfo;
    if (fs_.readPkgEntry(0x1000, sfo) > 0)
      return sfoGet(sfo.data(), sfo.size(), "TITLE_ID");
    return {};
  }

  // SOTTR workaround: cache every .manifest.bin's bytes keyed by its base name
  // (e.g. "PRIORITY7_ENGLISH"), so the count-setter can fill the header buffer
  // with correct data (the engine's async manifest reader races on our threads).
  void cacheManifests() {
    if (!std::getenv("DELTA_HDR_FILL"))
      return;
    std::vector<std::string> all;
    fs_.paths(all);
    for (const auto &p : all) {
      const char *suf = ".manifest.bin";
      size_t sl = std::strlen(suf);
      if (p.size() <= sl || p.compare(p.size() - sl, sl, suf) != 0)
        continue;
      const auto *node = fs_.find(p.c_str());
      if (!node)
        continue;
      std::vector<uint8_t> buf(node->size);
      int64_t n = fs_.read(*node, buf.data(), 0, node->size);
      if (n <= 0)
        continue;
      buf.resize(static_cast<size_t>(n));
      size_t start = (p[0] == '/') ? 1 : 0;
      std::string key = p.substr(start, p.size() - start - sl);
      krnl::vfs::cacheFile(key, std::move(buf));
    }
  }

  std::unique_ptr<krnl::vfs::VirtualFile> open(const char *rel) override {
    maybeDump();
    const auto *node = fs_.find(rel);
    if (!node)
      return nullptr;
    return std::make_unique<PkgFile>(&fs_, *node);
  }
  void maybeDump() {
    static bool done = false;
    const char *want = std::getenv("DELTA_PKG_DUMP");
    if (done || !want)
      return;
    done = true;
    if (const auto *node = fs_.find(want)) {
      std::vector<uint8_t> buf(node->size);
      int64_t n = fs_.read(*node, buf.data(), 0, node->size);
      const char *base = std::strrchr(want, '/');
      std::string out = std::string("/tmp/") + (base ? base + 1 : want);
      if (FILE *f = std::fopen(out.c_str(), "wb")) {
        std::fwrite(buf.data(), 1, n > 0 ? n : 0, f);
        std::fclose(f);
        std::fprintf(stderr, "[pkg] dumped %s -> %s (%lld bytes)\n", want,
                     out.c_str(), (long long)n);
      }
    }
  }
  bool stat(const char *rel, int64_t &size) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return false;
    size = static_cast<int64_t>(node->size);
    return true;
  }
  bool list(const char *rel, std::vector<krnl::vfs::DirEntry> &out) override {
    // Build "prefix/" so we match only paths inside this directory. Root ("" or
    // "/") -> "/". The pkg stores absolute paths with a leading '/'.
    std::string prefix(rel ? rel : "");
    while (!prefix.empty() && prefix.back() == '/')
      prefix.pop_back();
    prefix += "/";
    if (prefix.empty() || prefix[0] != '/')
      prefix.insert(prefix.begin(), '/');

    std::vector<std::string> all;
    fs_.paths(all);
    std::set<std::string> seen;
    for (const auto &p : all) {
      if (p.size() <= prefix.size() || p.compare(0, prefix.size(), prefix) != 0)
        continue;
      std::string rest = p.substr(prefix.size());
      auto slash = rest.find('/');
      bool isDir = slash != std::string::npos;
      std::string child = isDir ? rest.substr(0, slash) : rest;
      if (!child.empty() && seen.insert(child).second)
        out.push_back({child, isDir});
    }
    return !out.empty();
  }

private:
  struct PkgFile : krnl::vfs::VirtualFile {
    vfs::PkgFilesystem *fs;
    vfs::PkgFilesystem::Node node;
    PkgFile(vfs::PkgFilesystem *f, const vfs::PkgFilesystem::Node &n)
        : fs(f), node(n) {}
    int64_t read(void *buf, int64_t off, int64_t len) override {
      return fs->read(node, buf, off, len);
    }
    int64_t size() override { return static_cast<int64_t>(node.size); }
  };

  vfs::PkgFilesystem fs_;
};

// Bridges a UFS2 (*.ffpkg) game backup into the kernel VFS. The files inside are
// already decrypted, so this is a straight filesystem mount (no crypto chain).
class Ufs2Provider : public krnl::vfs::VirtualProvider {
public:
  explicit Ufs2Provider(const base::String &path) : fs_(path) {}
  bool valid() const { return fs_.valid(); }

  std::unique_ptr<krnl::vfs::VirtualFile> open(const char *rel) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return nullptr;
    return std::make_unique<Ufs2File>(&fs_, *node);
  }
  bool stat(const char *rel, int64_t &size) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return false;
    size = static_cast<int64_t>(node->size);
    return true;
  }
  bool list(const char *rel, std::vector<krnl::vfs::DirEntry> &out) override {
    std::string prefix(rel ? rel : "");
    while (!prefix.empty() && prefix.back() == '/')
      prefix.pop_back();
    prefix += "/";
    if (prefix[0] != '/')
      prefix.insert(prefix.begin(), '/');
    std::vector<std::string> all;
    fs_.paths(all);
    std::set<std::string> seen;
    for (const auto &p : all) {
      if (p.size() <= prefix.size() || p.compare(0, prefix.size(), prefix) != 0)
        continue;
      std::string rest = p.substr(prefix.size());
      auto slash = rest.find('/');
      bool isDir = slash != std::string::npos;
      std::string child = isDir ? rest.substr(0, slash) : rest;
      if (!child.empty() && seen.insert(child).second)
        out.push_back({child, isDir});
    }
    return !out.empty();
  }

  // True when the backup carries a decrypted/ tree of plaintext ELFs.
  bool hasDecrypted() { return fs_.find("/decrypted/eboot.bin") != nullptr; }

  // The title's id (e.g. "PPSA03311"). PS5 backups carry sce_sys/param.json
  // instead of the PS4 param.sfo; pull the "titleId" string out of it.
  std::string titleId() {
    const auto *node = fs_.find("/sce_sys/param.json");
    if (!node || node->size > (1u << 20))
      return {};
    std::string js(node->size, '\0');
    if (fs_.read(*node, js.data(), 0, static_cast<int64_t>(js.size())) <= 0)
      return {};
    return jsonGetString(js, "titleId");
  }

private:
  struct Ufs2File : krnl::vfs::VirtualFile {
    vfs::Ufs2Filesystem *fs;
    vfs::Ufs2Filesystem::Node node;
    Ufs2File(vfs::Ufs2Filesystem *f, const vfs::Ufs2Filesystem::Node &n)
        : fs(f), node(n) {}
    int64_t read(void *buf, int64_t off, int64_t len) override {
      return fs->read(node, buf, off, len);
    }
    int64_t size() override { return static_cast<int64_t>(node.size); }
  };

  vfs::Ufs2Filesystem fs_;
};

bool endsWithIgnoreCase(const base::String &s, const char *ext) {
  size_t n = s.length(), e = std::strlen(ext);
  if (n < e)
    return false;
  const char *p = s.c_str() + (n - e);
  for (size_t i = 0; i < e; ++i) {
    char a = p[i];
    if (a >= 'A' && a <= 'Z')
      a += 'a' - 'A';
    if (a != ext[i])
      return false;
  }
  return true;
}
} // namespace

void deltaCore::boot(const base::String &xdir) {
  base::String path = xdir;

#ifdef _WIN32
  for (auto &c : path)
    if (c == '/')
      c = '\\';
#endif

  const bool isPkg = endsWithIgnoreCase(xdir, ".pkg");
  const bool isFfpkg = endsWithIgnoreCase(xdir, ".ffpkg");
  // A raw app dump: the extracted /app0 tree itself, identified by its PS5
  // sce_sys/param.json. Host-mounted rather than read through an image reader.
  std::string appJson = std::string(path.c_str()) + "/sce_sys/param.json";
  const bool isAppDir = !isPkg && !isFfpkg && utl::File(base::String(appJson.c_str()),
                                                       utl::fileMode::read).Exists();
  base::String mainModule = path;

  if (isPkg) {
    auto provider = std::make_shared<PkgProvider>(path);
    if (!provider->valid()) {
      LOG_ERROR("failed to load pkg {}", path.c_str());
      return;
    }
    krnl::vfs::mountVirtual("/app0", provider);
    provider->cacheManifests();
    // Publish the title id so savedata can give this game its own host save
    // root (else saves for different titles collide under one directory).
    krnl::vfs::setTitleId(provider->titleId());
    mainModule = base::String("/app0/eboot.bin");
  } else if (isFfpkg) {
    // PS5 game backup (UFS2). Mount it at /app0 and prefer the decrypted/ tree
    // of plaintext ELFs when the dump provides one (the top-level eboot.bin is a
    // still-encrypted SELF).
    auto provider = std::make_shared<Ufs2Provider>(path);
    if (!provider->valid()) {
      LOG_ERROR("failed to load ffpkg {}", path.c_str());
      return;
    }
    bool decrypted = provider->hasDecrypted();
    krnl::vfs::mountVirtual("/app0", provider);
    krnl::vfs::setTitleId(provider->titleId());
    mainModule = base::String(decrypted ? "/app0/decrypted/eboot.bin"
                                        : "/app0/eboot.bin");
    LOG_INFO("mounted ffpkg at /app0 ({}), boot module {}",
             krnl::vfs::titleId().c_str(), mainModule.c_str());
  } else if (isAppDir) {
    krnl::vfs::mount("/app0", path.c_str());
    if (utl::File f(base::String(appJson.c_str()), utl::fileMode::read); f.IsOpen()) {
      std::string js(static_cast<size_t>(f.GetSize()), '\0');
      f.Read(js.data(), js.size());
      krnl::vfs::setTitleId(jsonGetString(js, "titleId"));
    }
    mainModule = base::String("/app0/eboot.bin");
    LOG_INFO("mounted app dir at /app0 ({}), boot module {}",
             krnl::vfs::titleId().c_str(), mainModule.c_str());
  }

  // These all boot from an /app0 mount rather than a bare host path.
  const bool mounted = isPkg || isFfpkg || isAppDir;
  const bool isPs5 = isFfpkg || isAppDir;
  // Name the window after the booted title, since the renderer and the videoout
  // HLE both bring it up with a generic title depending on who gets there first.
  {
    const std::string &tid = krnl::vfs::titleId();
    gfx::setTitle(("prosperity - " + (tid.empty() ? std::string("unknown") : tid) +
                   (isPs5 ? " (PS5)" : " (PS4)")).c_str());
  }
  std::thread ctx([mainModule = std::move(mainModule), mounted, isPs5]() {
    auto p = base::MakeUnique<krnl::proc>();
    if (isPs5)
      p->setPlatform(krnl::proc::platform::ps5);
    if (!p->create(mainModule, mounted))
      return;

    p->start();
  });

  ctx.detach();
}
