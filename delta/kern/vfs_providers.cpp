/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "kern/vfs_providers.h"

#include <cstdlib>
#include <cstring>

#include <base/logging.h>
#include <logger/logger.h>

#include <options/options.h>

#include "formats/archive_filesystem.h"
#include "formats/pkg_filesystem.h"
#include "formats/ufs2_filesystem.h"
#include "formats/title_metadata.h"
#include <base/containers/set.h>
#include <base/containers/vector.h>
#include <base/memory/move.h>
#include <base/memory/shared_pointer.h>
#include <base/memory/unique_pointer.h>
#include <base/strings/xstring.h>

namespace {
DELTA_OPTION(const char *, kPkgLs, "DELTA_PKG_LS", nullptr);
DELTA_OPTION(const char *, kPkgDump, "DELTA_PKG_DUMP", nullptr);
// Present a zero-filled header where the pkg has none, so a title that reads
// one during boot gets defined bytes rather than a short read.
DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);

constexpr u64 kMaxSfoSize = 1u << 20;
constexpr u64 kMaxIconSize = 16u << 20;

using formats::jsonGetString;
using formats::jsonGetTitleName;
using formats::parseSdkVersion;
using formats::sfoGet;
using formats::sfoGetU32;

// Bridges a PkgFilesystem into the kernel VFS as an on-demand virtual mount.
class PkgProvider : public krnl::vfs::VirtualProvider {
public:
  explicit PkgProvider(const base::String &path) : fs_(path) {
    if (const char *sub = kPkgLs) {
      base::Vector<base::String> all;
      fs_.paths(all);
      for (const auto &p : all)
        if (sub[0] == '1' || p.find(sub) != base::String::npos) {
          const auto *n = fs_.find(p.c_str());
          BASE_LOGI("pkg", "{:12}  {}", n ? (long long)n->size : -1LL,
                    p.c_str());
        }
    }
    if (const char *wantEnv = kPkgDump) {
      base::String list(wantEnv);
      size_t pos = 0;
      while (pos <= list.size()) {
        size_t comma = list.find(',', pos);
        base::String want = list.substr(pos, comma == base::String::npos
                                                ? base::String::npos
                                                : comma - pos);
        pos = comma == base::String::npos ? list.size() + 1 : comma + 1;
        if (want.empty())
          continue;
        if (const auto *node = fs_.find(want.c_str())) {
          base::Vector<u8> buf(node->size);
          i64 n = fs_.read(*node, buf.data(), 0, node->size);
          const char *base = std::strrchr(want.c_str(), '/');
          base::String out =
              base::String("/tmp/") + (base ? base + 1 : want.c_str());
          if (FILE *f = std::fopen(out.c_str(), "wb")) {
            std::fwrite(buf.data(), 1, n > 0 ? n : 0, f);
            std::fclose(f);
            BASE_LOGI("pkg", "dumped {} -> {} ({} bytes)", want.c_str(),
                      out.c_str(), (long long)n);
          }
        } else {
          BASE_LOGI("pkg", "DUMP: {} not found", want.c_str());
        }
      }
    }
  }
  bool valid() const { return fs_.valid(); }

  // The title's TITLE_ID from the outer-PKG param.sfo (entry 0x1000). That entry
  // lives in the PKG header, outside the encrypted PFS, so it reads even for
  // titles (e.g. Isaac) whose only param.sfo copy is there and never appears at
  // /app0/sce_sys. Returns "" when unavailable.
  base::String titleId() {
    base::Vector<u8> sfo;
    if (fs_.readPkgEntry(0x1000, sfo) > 0)
      return sfoGet(sfo.data(), sfo.size(), "TITLE_ID");
    return {};
  }

  base::String title() {
    base::Vector<u8> sfo;
    if (fs_.readPkgEntry(0x1000, sfo) > 0)
      return sfoGet(sfo.data(), sfo.size(), "TITLE");
    return {};
  }

  u32 attributes() {
    base::Vector<u8> sfo;
    if (fs_.readPkgEntry(0x1000, sfo) > 0)
      return sfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
    return 0;
  }

  base::Vector<u8> icon() {
    base::Vector<u8> png;
    fs_.readPkgEntry(0x1200, png);
    return png;
  }

  // SOTTR workaround: cache every .manifest.bin's bytes keyed by its base name
  // (e.g. "PRIORITY7_ENGLISH"), so the count-setter can fill the header buffer
  // with correct data (the engine's async manifest reader races on our threads).
  void cacheManifests() {
    if (!kHdrFill)
      return;
    base::Vector<base::String> all;
    fs_.paths(all);
    for (const auto &p : all) {
      const char *suf = ".manifest.bin";
      size_t sl = std::strlen(suf);
      if (p.size() <= sl || p.compare(p.size() - sl, sl, suf) != 0)
        continue;
      const auto *node = fs_.find(p.c_str());
      if (!node)
        continue;
      base::Vector<u8> buf(node->size);
      i64 n = fs_.read(*node, buf.data(), 0, node->size);
      if (n <= 0)
        continue;
      buf.resize(static_cast<size_t>(n));
      size_t start = (p[0] == '/') ? 1 : 0;
      base::String key = p.substr(start, p.size() - start - sl);
      krnl::vfs::cacheFile(key, base::move(buf));
    }
  }

  base::UniquePointer<krnl::vfs::VirtualFile> open(const char *rel) override {
    maybeDump();
    const auto *node = fs_.find(rel);
    if (!node)
      return nullptr;
    return base::MakeUnique<PkgFile>(&fs_, *node);
  }
  void maybeDump() {
    static bool done = false;
    const char *want = kPkgDump;
    if (done || !want)
      return;
    done = true;
    if (const auto *node = fs_.find(want)) {
      base::Vector<u8> buf(node->size);
      i64 n = fs_.read(*node, buf.data(), 0, node->size);
      const char *base = std::strrchr(want, '/');
      base::String out = base::String("/tmp/") + (base ? base + 1 : want);
      if (FILE *f = std::fopen(out.c_str(), "wb")) {
        std::fwrite(buf.data(), 1, n > 0 ? n : 0, f);
        std::fclose(f);
        BASE_LOGI("pkg", "dumped {} -> {} ({} bytes)", want, out.c_str(),
                  (long long)n);
      }
    }
  }
  bool stat(const char *rel, i64 &size) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return false;
    size = static_cast<i64>(node->size);
    return true;
  }
  bool list(const char *rel, base::Vector<krnl::vfs::DirEntry> &out) override {
    // Build "prefix/" so we match only paths inside this directory. Root ("" or
    // "/") -> "/". The pkg stores absolute paths with a leading '/'.
    base::String prefix(rel ? rel : "");
    while (!prefix.empty() && prefix.back() == '/')
      prefix.pop_back();
    prefix += "/";
    if (prefix.empty() || prefix[0] != '/')
      prefix.insert(0, 1, '/');

    base::Vector<base::String> all;
    fs_.paths(all);
    base::Set<base::String> seen;
    for (const auto &p : all) {
      if (p.size() <= prefix.size() || p.compare(0, prefix.size(), prefix) != 0)
        continue;
      base::String rest = p.substr(prefix.size());
      auto slash = rest.find('/');
      bool isDir = slash != base::String::npos;
      base::String child = isDir ? rest.substr(0, slash) : rest;
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
    i64 read(void *buf, i64 off, i64 len) override {
      return fs->read(node, buf, off, len);
    }
    i64 size() override { return static_cast<i64>(node.size); }
  };

  vfs::PkgFilesystem fs_;
};

// Bridges a UFS2 (*.ffpkg) game backup into the kernel VFS. The files inside are
// already decrypted, so this is a straight filesystem mount (no crypto chain).
class Ufs2Provider : public krnl::vfs::VirtualProvider {
public:
  explicit Ufs2Provider(const base::String &path) : fs_(path) {}
  bool valid() const { return fs_.valid(); }

  base::UniquePointer<krnl::vfs::VirtualFile> open(const char *rel) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return nullptr;
    return base::MakeUnique<Ufs2File>(&fs_, *node);
  }
  bool stat(const char *rel, i64 &size) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return false;
    size = static_cast<i64>(node->size);
    return true;
  }
  bool list(const char *rel, base::Vector<krnl::vfs::DirEntry> &out) override {
    base::String prefix(rel ? rel : "");
    while (!prefix.empty() && prefix.back() == '/')
      prefix.pop_back();
    prefix += "/";
    if (prefix[0] != '/')
      prefix.insert(0, 1, '/');
    base::Vector<base::String> all;
    fs_.paths(all);
    base::Set<base::String> seen;
    for (const auto &p : all) {
      if (p.size() <= prefix.size() || p.compare(0, prefix.size(), prefix) != 0)
        continue;
      base::String rest = p.substr(prefix.size());
      auto slash = rest.find('/');
      bool isDir = slash != base::String::npos;
      base::String child = isDir ? rest.substr(0, slash) : rest;
      if (!child.empty() && seen.insert(child).second)
        out.push_back({child, isDir});
    }
    return !out.empty();
  }

  // True when the backup carries a decrypted/ tree of plaintext ELFs.
  bool hasDecrypted() { return fs_.find("/decrypted/eboot.bin") != nullptr; }

  // The title's id (e.g. "PPSA03311"). PS5 backups carry sce_sys/param.json
  // instead of the PS4 param.sfo; pull the "titleId" string out of it.
  base::String titleId() { return paramJsonField("titleId"); }

  base::String title() { return jsonGetTitleName(paramJson()); }

  base::Vector<u8> icon() {
    const auto *node = fs_.find("/sce_sys/icon0.png");
    if (!node || node->size > kMaxIconSize)
      return {};
    base::Vector<u8> png(node->size);
    const i64 read = fs_.read(*node, png.data(), 0, node->size);
    if (read <= 0)
      return {};
    png.resize(static_cast<size_t>(read));
    return png;
  }

  // param.json spells the SDK version as a 64-bit hex string ("0x0300...")
  // whose top half is the 0xMMmmpppp form libkernel compares against.
  u32 sdkVersion() { return parseSdkVersion(paramJsonField("sdkVersion")); }

private:
  base::String paramJson() {
    const auto *node = fs_.find("/sce_sys/param.json");
    if (!node || node->size > (1u << 20))
      return {};
    base::String js(node->size, '\0');
    if (fs_.read(*node, js.data(), 0, static_cast<i64>(js.size())) <= 0)
      return {};
    return js;
  }

  base::String paramJsonField(const char *key) {
    return jsonGetString(paramJson(), key);
  }

  struct Ufs2File : krnl::vfs::VirtualFile {
    vfs::Ufs2Filesystem *fs;
    vfs::Ufs2Filesystem::Node node;
    Ufs2File(vfs::Ufs2Filesystem *f, const vfs::Ufs2Filesystem::Node &n)
        : fs(f), node(n) {}
    i64 read(void *buf, i64 off, i64 len) override {
      return fs->read(node, buf, off, len);
    }
    i64 size() override { return static_cast<i64>(node.size); }
  };

  vfs::Ufs2Filesystem fs_;
};

// Bridges a plain .rar/.zip of a game dump into the kernel VFS. Same shape as
// the pkg and ufs2 providers, but the archive holds an ordinary extracted app
// tree, so the only work beyond decompression is reading the metadata out of it
// to tell a PS4 title from a PS5 one.
class ArchiveProvider : public krnl::vfs::VirtualProvider {
public:
  explicit ArchiveProvider(const base::String &path) : fs_(path) {}
  bool valid() const { return fs_.valid(); }

  base::UniquePointer<krnl::vfs::VirtualFile> open(const char *rel) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return nullptr;
    return base::MakeUnique<ArchiveFile>(&fs_, *node);
  }
  bool stat(const char *rel, i64 &size) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return false;
    size = static_cast<i64>(node->size);
    return true;
  }
  bool list(const char *rel, base::Vector<krnl::vfs::DirEntry> &out) override {
    base::Vector<vfs::ArchiveFilesystem::Child> children;
    if (!fs_.list(rel, children))
      return false;
    for (auto &c : children)
      out.push_back({base::move(c.name), c.isDir});
    return true;
  }

  // A PS5 dump carries sce_sys/param.json, a PS4 one sce_sys/param.sfo.
  bool isPs5() { return fs_.find("/sce_sys/param.json") != nullptr; }
  bool hasDecrypted() { return fs_.find("/decrypted/eboot.bin") != nullptr; }

  base::String titleId() {
    if (isPs5())
      return jsonGetString(paramJson(), "titleId");
    base::Vector<u8> sfo = readWhole("/sce_sys/param.sfo", kMaxSfoSize);
    return sfoGet(sfo.data(), sfo.size(), "TITLE_ID");
  }

  base::String title() {
    if (isPs5())
      return jsonGetTitleName(paramJson());
    base::Vector<u8> sfo = readWhole("/sce_sys/param.sfo", kMaxSfoSize);
    return sfoGet(sfo.data(), sfo.size(), "TITLE");
  }

  u32 attributes() {
    base::Vector<u8> sfo = readWhole("/sce_sys/param.sfo", kMaxSfoSize);
    return sfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
  }

  u32 sdkVersion() {
    return parseSdkVersion(jsonGetString(paramJson(), "sdkVersion"));
  }

  base::Vector<u8> icon() { return readWhole("/sce_sys/icon0.png", kMaxIconSize); }

private:
  base::Vector<u8> readWhole(const char *rel, u64 maxSize) {
    const auto *node = fs_.find(rel);
    if (!node || node->size == 0 || node->size > maxSize)
      return {};
    base::Vector<u8> buf(node->size);
    const i64 read = fs_.read(*node, buf.data(), 0, static_cast<i64>(buf.size()));
    if (read <= 0)
      return {};
    buf.resize(static_cast<size_t>(read));
    return buf;
  }

  base::String paramJson() {
    const base::Vector<u8> js = readWhole("/sce_sys/param.json", kMaxSfoSize);
    return base::String(reinterpret_cast<const char *>(js.data()), js.size());
  }

  struct ArchiveFile : krnl::vfs::VirtualFile {
    vfs::ArchiveFilesystem *fs;
    vfs::ArchiveFilesystem::Node node;
    ArchiveFile(vfs::ArchiveFilesystem *f,
                const vfs::ArchiveFilesystem::Node &n)
        : fs(f), node(n) {}
    i64 read(void *buf, i64 off, i64 len) override {
      return fs->read(node, buf, off, len);
    }
    i64 size() override { return static_cast<i64>(node.size); }
  };

  vfs::ArchiveFilesystem fs_;
};

}  // namespace

namespace krnl::vfs {

namespace {
// Every container answers the same questions; only the accessors differ.
template <typename P>
TitleMount mount(const base::String &path, bool wantIcon, const char *what) {
  TitleMount m;
  auto p = base::MakeShared<P>(path);
  if (!p->valid()) {
    LOG_ERROR("failed to load {} {}", what, path.c_str());
    return m;
  }
  m.titleId = p->titleId();
  m.title = p->title();
  if (wantIcon)
    m.icon = p->icon();
  m.provider = p;
  return m;
}
}  // namespace

TitleMount mountPkg(const base::String &path, bool wantIcon) {
  TitleMount m = mount<PkgProvider>(path, wantIcon, "pkg");
  if (m) {
    auto *p = static_cast<PkgProvider *>(m.provider.get());
    m.attributes = p->attributes();
    p->cacheManifests();
  }
  return m;
}

TitleMount mountFfpkg(const base::String &path, bool wantIcon) {
  TitleMount m = mount<Ufs2Provider>(path, wantIcon, "ffpkg");
  if (m) {
    auto *p = static_cast<Ufs2Provider *>(m.provider.get());
    m.sdkVersion = p->sdkVersion();
    m.hasDecrypted = p->hasDecrypted();
    m.isPs5 = true;
  }
  return m;
}

TitleMount mountArchive(const base::String &path, bool wantIcon) {
  TitleMount m = mount<ArchiveProvider>(path, wantIcon, "archive");
  if (m) {
    auto *p = static_cast<ArchiveProvider *>(m.provider.get());
    m.isPs5 = p->isPs5();
    m.hasDecrypted = p->hasDecrypted();
    if (m.isPs5)
      m.sdkVersion = p->sdkVersion();
    else
      m.attributes = p->attributes();
  }
  return m;
}

}  // namespace krnl::vfs
