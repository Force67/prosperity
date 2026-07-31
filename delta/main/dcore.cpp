/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base/environment_variables.h>
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
#include <gpu/ps4/cmd_processor.h>
#include <kern/ps4/hardware_mode.h>
#include <kern/vfs.h>

#include "formats/pkg_object.h"
#include "formats/pup_object.h"
#include "formats/ufs2_object.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kPkgLs, "DELTA_PKG_LS", nullptr);
DELTA_OPTION(const char *, kPkgDump, "DELTA_PKG_DUMP", nullptr);
DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);
}  // namespace

deltaCore::deltaCore() = default;
deltaCore::~deltaCore() = default;

bool deltaCore::init() {
  LOG_INFO("Initializing deltaCore " rsc_copyright);
  return true;
}

namespace {
constexpr uint64_t kMaxSfoSize = 1u << 20;
constexpr uint64_t kMaxIconSize = 16u << 20;

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

uint32_t sfoGetU32(const uint8_t *d, size_t n, const char *key) {
  if (n < 20)
    return 0;
  auto rd16 = [&](size_t o) -> uint16_t {
    return o + 2 <= n ? uint16_t(d[o] | (d[o + 1] << 8)) : 0;
  };
  auto rd32 = [&](size_t o) -> uint32_t {
    return o + 4 <= n ? uint32_t(d[o]) | (uint32_t(d[o + 1]) << 8) |
                             (uint32_t(d[o + 2]) << 16) |
                             (uint32_t(d[o + 3]) << 24)
                       : 0;
  };
  if (rd32(0) != 0x46535000u)
    return 0;
  const uint32_t keyStart = rd32(8), dataStart = rd32(12), count = rd32(16);
  const size_t keyLength = std::strlen(key);
  for (uint32_t i = 0, index = 20; i < count; i++, index += 16) {
    if (index + 16 > n)
      break;
    const size_t keyPosition = size_t(keyStart) + rd16(index);
    if (keyPosition + keyLength + 1 > n ||
        std::memcmp(d + keyPosition, key, keyLength) != 0 ||
        d[keyPosition + keyLength] != '\0')
      continue;
    const size_t dataPosition = size_t(dataStart) + rd32(index + 12);
    return dataPosition + sizeof(uint32_t) <= n ? rd32(dataPosition) : 0;
  }
  return 0;
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

// param.json keeps the display name under localizedParameters.<defaultLanguage>
// .titleName. Search from the default language's block so a title shipping
// several languages doesn't pick whichever one happens to come first.
std::string jsonGetTitleName(const std::string &js) {
  const std::string lang = jsonGetString(js, "defaultLanguage");
  if (!lang.empty()) {
    const size_t block = js.find("\"" + lang + "\"");
    if (block != std::string::npos) {
      std::string name = jsonGetString(js.substr(block), "titleName");
      if (!name.empty())
        return name;
    }
  }
  return jsonGetString(js, "titleName");
}

bool readHostFile(const std::string &path, uint64_t maxSize,
                  std::vector<uint8_t> &out) {
  utl::File file(base::String(path.c_str()), utl::fileMode::read);
  if (!file.IsOpen())
    return false;
  const uint64_t size = file.GetSize();
  if (size == 0 || size > maxSize)
    return false;
  out.resize(static_cast<size_t>(size));
  if (file.Read(out.data(), out.size()) != size) {
    out.clear();
    return false;
  }
  return true;
}

std::string parentPath(const base::String &path) {
  const std::string value(path.c_str());
  const size_t slash = value.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : value.substr(0, slash);
}

// param.json stores sdkVersion as "0xMMmmpppp00000000"; libkernel wants the top
// half (0x03000000 for a 3.00 title). Empty/unparsable -> 0.
uint32_t parseSdkVersion(const std::string &s) {
  if (s.empty())
    return 0;
  return static_cast<uint32_t>(std::strtoull(s.c_str(), nullptr, 0) >> 32);
}

// Bridges a PkgFilesystem into the kernel VFS as an on-demand virtual mount.
class PkgProvider : public krnl::vfs::VirtualProvider {
public:
  explicit PkgProvider(const base::String &path) : fs_(path) {
    if (const char *sub = kPkgLs) {
      std::vector<std::string> all;
      fs_.paths(all);
      for (const auto &p : all)
        if (sub[0] == '1' || p.find(sub) != std::string::npos) {
          const auto *n = fs_.find(p.c_str());
          std::fprintf(stderr, "[pkg] %12lld  %s\n",
                       n ? (long long)n->size : -1LL, p.c_str());
        }
    }
    if (const char *wantEnv = kPkgDump) {
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

  std::string title() {
    std::vector<uint8_t> sfo;
    if (fs_.readPkgEntry(0x1000, sfo) > 0)
      return sfoGet(sfo.data(), sfo.size(), "TITLE");
    return {};
  }

  uint32_t attributes() {
    std::vector<uint8_t> sfo;
    if (fs_.readPkgEntry(0x1000, sfo) > 0)
      return sfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
    return 0;
  }

  std::vector<uint8_t> icon() {
    std::vector<uint8_t> png;
    fs_.readPkgEntry(0x1200, png);
    return png;
  }

  // SOTTR workaround: cache every .manifest.bin's bytes keyed by its base name
  // (e.g. "PRIORITY7_ENGLISH"), so the count-setter can fill the header buffer
  // with correct data (the engine's async manifest reader races on our threads).
  void cacheManifests() {
    if (!kHdrFill)
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
    const char *want = kPkgDump;
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
  std::string titleId() { return paramJsonField("titleId"); }

  std::string title() { return jsonGetTitleName(paramJson()); }

  std::vector<uint8_t> icon() {
    const auto *node = fs_.find("/sce_sys/icon0.png");
    if (!node || node->size > kMaxIconSize)
      return {};
    std::vector<uint8_t> png(node->size);
    const int64_t read = fs_.read(*node, png.data(), 0, node->size);
    if (read <= 0)
      return {};
    png.resize(static_cast<size_t>(read));
    return png;
  }

  // param.json spells the SDK version as a 64-bit hex string ("0x0300...")
  // whose top half is the 0xMMmmpppp form libkernel compares against.
  uint32_t sdkVersion() { return parseSdkVersion(paramJsonField("sdkVersion")); }

private:
  std::string paramJson() {
    const auto *node = fs_.find("/sce_sys/param.json");
    if (!node || node->size > (1u << 20))
      return {};
    std::string js(node->size, '\0');
    if (fs_.read(*node, js.data(), 0, static_cast<int64_t>(js.size())) <= 0)
      return {};
    return js;
  }

  std::string paramJsonField(const char *key) {
    return jsonGetString(paramJson(), key);
  }

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
  // A raw app dump: the extracted /app0 tree itself, identified by its console
  // metadata. Host-mounted rather than read through an image reader.
  const std::string appRoot(path.c_str());
  const std::string appSfo = appRoot + "/sce_sys/param.sfo";
  const std::string appJson = appRoot + "/sce_sys/param.json";
  // IsOpen(), not Exists(): the File ctor always allocates its backing object, so
  // Exists() is true even for a missing path. A PS5 dump has no param.sfo, and
  // treating it as a PS4 app dir loses both the title id and the platform.
  const bool isPs4AppDir =
      !isPkg && !isFfpkg &&
      utl::File(base::String(appSfo.c_str()), utl::fileMode::read).IsOpen();
  const bool isPs5AppDir =
      !isPkg && !isFfpkg && !isPs4AppDir &&
      utl::File(base::String(appJson.c_str()), utl::fileMode::read).IsOpen();
  const bool isAppDir = isPs4AppDir || isPs5AppDir;
  base::String mainModule = path;
  uint32_t sdkVersion = 0;
  uint32_t ps4Attributes = 0;
  std::string gameTitle;
#if defined(__linux__) && !defined(__ANDROID__)
  std::vector<uint8_t> gameIcon;
#endif

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
    gameTitle = provider->title();
    ps4Attributes = provider->attributes();
#if defined(__linux__) && !defined(__ANDROID__)
    gameIcon = provider->icon();
#endif
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
    gameTitle = provider->title();
#if defined(__linux__) && !defined(__ANDROID__)
    gameIcon = provider->icon();
#endif
    sdkVersion = provider->sdkVersion();
    mainModule = base::String(decrypted ? "/app0/decrypted/eboot.bin"
                                        : "/app0/eboot.bin");
    LOG_INFO("mounted ffpkg at /app0 ({}), boot module {}",
             krnl::vfs::titleId().c_str(), mainModule.c_str());
  } else if (isAppDir) {
    krnl::vfs::mount("/app0", path.c_str());
    if (isPs4AppDir) {
      std::vector<uint8_t> sfo;
      if (readHostFile(appSfo, kMaxSfoSize, sfo)) {
        krnl::vfs::setTitleId(sfoGet(sfo.data(), sfo.size(), "TITLE_ID"));
        gameTitle = sfoGet(sfo.data(), sfo.size(), "TITLE");
        ps4Attributes = sfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
      }
#if defined(__linux__) && !defined(__ANDROID__)
      if (!readHostFile(appRoot + "/sce_sys/icon0.png", kMaxIconSize, gameIcon))
        readHostFile(appRoot + "/icon0.png", kMaxIconSize, gameIcon);
#endif
    } else {
      std::vector<uint8_t> json;
      readHostFile(appJson, kMaxSfoSize, json);
      const std::string js(json.begin(), json.end());
      krnl::vfs::setTitleId(jsonGetString(js, "titleId"));
      gameTitle = jsonGetTitleName(js);
      sdkVersion = parseSdkVersion(jsonGetString(js, "sdkVersion"));
#if defined(__linux__) && !defined(__ANDROID__)
      if (!readHostFile(appRoot + "/sce_sys/icon0.png", kMaxIconSize, gameIcon))
        readHostFile(appRoot + "/icon0.png", kMaxIconSize, gameIcon);
#endif
    }
    mainModule = base::String("/app0/eboot.bin");
    LOG_INFO("mounted app dir at /app0 ({}), boot module {}",
             krnl::vfs::titleId().c_str(), mainModule.c_str());
  } else {
    const std::string root = parentPath(path);
    std::vector<uint8_t> sfo;
    if (!readHostFile(root + "/sce_sys/param.sfo", kMaxSfoSize, sfo))
      readHostFile(root + "/param.sfo", kMaxSfoSize, sfo);
    if (!sfo.empty()) {
      krnl::vfs::setTitleId(sfoGet(sfo.data(), sfo.size(), "TITLE_ID"));
      gameTitle = sfoGet(sfo.data(), sfo.size(), "TITLE");
      ps4Attributes = sfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
    }
#if defined(__linux__) && !defined(__ANDROID__)
    if (!readHostFile(root + "/sce_sys/icon0.png", kMaxIconSize, gameIcon))
      readHostFile(root + "/icon0.png", kMaxIconSize, gameIcon);
#endif
  }

  // /download0 is the title's writable data volume (patches, add-on content,
  // its own bookkeeping). It always exists on the console, and a title that
  // writes there and reads back fails hard when it doesn't: Skyrim rebuilds its
  // plugin list into /download0/Plugins.txt, and with the write lost it boots
  // with no plugins, no archives and a null menu movie.
  if (isPkg || isFfpkg || isAppDir) {
    base::StringU8 home;
    base::GetEnvironmentVariable(u8"HOME", home);
    std::string tid = krnl::vfs::titleId();
    std::string dl =
        std::string(home.empty() ? "." : (const char *)home.c_str()) +
        "/.prosperity/download/" +
        (tid.empty() ? std::string("UNKNOWN") : tid);
    krnl::vfs::mountWritable("/download0", dl.c_str());
  }

  // The title is known now, so the settings we ship for it can fill in
  // everything the environment / an options file / the command line didn't.
  // Before the guest starts: the knobs below and in the boot thread latch.
  utl::loadGameProfile(krnl::vfs::titleId().c_str());

  // These all boot from an /app0 mount rather than a bare host path.
  const bool mounted = isPkg || isFfpkg || isAppDir;
  const bool isPs5 = isFfpkg || isPs5AppDir;
  krnl::ps4::setTitleAttributes(isPs5 ? 0 : ps4Attributes);
  gpu::SetPs4NeoMode(!isPs5 && krnl::ps4::isNeoMode());
  // Name the window after the booted game, since the renderer and the videoout
  // HLE both bring it up with a generic title depending on who gets there first.
  {
    const std::string &tid = krnl::vfs::titleId();
    std::string title = "prosperity - ";
    title += gameTitle.empty() ? std::string("unknown") : gameTitle;
    title += " - [";
    title += tid.empty() ? std::string("unknown") : tid;
    title += isPs5 ? "] (PS5)" : "] (PS4)";
    LOG_INFO("window title: {}", title.c_str());
    gfx::setTitle(title.c_str());
  }
#if defined(__linux__) && !defined(__ANDROID__)
  if (!gameIcon.empty())
    gfx::setIcon(gameIcon.data(), gameIcon.size());
#endif
  std::thread ctx([mainModule = std::move(mainModule), mounted, isPs5, sdkVersion]() {
    auto p = base::MakeUnique<krnl::proc>();
    if (isPs5)
      p->setPlatform(krnl::proc::platform::ps5);
    p->setSdkVersion(sdkVersion);
    if (!p->create(mainModule, mounted))
      return;

    p->start();
  });

  ctx.detach();
}
