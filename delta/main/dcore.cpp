/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base/environment_variables.h>
#include "base/arch.h"
#include <base/logging.h>
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
#include <gpu/rhi/renderer.h>
#include <gfx/gfx_audio.h>
#include <kern/ps4/audio_sink.h>
#include <kern/ps4/hardware_mode.h>
#include <kern/crash.h>
#include <kern/probe/probe_arm.h>
#include <kern/vfs.h>
#include <kern/vfs_providers.h>

#include "formats/archive_object.h"
#include "formats/pup_object.h"
#include "formats/title_metadata.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);
}  // namespace

deltaCore::deltaCore() = default;
deltaCore::~deltaCore() = default;

bool deltaCore::init() {
  LOG_INFO("Initializing deltaCore " rsc_copyright);
  // Three collaborators kern is not allowed to name: the PM4 write-watch the
  // probes arm, the CS-range describer the crash dump asks for, and the audio
  // daemon's host sink. The composition root introduces them.
  gpu::ps4::SetWriteWatchCallback(&krnl::probe::startWriteWatch);
  krnl::setCsRangeDescriber(&gpu::rhi::DescribeCsRangeCovering);
  krnl::ps4::setAudioSink({prosperity_audio_open, prosperity_audio_output,
                           prosperity_audio_volume, prosperity_audio_close});
  return true;
}

namespace {
// The window title bar is the only consumer of a title's icon, so only the
// desktop build pays for reading it.
#if defined(__linux__) && !defined(__ANDROID__)
constexpr bool kWantIcon = true;
#else
constexpr bool kWantIcon = false;
#endif

constexpr u64 kMaxSfoSize = 1u << 20;
constexpr u64 kMaxIconSize = 16u << 20;

using formats::jsonGetString;
using formats::jsonGetTitleName;
using formats::parseSdkVersion;
using formats::sfoGet;
using formats::sfoGetU32;

bool readHostFile(const std::string &path, u64 maxSize,
                  std::vector<u8> &out) {
  utl::File file(base::String(path.c_str()), utl::fileMode::read);
  if (!file.IsOpen())
    return false;
  const u64 size = file.GetSize();
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
  // A game left inside the container it was distributed in (.rar, .zip). The
  // tree inside is an ordinary app dump; we just decompress it on demand rather
  // than making the host find room for the extracted copy.
  const bool isArchive = !isPkg && !isFfpkg && vfs::isArchivePath(xdir.c_str());
  // A raw app dump: the extracted /app0 tree itself, identified by its console
  // metadata. Host-mounted rather than read through an image reader.
  const std::string appRoot(path.c_str());
  const std::string appSfo = appRoot + "/sce_sys/param.sfo";
  const std::string appJson = appRoot + "/sce_sys/param.json";
  // IsOpen(), not Exists(): the File ctor always allocates its backing object, so
  // Exists() is true even for a missing path. A PS5 dump has no param.sfo, and
  // treating it as a PS4 app dir loses both the title id and the platform.
  const bool isPs4AppDir =
      !isPkg && !isFfpkg && !isArchive &&
      utl::File(base::String(appSfo.c_str()), utl::fileMode::read).IsOpen();
  const bool isPs5AppDir =
      !isPkg && !isFfpkg && !isArchive && !isPs4AppDir &&
      utl::File(base::String(appJson.c_str()), utl::fileMode::read).IsOpen();
  const bool isAppDir = isPs4AppDir || isPs5AppDir;
  bool isPs5Archive = false;
  base::String mainModule = path;
  u32 sdkVersion = 0;
  u32 ps4Attributes = 0;
  std::string gameTitle;
#if defined(__linux__) && !defined(__ANDROID__)
  std::vector<u8> gameIcon;
#endif

  if (isPkg) {
    auto mount = krnl::vfs::mountPkg(path, kWantIcon);
    if (!mount)
      return;
    krnl::vfs::mountVirtual("/app0", mount.provider);
    // Publish the title id so savedata can give this game its own host save
    // root (else saves for different titles collide under one directory).
    krnl::vfs::setTitleId(mount.titleId);
    gameTitle = mount.title;
    ps4Attributes = mount.attributes;
#if defined(__linux__) && !defined(__ANDROID__)
    gameIcon = std::move(mount.icon);
#endif
    mainModule = base::String("/app0/eboot.bin");
  } else if (isFfpkg) {
    // PS5 game backup (UFS2). Mount it at /app0 and prefer the decrypted/ tree
    // of plaintext ELFs when the dump provides one (the top-level eboot.bin is a
    // still-encrypted SELF).
    auto mount = krnl::vfs::mountFfpkg(path, kWantIcon);
    if (!mount)
      return;
    krnl::vfs::mountVirtual("/app0", mount.provider);
    krnl::vfs::setTitleId(mount.titleId);
    gameTitle = mount.title;
#if defined(__linux__) && !defined(__ANDROID__)
    gameIcon = std::move(mount.icon);
#endif
    sdkVersion = mount.sdkVersion;
    mainModule = base::String(mount.hasDecrypted ? "/app0/decrypted/eboot.bin"
                                                 : "/app0/eboot.bin");
    LOG_INFO("mounted ffpkg at /app0 ({}), boot module {}",
             krnl::vfs::titleId().c_str(), mainModule.c_str());
  } else if (isArchive) {
    auto mount = krnl::vfs::mountArchive(path, kWantIcon);
    if (!mount)
      return;
    isPs5Archive = mount.isPs5;
    krnl::vfs::setTitleId(mount.titleId);
    gameTitle = mount.title;
    sdkVersion = mount.sdkVersion;
    ps4Attributes = mount.attributes;
#if defined(__linux__) && !defined(__ANDROID__)
    gameIcon = std::move(mount.icon);
#endif
    krnl::vfs::mountVirtual("/app0", mount.provider);
    mainModule = base::String(mount.hasDecrypted ? "/app0/decrypted/eboot.bin"
                                                 : "/app0/eboot.bin");
    LOG_INFO("mounted archive at /app0 ({}), boot module {}",
             krnl::vfs::titleId().c_str(), mainModule.c_str());
  } else if (isAppDir) {
    krnl::vfs::mount("/app0", path.c_str());
    if (isPs4AppDir) {
      std::vector<u8> sfo;
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
      std::vector<u8> json;
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
    std::vector<u8> sfo;
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
  if (isPkg || isFfpkg || isAppDir || isArchive) {
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
  const bool mounted = isPkg || isFfpkg || isAppDir || isArchive;
  const bool isPs5 = isFfpkg || isPs5AppDir || isPs5Archive;
  krnl::ps4::setTitleAttributes(isPs5 ? 0 : ps4Attributes);
  gpu::ps4::SetPs4NeoMode(!isPs5 && krnl::ps4::isNeoMode());
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
