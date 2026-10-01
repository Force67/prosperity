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
#include "base/arch.h"
#include "base/environment_variables.h"
#include "base/logging.h"

#include "io/file.h"
#include "logger/logger.h"
#include "main/launcher.h"

#include "gpu/ps4/cmd_processor.h"
#include "gpu/render/renderer.h"
#include "host/audio_output.h"
#include "host/window.h"
#include "kern/crash.h"
#include "kern/probe/probe_arm.h"
#include "kern/ps4/audio_sink.h"
#include "kern/ps4/hardware_mode.h"
#include "kern/vfs.h"
#include "kern/vfs_providers.h"
#include "kern/vm_map.h"

#include "base/containers/vector.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/thread.h"
#include "formats/archive_filesystem.h"
#include "formats/pup_reader.h"
#include "formats/title_metadata.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);
}  // namespace

Launcher::Launcher() = default;
Launcher::~Launcher() = default;

bool Launcher::Init() {
  LOG_INFO("Initializing prosperity " rsc_copyright);
  // Collaborators kern is not allowed to name: the PM4 write-watch the probes
  // arm, the CS-range describer the crash dump asks for, the guest write
  // tracker that must hear about remapped memory, and the audio daemon's host
  // sink. The composition root introduces them.
  gpu::ps4::SetWriteWatchCallback(&kern::probe::StartWriteWatch);
  kern::SetCsRangeDescriber(&gpu::render::DescribeCsRangeCovering);
  kern::SetMappingChangedHook(&gpu::render::NoteGuestRemap);
  kern::ps4::SetAudioSink({host::OpenAudioPort, host::QueueAudio,
                           host::SetAudioPortVolume, host::CloseAudioPort});
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

using formats::JsonGetString;
using formats::JsonGetTitleName;
using formats::ParseSdkVersion;
using formats::SfoGet;
using formats::SfoGetU32;

bool ReadHostFile(const base::String& path,
                  u64 max_size,
                  base::Vector<u8>& out) {
  io::File file(base::String(path.c_str()), io::FileMode::kRead);
  if (!file.IsOpen())
    return false;
  const u64 size = file.GetSize();
  if (size == 0 || size > max_size)
    return false;
  out.resize(static_cast<size_t>(size));
  if (file.Read(out.data(), out.size()) != size) {
    out.clear();
    return false;
  }
  return true;
}

base::String ParentPath(const base::String& path) {
  const base::String value(path.c_str());
  const size_t slash = value.find_last_of("/\\");
  return slash == base::String::npos ? base::String(".")
                                     : value.substr(0, slash);
}

bool EndsWithIgnoreCase(const base::String& s, const char* ext) {
  size_t n = s.length(), e = std::strlen(ext);
  if (n < e)
    return false;
  const char* p = s.c_str() + (n - e);
  for (size_t i = 0; i < e; ++i) {
    char a = p[i];
    if (a >= 'A' && a <= 'Z')
      a += 'a' - 'A';
    if (a != ext[i])
      return false;
  }
  return true;
}
}  // namespace

void Launcher::Boot(const base::String& xdir) {
  base::String path = xdir;

#ifdef _WIN32
  for (auto& c : path)
    if (c == '/')
      c = '\\';
#endif

  const bool is_pkg = EndsWithIgnoreCase(xdir, ".pkg");
  const bool is_ffpkg = EndsWithIgnoreCase(xdir, ".ffpkg");
  // A game left inside the container it was distributed in (.rar, .zip). The
  // tree inside is an ordinary app dump; we just decompress it on demand rather
  // than making the host find room for the extracted copy.
  const bool is_archive =
      !is_pkg && !is_ffpkg && formats::IsArchivePath(xdir.c_str());
  // A raw app dump: the extracted /app0 tree itself, identified by its console
  // metadata. Host-mounted rather than read through an image reader.
  const base::String app_root(path.c_str());
  const base::String app_sfo = app_root + "/sce_sys/param.sfo";
  const base::String app_json = app_root + "/sce_sys/param.json";
  // IsOpen(), not Exists(): the File ctor always allocates its backing object,
  // so Exists() is true even for a missing path. A PS5 dump has no param.sfo,
  // and treating it as a PS4 app dir loses both the title id and the platform.
  const bool is_ps4_app_dir =
      !is_pkg && !is_ffpkg && !is_archive &&
      io::File(base::String(app_sfo.c_str()), io::FileMode::kRead).IsOpen();
  const bool is_ps5_app_dir =
      !is_pkg && !is_ffpkg && !is_archive && !is_ps4_app_dir &&
      io::File(base::String(app_json.c_str()), io::FileMode::kRead).IsOpen();
  const bool is_app_dir = is_ps4_app_dir || is_ps5_app_dir;
  bool is_ps5_archive = false;
  base::String main_module = path;
  u32 sdk_version = 0;
  u32 ps4_attributes = 0;
  base::String game_title;
#if defined(__linux__) && !defined(__ANDROID__)
  base::Vector<u8> game_icon;
#endif

  if (is_pkg) {
    auto mount = kern::vfs::MountPkg(path, kWantIcon);
    if (!mount)
      return;
    kern::vfs::MountVirtual("/app0", mount.provider);
    // Publish the title id so savedata can give this game its own host save
    // root (else saves for different titles collide under one directory).
    kern::vfs::SetTitleId(mount.title_id);
    game_title = mount.title;
    ps4_attributes = mount.attributes;
#if defined(__linux__) && !defined(__ANDROID__)
    game_icon = base::move(mount.icon);
#endif
    main_module = base::String("/app0/eboot.bin");
  } else if (is_ffpkg) {
    // PS5 game backup (UFS2). Mount it at /app0 and prefer the decrypted/ tree
    // of plaintext ELFs when the dump provides one (the top-level eboot.bin is
    // a still-encrypted SELF).
    auto mount = kern::vfs::MountFfpkg(path, kWantIcon);
    if (!mount)
      return;
    kern::vfs::MountVirtual("/app0", mount.provider);
    kern::vfs::SetTitleId(mount.title_id);
    game_title = mount.title;
#if defined(__linux__) && !defined(__ANDROID__)
    game_icon = base::move(mount.icon);
#endif
    sdk_version = mount.sdk_version;
    main_module = base::String(mount.has_decrypted ? "/app0/decrypted/eboot.bin"
                                                   : "/app0/eboot.bin");
    LOG_INFO("mounted ffpkg at /app0 ({}), boot module {}",
             kern::vfs::TitleId().c_str(), main_module.c_str());
  } else if (is_archive) {
    auto mount = kern::vfs::MountArchive(path, kWantIcon);
    if (!mount)
      return;
    is_ps5_archive = mount.is_ps5;
    kern::vfs::SetTitleId(mount.title_id);
    game_title = mount.title;
    sdk_version = mount.sdk_version;
    ps4_attributes = mount.attributes;
#if defined(__linux__) && !defined(__ANDROID__)
    game_icon = base::move(mount.icon);
#endif
    kern::vfs::MountVirtual("/app0", mount.provider);
    main_module = base::String(mount.has_decrypted ? "/app0/decrypted/eboot.bin"
                                                   : "/app0/eboot.bin");
    LOG_INFO("mounted archive at /app0 ({}), boot module {}",
             kern::vfs::TitleId().c_str(), main_module.c_str());
  } else if (is_app_dir) {
    kern::vfs::Mount("/app0", path.c_str());
    if (is_ps4_app_dir) {
      base::Vector<u8> sfo;
      if (ReadHostFile(app_sfo, kMaxSfoSize, sfo)) {
        kern::vfs::SetTitleId(SfoGet(sfo.data(), sfo.size(), "TITLE_ID"));
        game_title = SfoGet(sfo.data(), sfo.size(), "TITLE");
        ps4_attributes = SfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
      }
#if defined(__linux__) && !defined(__ANDROID__)
      if (!ReadHostFile(app_root + "/sce_sys/icon0.png", kMaxIconSize,
                        game_icon))
        ReadHostFile(app_root + "/icon0.png", kMaxIconSize, game_icon);
#endif
    } else {
      base::Vector<u8> json;
      ReadHostFile(app_json, kMaxSfoSize, json);
      const base::String js(reinterpret_cast<const char*>(json.data()),
                            json.size());
      kern::vfs::SetTitleId(JsonGetString(js, "titleId"));
      game_title = JsonGetTitleName(js);
      sdk_version = ParseSdkVersion(JsonGetString(js, "sdkVersion"));
#if defined(__linux__) && !defined(__ANDROID__)
      if (!ReadHostFile(app_root + "/sce_sys/icon0.png", kMaxIconSize,
                        game_icon))
        ReadHostFile(app_root + "/icon0.png", kMaxIconSize, game_icon);
#endif
    }
    main_module = base::String("/app0/eboot.bin");
    LOG_INFO("mounted app dir at /app0 ({}), boot module {}",
             kern::vfs::TitleId().c_str(), main_module.c_str());
  } else {
    const base::String root = ParentPath(path);
    base::Vector<u8> sfo;
    if (!ReadHostFile(root + "/sce_sys/param.sfo", kMaxSfoSize, sfo))
      ReadHostFile(root + "/param.sfo", kMaxSfoSize, sfo);
    if (!sfo.empty()) {
      kern::vfs::SetTitleId(SfoGet(sfo.data(), sfo.size(), "TITLE_ID"));
      game_title = SfoGet(sfo.data(), sfo.size(), "TITLE");
      ps4_attributes = SfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
    }
#if defined(__linux__) && !defined(__ANDROID__)
    if (!ReadHostFile(root + "/sce_sys/icon0.png", kMaxIconSize, game_icon))
      ReadHostFile(root + "/icon0.png", kMaxIconSize, game_icon);
#endif
  }

  // /download0 is the title's writable data volume (patches, add-on content,
  // its own bookkeeping). It always exists on the console, and a title that
  // writes there and reads back fails hard when it doesn't: Skyrim rebuilds its
  // plugin list into /download0/Plugins.txt, and with the write lost it boots
  // with no plugins, no archives and a null menu movie.
  if (is_pkg || is_ffpkg || is_app_dir || is_archive) {
    base::StringU8 home;
    base::GetEnvironmentVariable(u8"HOME", home);
    base::String tid = kern::vfs::TitleId();
    base::String dl =
        base::String(home.empty() ? "." : (const char*)home.c_str()) +
        "/.prosperity/download/" +
        (tid.empty() ? base::String("UNKNOWN") : tid);
    kern::vfs::MountWritable("/download0", dl.c_str());
  }

  // The title is known now, so the settings we ship for it can fill in
  // everything the environment / an options file / the command line didn't.
  // Before the guest starts: the knobs below and in the boot thread latch.
  options::LoadGameProfile(kern::vfs::TitleId().c_str());

  // These all boot from an /app0 mount rather than a bare host path.
  const bool mounted = is_pkg || is_ffpkg || is_app_dir || is_archive;
  const bool is_ps5 = is_ffpkg || is_ps5_app_dir || is_ps5_archive;
  kern::ps4::SetTitleAttributes(is_ps5 ? 0 : ps4_attributes);
  gpu::ps4::SetPs4NeoMode(!is_ps5 && kern::ps4::IsNeoMode());
  // Name the window after the booted game, since the renderer and the videoout
  // HLE both bring it up with a generic title depending on who gets there
  // first.
  {
    const base::String& tid = kern::vfs::TitleId();
    base::String title = "prosperity - ";
    title += game_title.empty() ? base::String("unknown") : game_title;
    title += " - [";
    title += tid.empty() ? base::String("unknown") : tid;
    title += is_ps5 ? "] (PS5)" : "] (PS4)";
    LOG_INFO("window title: {}", title.c_str());
    host::SetTitle(title.c_str());
  }
#if defined(__linux__) && !defined(__ANDROID__)
  if (!game_icon.empty())
    host::SetIcon(game_icon.data(), game_icon.size());
  if (io::File art = kern::vfs::OpenRead("/app0/sce_sys/pic0.png");
      art.Exists() && art.GetSize() <= kMaxIconSize) {
    base::Vector<u8> png(art.GetSize());
    if (art.Read(png.data(), png.size()) == png.size())
      host::ShowSplash(base::move(png));
  }
#endif
  base::SpawnDetachedThread(
      "guest-main",
      [main_module = base::move(main_module), mounted, is_ps5, sdk_version]() {
        auto p = base::MakeUnique<kern::Process>();
        if (is_ps5)
          p->SetPlatform(kern::Process::Platform::kPs5);
        p->SetSdkVersion(sdk_version);
        if (!p->Create(main_module, mounted))
          return;

        p->Start();
      });
}
