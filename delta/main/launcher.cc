/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "main/launcher.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include "cpu/backend.h"
#include "guest/pause.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "kern/guest_va_space.h"

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/environment_variables.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/thread.h"
#include "formats/archive_filesystem.h"
#include "formats/title_metadata.h"
#include "gpu/ps4/cmd_processor.h"
#include "gpu/render/renderer.h"
#include "host/audio_output.h"
#include "host/window.h"
#include "io/file.h"
#include "kern/crash.h"
#include "kern/probe/probe_arm.h"
#include "kern/ps4/audio_sink.h"
#include "kern/ps4/hardware_mode.h"
#include "kern/vfs.h"
#include "kern/vfs_providers.h"
#include "kern/vm_map.h"
#include "logger/logger.h"
#include "main/game_directory.h"
#include "options/options.h"
#include "ui/pause_menu.h"
#if defined(__linux__) && !defined(__ANDROID__)
#include "main/home_screen.h"
#include "main/recent_games.h"
#endif

namespace {

DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);

#if defined(__linux__) && !defined(__ANDROID__)
constexpr bool kWantIcon = true;
#else
constexpr bool kWantIcon = false;
#endif

constexpr u64 kMaxMetadataSize = 1u << 20;
constexpr u64 kMaxImageSize = 16u << 20;

struct BootTitle {
  base::String main_module;
  base::String title_id;
  base::String name;
  base::Vector<u8> icon;
  u32 sdk_version = 0;
  u32 attributes = 0;
  bool mounted = false;
  bool is_ps5 = false;
};

bool IsHostFileReadable(const base::String& path) {
  // io::File::Exists() only tests whether a backing object was allocated.
  return io::File(path, io::FileMode::kRead).IsOpen();
}

bool ReadHostFile(const base::String& path,
                  u64 max_size,
                  base::Vector<u8>* out) {
  io::File file(path, io::FileMode::kRead);
  if (!file.IsOpen())
    return false;
  const u64 size = file.GetSize();
  if (size == 0 || size > max_size)
    return false;
  out->resize(static_cast<mem_size>(size));
  if (file.Read(out->data(), out->size()) != size) {
    out->clear();
    return false;
  }
  return true;
}

base::String ParentPath(const base::String& path) {
  const mem_size slash = path.find_last_of("/\\", base::String::npos, 2);
  if (slash == base::String::npos)
    return base::String(".");
  return path.substr(0, slash == 0 ? 1 : slash);
}

bool EndsWithIgnoreCase(const base::String& path, const char* extension) {
  const mem_size length = base::CountStringLength(extension);
  if (path.size() < length)
    return false;
  const char* suffix = path.c_str() + path.size() - length;
  for (mem_size i = 0; i < length; ++i) {
    char c = suffix[i];
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
    if (c != extension[i])
      return false;
  }
  return true;
}

bool ReadPs4Metadata(const base::String& path, BootTitle* title) {
  base::Vector<u8> sfo;
  if (!ReadHostFile(path, kMaxMetadataSize, &sfo))
    return false;
  title->title_id = formats::SfoGet(sfo.data(), sfo.size(), "TITLE_ID");
  title->name = formats::SfoGet(sfo.data(), sfo.size(), "TITLE");
  title->attributes = formats::SfoGetU32(sfo.data(), sfo.size(), "ATTRIBUTE");
  return true;
}

void ReadPs5Metadata(const base::String& path, BootTitle* title) {
  base::Vector<u8> bytes;
  if (!ReadHostFile(path, kMaxMetadataSize, &bytes))
    return;
  const base::String json(reinterpret_cast<const char*>(bytes.data()),
                          bytes.size());
  title->title_id = formats::JsonGetString(json, "titleId");
  title->name = formats::JsonGetTitleName(json);
  title->sdk_version =
      formats::ParseSdkVersion(formats::JsonGetString(json, "sdkVersion"));
}

void ReadTitleIcon(const base::String& root, BootTitle* title) {
  if (!kWantIcon)
    return;
  if (!ReadHostFile(root + "/sce_sys/icon0.png", kMaxImageSize, &title->icon))
    ReadHostFile(root + "/icon0.png", kMaxImageSize, &title->icon);
}

bool MountContainer(kern::vfs::TitleMount mount, BootTitle* title) {
  if (!mount)
    return false;
  kern::vfs::MountVirtual("/app0", mount.provider);
  title->title_id = base::move(mount.title_id);
  title->name = base::move(mount.title);
  title->icon = base::move(mount.icon);
  title->attributes = mount.attributes;
  title->sdk_version = mount.sdk_version;
  title->is_ps5 = mount.is_ps5;
  title->mounted = true;
  title->main_module =
      mount.has_decrypted ? "/app0/decrypted/eboot.bin" : "/app0/eboot.bin";
  return true;
}

bool LoadTitle(base::String& path, BootTitle* title) {
  if (EndsWithIgnoreCase(path, ".pkg"))
    return MountContainer(kern::vfs::MountPkg(path, kWantIcon), title);
  if (EndsWithIgnoreCase(path, ".ffpkg"))
    return MountContainer(kern::vfs::MountFfpkg(path, kWantIcon), title);
  if (formats::IsArchivePath(path.c_str()))
    return MountContainer(kern::vfs::MountArchive(path, kWantIcon), title);

  const auto directory = cli::FindGameDirectory(path);
  if (directory.root.empty()) {
    const base::String root = ParentPath(path);
    if (!ReadPs4Metadata(root + "/sce_sys/param.sfo", title))
      ReadPs4Metadata(root + "/param.sfo", title);
    ReadTitleIcon(root, title);
    title->main_module = path;
    return true;
  }
  path = directory.root;
  const base::String sfo_path = path + "/sce_sys/param.sfo";
  const base::String json_path = path + "/sce_sys/param.json";
  if (IsHostFileReadable(sfo_path)) {
    ReadPs4Metadata(sfo_path, title);
  } else if (IsHostFileReadable(json_path)) {
    title->is_ps5 = true;
    ReadPs5Metadata(json_path, title);
  }

  kern::vfs::Mount("/app0", path.c_str());
  title->mounted = true;
  title->main_module = directory.main_module;
  ReadTitleIcon(path, title);
  return true;
}

void MountDownloadStorage(const base::String& title_id) {
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  const auto path = base::Format(
      "{}/.prosperity/download/{}",
      home.empty() ? "." : reinterpret_cast<const char*>(home.c_str()),
      title_id.empty() ? "UNKNOWN" : title_id.c_str());
  kern::vfs::MountWritable("/download0", path.c_str());
}

void SetWindowTitle(const BootTitle& title) {
  const auto window_title =
      base::Format("prosperity - {} - [{}] ({})",
                   title.name.empty() ? "unknown" : title.name.c_str(),
                   title.title_id.empty() ? "unknown" : title.title_id.c_str(),
                   title.is_ps5 ? "PS5" : "PS4");
  LOG_INFO("window title: {}", window_title.c_str());
  host::SetTitle(window_title.c_str());
  ui::PauseMenuSetGameTitle(title.name);
}

#if defined(__linux__) && !defined(__ANDROID__)
base::Vector<u8> ReadTitleArtwork(const BootTitle& title,
                                  const base::String& path) {
  base::Vector<u8> png;
  for (const char* filename : {"pic0.png", "pic1.png"}) {
    const auto relative = base::String("/sce_sys/") + filename;
    io::File art =
        title.mounted
            ? kern::vfs::OpenRead(("/app0" + relative).c_str())
            : io::File(ParentPath(path) + relative, io::FileMode::kRead);
    if (!art.Exists() || !art.IsOpen() || art.GetSize() == 0 ||
        art.GetSize() > kMaxImageSize)
      continue;
    png.resize(art.GetSize());
    if (art.Read(png.data(), png.size()) == png.size())
      break;
    png.clear();
  }
  return png;
}

void SetWindowArtwork(const BootTitle& title, const base::String& path) {
  if (!title.icon.empty())
    host::SetIcon(title.icon.data(), title.icon.size());
  auto png = ReadTitleArtwork(title, path);
  cli::RememberGame(path, title.name, title.title_id, title.is_ps5, title.icon,
                    png);
  if (!png.empty())
    host::ShowSplash(base::move(png));
}
#endif


}  // namespace

Launcher::Launcher() = default;
Launcher::~Launcher() = default;

bool Launcher::Init() {
  LOG_INFO("Initializing prosperity " rsc_copyright);
  // Wire host and GPU callbacks here to preserve the kernel's module
  // boundaries.
  gpu::ps4::SetWriteWatchCallback(&kern::probe::StartWriteWatch);
  kern::SetCsRangeDescriber(&gpu::render::DescribeCsRangeCovering);
  kern::SetMappingChangedHook(&gpu::render::NoteGuestRemap);
  kern::ps4::SetAudioSink({host::OpenAudioPort, host::QueueAudio,
                           host::SetAudioPortVolume, host::CloseAudioPort});
  return true;
}

#if defined(__linux__) && !defined(__ANDROID__)
base::String cli::AddHomeGame(const base::String& game_path) {
  base::String path = game_path;
  BootTitle title;
  if (!LoadTitle(path, &title))
    return "Could not read this game.";
  RememberGame(path, title.name, title.title_id, title.is_ps5, title.icon,
               ReadTitleArtwork(title, path));
  return {};
}
#endif

void Launcher::Boot(const base::String& game_path) {
  guest::BeginSession();
  host_memory::BeginMemorySession();
  options::BeginGameSession();
  base::String path = game_path;
#ifdef _WIN32
  for (auto& c : path) {
    if (c == '/')
      c = '\\';
  }
#endif

  BootTitle title;
  if (!LoadTitle(path, &title)) {
    guest::RequestStop();
    return;
  }
  kern::vfs::SetTitleId(title.title_id);
  if (title.mounted) {
    LOG_INFO("mounted title at /app0 ({}), boot module {}",
             title.title_id.c_str(), title.main_module.c_str());
    MountDownloadStorage(title.title_id);
  }

  // Profiles must apply before platform settings and guest startup read
  // options.
  options::LoadGameProfile(title.title_id.c_str());
  gpu::render::Init(gpu::render::DefaultRenderer());
  kern::ps4::SetTitleAttributes(title.is_ps5 ? 0 : title.attributes);
  gpu::ps4::SetPs4NeoMode(!title.is_ps5 && kern::ps4::IsNeoMode());
  SetWindowTitle(title);
#if defined(__linux__) && !defined(__ANDROID__)
  SetWindowArtwork(title, path);
#endif
  proc_ = base::MakeUnique<kern::Process>();
  if (title.is_ps5)
    proc_->SetPlatform(kern::Process::Platform::kPs5);
  proc_->SetSdkVersion(title.sdk_version);
  if (!guest::SpawnThread("guest-main",
                          [this, main_module = base::move(title.main_module),
                           mounted = title.mounted] {
                            if (!proc_->Create(main_module, mounted)) {
                              guest::RequestStop();
                              return;
                            }
                            if (!guest::Stopping())
                              proc_->Start();
                          }))
    guest::RequestStop();
}

void Launcher::Stop() {
  guest::RequestStop();
  gpu::render::StopPresentation();
  guest::JoinThreads();
  gpu::ps4::StopRenderQueue();
  host::CloseAllAudioPorts();
  gpu::render::ResetSession(gpu::render::DefaultRenderer());
  cpu::GetBackend().ResetSession();
  guest::ResetSessionResources();
  proc_ = {};
  guest::ResetCodeRanges();
  const auto released = host_memory::EndMemorySession();
  kern::RestoreGuestVaSpace();
  options::EndGameSession();
  host::ResetGuest();
  ui::PauseMenuReset();
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
  LOG_INFO("guest stopped: {} threads, released {} MiB of guest address space",
           guest::LiveThreads(), released >> 20);
}
