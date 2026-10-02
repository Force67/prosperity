#include "main/game_firmware.h"

#include "base/strings/format.h"
#include "formats/archive_filesystem.h"
#include "formats/title_metadata.h"
#include "io/file.h"
#include "kern/vfs_providers.h"
#include "main/firmware_config.h"
#include "main/game_directory.h"

namespace cli {
base::String CheckGameFirmware(const base::String& path) {
  base::String json;
  base::String extension = path.substr(path.find_last_of('.') + 1);
  for (char& c : extension) {
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  }
  if (formats::IsArchivePath(path.c_str()) || extension == "ffpkg") {
    auto mount = formats::IsArchivePath(path.c_str())
                     ? kern::vfs::MountArchive(path, false)
                     : kern::vfs::MountFfpkg(path, false);
    if (!mount || !mount.is_ps5)
      return {};
    auto file = mount.provider->Open("/sce_sys/param.json");
    if (!file || file->Size() <= 0 || file->Size() > (1u << 20))
      return {};
    json.resize(file->Size());
    if (file->Read(json.data(), 0, json.size()) != i64(json.size()))
      return {};
  } else {
    const auto directory = FindGameDirectory(path);
    if (directory.root.empty())
      return {};
    io::File file(directory.root + "/sce_sys/param.json", io::FileMode::kRead);
    if (!file.IsOpen() || file.GetSize() == 0 || file.GetSize() > (1u << 20))
      return {};
    json.resize(file.GetSize());
    if (file.Read(json.data(), json.size()) != json.size())
      return {};
  }
  const u32 required = formats::ParseSdkVersion(
      formats::JsonGetString(json, "requiredSystemSoftwareVersion"));
  if (!required || Ps5FirmwareVersion() >= required)
    return {};
  return base::Format(
      "Install PS5 firmware {:02x}.{:02x} or later to play this game.",
      required >> 24, (required >> 16) & 0xff);
}
}  // namespace cli
