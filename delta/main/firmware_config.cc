#include "main/firmware_config.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "base/environment_variables.h"
#include "base/filesystem/file.h"
#include "base/option_file.h"
#include "base/standard_streams.h"
#include "base/strings/format.h"
#include "base/strings/string_compare.h"
#include "io/file.h"

namespace cli {
namespace {

bool ReportError(const base::String& message) {
  const auto text = base::Format("ps4delta: {}\n", message);
  base::WriteStandardError(text.data(), text.size());
  return false;
}

base::String ModuleDirectory() {
  base::StringU8 path;
  if (base::GetEnvironmentVariable(u8"XDG_DATA_HOME", path) && !path.empty() &&
      path.front() == '/') {
    return base::String(reinterpret_cast<const char*>(path.c_str())) +
           "/prosperity/modules";
  }
  if (base::GetEnvironmentVariable(u8"HOME", path) && !path.empty()) {
    return base::String(reinterpret_cast<const char*>(path.c_str())) +
           "/.local/share/prosperity/modules";
  }
  return {};
}

base::Path Utf8Path(const base::String& path) {
  return base::Path(
      base::StringRefU8(reinterpret_cast<const char8_t*>(path.c_str())));
}

bool CreateDirectories(const base::String& path) {
  for (mem_size i = 1; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/')
      continue;
    const auto directory = path.substr(0, i);
    if (::mkdir(directory.c_str(), 0700) != 0 && errno != EEXIST)
      return false;
  }
  return true;
}

bool HasModule(const base::String& directory, const char* name, bool ps5) {
  for (const char* extension : {".native.sprx", ".sprx"}) {
    if (!ps5 && base::StrEqual(extension, ".native.sprx"))
      continue;
    io::File file(directory + "/" + name + extension, io::FileMode::kRead);
    u8 header[20]{};
    if (file.IsOpen() && file.Read(header, sizeof(header)) == sizeof(header) &&
        header[0] == 0x7f && header[1] == 'E' && header[2] == 'L' &&
        header[3] == 'F' && header[4] == 2 && header[5] == 1 &&
        header[18] == 62 && header[19] == 0) {
      return true;
    }
  }
  return false;
}

bool ResolveModules(const base::String& input, bool ps5, base::String* output) {
  bool kernel = false;
  bool libc = false;
  mem_size begin = 0;
  do {
    const mem_size end = ps5 ? input.find(':', begin) : base::String::npos;
    const auto directory = input.substr(
        begin, end == base::String::npos ? base::String::npos : end - begin);
    char resolved[PATH_MAX];
    struct stat info{};
    if (directory.empty() || !::realpath(directory.c_str(), resolved) ||
        ::stat(resolved, &info) != 0 || !S_ISDIR(info.st_mode)) {
      return ReportError(base::Format("not a module directory: {}", directory));
    }
    if (base::String(resolved).find_first_of("\r\n") != base::String::npos)
      return ReportError("module paths cannot contain line breaks");
    if (!output->empty())
      *output += ":";
    *output += resolved;
    kernel |= HasModule(base::String(resolved), "libkernel", ps5);
    libc |= HasModule(base::String(resolved), "libSceLibcInternal", ps5);
    if (end == base::String::npos)
      break;
    begin = end + 1;
  } while (begin <= input.size());
  if (!kernel || !libc) {
    return ReportError(
        "module folders need decrypted x86-64 ELF files for libkernel and "
        "libSceLibcInternal (see docs/installation.md)");
  }
  return true;
}

bool CopyModule(const base::String& source, const base::String& destination) {
  base::File input(Utf8Path(source),
                   base::File::FLAG_OPEN | base::File::FLAG_READ);
  base::File output(Utf8Path(destination),
                    base::File::FLAG_CREATE | base::File::FLAG_WRITE);
  if (!input.IsValid() || !output.IsValid())
    return false;
  char buffer[65536];
  for (;;) {
    const int count = input.ReadAtCurrentPos(buffer, sizeof(buffer));
    if (count < 0)
      return false;
    if (count == 0)
      return true;
    if (output.WriteAtCurrentPos(buffer, count) != count)
      return false;
  }
}

bool CopyModules(const base::String& sources,
                 const base::String& destination,
                 bool ps5) {
  mem_size begin = 0;
  do {
    const mem_size end = ps5 ? sources.find(':', begin) : base::String::npos;
    const auto source = sources.substr(
        begin, end == base::String::npos ? base::String::npos : end - begin);
    DIR* directory = ::opendir(source.c_str());
    if (!directory)
      return false;
    bool copied = true;
    errno = 0;
    while (dirent* entry = ::readdir(directory)) {
      const base::String name(entry->d_name);
      if (!name.ends_with(".sprx"))
        continue;
      const auto input = source + "/" + name;
      const auto output = destination + "/" + name;
      struct stat info{};
      if (::stat(input.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) {
        copied = false;
        break;
      }
      // Earlier source directories win when filenames are duplicated.
      if (::stat(output.c_str(), &info) != 0 && !CopyModule(input, output)) {
        copied = false;
        break;
      }
      errno = 0;
    }
    const int read_error = errno;
    ::closedir(directory);
    if (!copied || read_error != 0)
      return false;
    if (end == base::String::npos)
      return true;
    begin = end + 1;
  } while (begin < sources.size());
  return true;
}

void RemoveImportedDirectory(const base::String& path) {
  DIR* directory = ::opendir(path.c_str());
  if (!directory)
    return;
  while (dirent* entry = ::readdir(directory)) {
    if (base::StrEqual(entry->d_name, ".") ||
        base::StrEqual(entry->d_name, ".."))
      continue;
    ::unlink((path + "/" + entry->d_name).c_str());
  }
  ::closedir(directory);
  ::rmdir(path.c_str());
}

bool ImportModules(const base::String& sources,
                   const base::String& destination,
                   bool ps5) {
  auto staging = destination + ".new-XXXXXX";
  if (!::mkdtemp(staging.data()))
    return ReportError(base::Format("cannot stage modules in {}", destination));
  if (!CopyModules(sources, staging, ps5) ||
      !HasModule(staging, "libkernel", ps5) ||
      !HasModule(staging, "libSceLibcInternal", ps5)) {
    RemoveImportedDirectory(staging);
    return ReportError(
        "cannot import module files; previous modules were kept");
  }

  const auto backup = staging + ".old";
  struct stat info{};
  const bool replacing = ::lstat(destination.c_str(), &info) == 0;
  if (replacing && (!S_ISDIR(info.st_mode) ||
                    ::rename(destination.c_str(), backup.c_str()) != 0)) {
    RemoveImportedDirectory(staging);
    return ReportError(base::Format("cannot replace {}", destination));
  }
  if (::rename(staging.c_str(), destination.c_str()) != 0) {
    if (replacing)
      ::rename(backup.c_str(), destination.c_str());
    RemoveImportedDirectory(staging);
    return ReportError(
        base::Format("cannot install modules in {}", destination));
  }
  if (replacing)
    RemoveImportedDirectory(backup);
  const auto message =
      base::Format("Imported firmware modules: {}\n", destination);
  base::WriteStandardOutput(message.data(), message.size());
  return true;
}

}  // namespace

bool ConfigureFirmware(const CommandLine& command) {
  const bool configuring =
      !command.configure_ps4_fw.empty() || !command.configure_ps5_fw.empty();
  const auto modules = ModuleDirectory();
  if (modules.empty())
    return !configuring ||
           ReportError("set HOME or XDG_DATA_HOME to store firmware modules");

  const auto ps4_directory = modules + "/ps4";
  const auto ps5_directory = modules + "/ps5";
  struct stat info{};
  if (::stat(ps4_directory.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
    base::SetOptionValue("DELTA_PS4_MODULES", ps4_directory);
  if (::stat(ps5_directory.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
    base::SetOptionValue("DELTA_PS5_MODULES", ps5_directory);

  base::String ps4;
  base::String ps5;
  if (!command.configure_ps4_fw.empty() &&
      !ResolveModules(command.configure_ps4_fw, false, &ps4))
    return false;
  if (!command.configure_ps5_fw.empty() &&
      !ResolveModules(command.configure_ps5_fw, true, &ps5))
    return false;
  if (!configuring)
    return true;
  if (!CreateDirectories(modules))
    return ReportError(base::Format("cannot create {}", modules));
  if (!ps4.empty()) {
    if (!ImportModules(ps4, ps4_directory, false))
      return false;
    base::SetOptionValue("DELTA_PS4_MODULES", ps4_directory);
  }
  if (!ps5.empty()) {
    if (!ImportModules(ps5, ps5_directory, true))
      return false;
    base::SetOptionValue("DELTA_PS5_MODULES", ps5_directory);
  }
  return true;
}

}  // namespace cli
