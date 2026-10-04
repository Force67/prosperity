#include "kern/save_storage.h"

#include <sys/stat.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#ifdef _WIN32
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "kern/lv2/error_table.h"

namespace kern::save_storage {
namespace {
std::mutex g_mutex;
std::map<io::File*, std::filesystem::path> g_open_files;
struct FileInfo {
  std::pair<u64, u64> identity;
  u64 size, links;
};
bool StatFile(FILE* stream, FileInfo& info) {
#ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION status;
  const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream)));
  if (!GetFileInformationByHandle(handle, &status))
    return false;
  info = {{status.dwVolumeSerialNumber,
           (u64(status.nFileIndexHigh) << 32) | status.nFileIndexLow},
          (u64(status.nFileSizeHigh) << 32) | status.nFileSizeLow,
          status.nNumberOfLinks};
#else
  struct stat status;
  if (fstat(fileno(stream), &status))
    return false;
  info = {{status.st_dev, status.st_ino},
          static_cast<u64>(status.st_size),
          status.st_nlink};
#endif
  return true;
}
bool StatFile(const std::filesystem::path& path, FileInfo& info) {
#ifdef _WIN32
  FILE* stream = _wfopen(path.c_str(), L"rb");
  if (!stream)
    return false;
  const bool result = StatFile(stream, info);
  std::fclose(stream);
  return result;
#else
  struct stat status;
  if (stat(path.c_str(), &status))
    return false;
  info = {{status.st_dev, status.st_ino},
          static_cast<u64>(status.st_size),
          status.st_nlink};
  return true;
#endif
}
bool Track(io::File& file, const base::String& root) {
  if (root.empty())
    return true;
  std::error_code error;
  const auto canonical = std::filesystem::canonical(root.c_str(), error);
  FileInfo info;
  if (error || !StatFile(static_cast<FILE*>(file.GetNativeHandle()), info))
    return false;
  g_open_files[&file] = canonical;
  return true;
}
constexpr u64 kBlockSize = 32768;
u64 Blocks(u64 size) {
  return size / kBlockSize + (size % kBlockSize != 0);
}
i32 InfoUnlocked(const base::String& host, u64& total, u64& free) {
  total = 60000;
  const auto quota = host + "/.sce_quota.bin";
  if (FILE* file = std::fopen(quota.c_str(), "rb")) {
    const auto read = std::fread(&total, 1, 8, file);
    const bool failed =
        read != 8 || std::fgetc(file) != EOF || std::ferror(file);
    std::fclose(file);
    if (failed || total < 96)
      return -static_cast<i32>(SysError::eIO);
  } else if (errno != ENOENT) {
    return -static_cast<i32>(SysError::eIO);
  }
  u64 used = 0;
  std::set<std::pair<u64, u64>> counted;
  const auto account = [&](const FileInfo& info) {
    if (counted.insert(info.identity).second)
      used += std::min(Blocks(info.size), total - used);
  };
  std::error_code error;
  std::filesystem::recursive_directory_iterator files(host.c_str(), error), end;
  if (error)
    return -static_cast<i32>(SysError::eIO);
  for (; files != end; files.increment(error)) {
    if (error)
      return -static_cast<i32>(SysError::eIO);
    const auto status = files->symlink_status(error);
    if (error)
      return -static_cast<i32>(SysError::eIO);
    if (std::filesystem::is_regular_file(status) &&
        files->path() != std::filesystem::path(quota.c_str()) &&
        files->path() != std::filesystem::path((quota + ".tmp").c_str())) {
      FileInfo info;
      if (!StatFile(files->path(), info))
        return -static_cast<i32>(SysError::eIO);
      account(info);
    }
  }
  if (error)
    return -static_cast<i32>(SysError::eIO);
  const auto canonical = std::filesystem::canonical(host.c_str(), error);
  if (error)
    return -static_cast<i32>(SysError::eIO);
  for (const auto& [file, root] : g_open_files) {
    if (root != canonical)
      continue;
    FileInfo info;
    if (!StatFile(static_cast<FILE*>(file->GetNativeHandle()), info))
      return -static_cast<i32>(SysError::eIO);
    account(info);
  }
  free = total - used;
  return 0;
}

i32 Limit(const base::String& root, u64 size, u64& limit) {
  u64 total, free;
  if (const auto result = InfoUnlocked(root, total, free))
    return result;
  const u64 current = Blocks(size);
  constexpr u64 maximum = std::numeric_limits<i64>::max();
  limit =
      current >= maximum / kBlockSize || free > maximum / kBlockSize - current
          ? maximum
          : (current + free) * kBlockSize;
  return 0;
}

i32 ReplaceLimit(const base::String& root,
                 const base::String& path,
                 u64& limit) {
  FileInfo target{};
  if (!StatFile(std::filesystem::path(path.c_str()), target)) {
    if (errno != ENOENT)
      return -static_cast<i32>(SysError::eIO);
  } else {
    bool retained = target.links > 1;
    for (const auto& [file, ignored] : g_open_files) {
      FileInfo info;
      if (!StatFile(static_cast<FILE*>(file->GetNativeHandle()), info))
        return -static_cast<i32>(SysError::eIO);
      retained |= info.identity == target.identity;
    }
    if (retained)
      target.size = 0;
  }
  return Limit(root, target.size, limit);
}

bool Confined(const base::String& root, const base::String& path) {
  if (root.empty())
    return true;
  std::error_code error;
  const auto base = std::filesystem::canonical(root.c_str(), error);
  if (error)
    return false;
  const auto target = std::filesystem::weakly_canonical(path.c_str(), error);
  if (error)
    return false;
  auto element = target.begin();
  for (const auto& part : base) {
    if (element == target.end() || *element++ != part)
      return false;
  }
  if (element == target.end())
    return false;
  return target != base / ".sce_quota.bin" &&
         target != base / ".sce_quota.bin.tmp";
}
i32 TruncateUnlocked(io::File& file, const base::String& root, i64 length) {
  if (length < 0)
    return -static_cast<i32>(SysError::eINVAL);
  if (!root.empty() && static_cast<u64>(length) > file.GetSize()) {
    u64 limit;
    if (const auto result = Limit(root, file.GetSize(), limit))
      return result;
    if (static_cast<u64>(length) > limit)
      return -static_cast<i32>(SysError::eNOSPC);
  }
  file.Flush();
  const u64 position = file.Tell();
  auto* stream = static_cast<FILE*>(file.GetNativeHandle());
#ifdef _WIN32
  const int error = _chsize_s(_fileno(stream), length);
#else
  const int error = ftruncate(fileno(stream), length) ? errno : 0;
#endif
  file.Seek(position, io::SeekMode::kSeekSet);
  if (error == ENOSPC)
    return -static_cast<i32>(SysError::eNOSPC);
  if (error == EFBIG)
    return -static_cast<i32>(SysError::eFBIG);
  return error ? -static_cast<i32>(SysError::eIO) : 0;
}
}  // namespace

i32 Info(const base::String& root, u64& total, u64& free) {
  std::lock_guard lock(g_mutex);
  return InfoUnlocked(root, total, free);
}

bool OpenWritable(io::File& file,
                  const base::String& root,
                  const base::String& path,
                  bool create,
                  bool truncate) {
  std::lock_guard lock(g_mutex);
  if (!Confined(root, path))
    return false;
  io::File opened(path, io::FileMode::kReadWrite, false);
  if (!opened.IsOpen()) {
    if (!create)
      return false;
    std::error_code error;
    const bool exists = std::filesystem::exists(path.c_str(), error);
    if (exists || error)
      return false;
    opened.Reset(io::File(path, io::FileMode::kCreate, false).GetBase());
    if (!opened.IsOpen())
      return false;
  }
  if (truncate && TruncateUnlocked(opened, root, 0))
    return false;
  file.Reset(opened.GetBase());
  return Track(file, root);
}

bool OpenRead(io::File& file,
              const base::String& root,
              const base::String& path) {
  std::lock_guard lock(g_mutex);
  if (!Confined(root, path))
    return false;
  io::File opened(path, io::FileMode::kRead, false);
  if (!opened.IsOpen())
    return false;
  file.Reset(opened.GetBase());
  return Track(file, root);
}

void Close(io::File& file) {
  std::lock_guard lock(g_mutex);
  g_open_files.erase(&file);
  file.Close();
}

i64 Write(io::File& file,
          const base::String& root,
          const void* data,
          size_t size) {
  std::lock_guard lock(g_mutex);
  if (!size)
    return 0;
  const u64 position = file.Tell();
  constexpr u64 maximum = std::numeric_limits<i64>::max();
  if (position > maximum || size > maximum - position)
    return -static_cast<i32>(SysError::eFBIG);
  if (!root.empty()) {
    u64 limit;
    if (const auto result = Limit(root, file.GetSize(), limit))
      return result;
    if (position + size > limit) {
      if (position >= limit)
        return -static_cast<i32>(SysError::eNOSPC);
      size = std::min<u64>(size, limit - position);
    }
  }
  if (!file.Write(data, size))
    return 0;
  file.Flush();
  return size;
}

i32 Truncate(io::File& file, const base::String& root, i64 length) {
  std::lock_guard lock(g_mutex);
  return TruncateUnlocked(file, root, length);
}

i32 Truncate(const base::String& root, const base::String& path, i64 length) {
  std::lock_guard lock(g_mutex);
  if (length < 0)
    return -static_cast<i32>(SysError::eINVAL);
  if (!Confined(root, path))
    return -static_cast<i32>(SysError::eACCES);
  std::error_code error;
  const auto status = std::filesystem::status(path.c_str(), error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(status)))
    return -static_cast<i32>(SysError::eNOENT);
  if (error)
    return -static_cast<i32>(SysError::eIO);
  if (std::filesystem::is_directory(status))
    return -static_cast<i32>(SysError::eISDIR);
  if (!std::filesystem::is_regular_file(status))
    return -static_cast<i32>(SysError::eINVAL);
  io::File file(path, io::FileMode::kReadWrite);
  if (!file.IsOpen())
    return -static_cast<i32>(errno == EACCES ? SysError::eACCES
                                             : SysError::eIO);
  return TruncateUnlocked(file, root, length);
}

i32 Replace(const base::String& root,
            const base::String& path,
            const void* data,
            size_t size) {
  std::lock_guard lock(g_mutex);
  if (path != root + "/.sce_quota.bin") {
    u64 limit;
    if (const auto result = ReplaceLimit(root, path, limit))
      return result;
    if (size > limit)
      return -static_cast<i32>(SysError::eNOSPC);
  }
  const auto temporary = path + ".tmp";
  std::ofstream file(temporary.c_str(), std::ios::binary | std::ios::trunc);
  if (!file)
    return -static_cast<i32>(SysError::eIO);
  file.write(static_cast<const char*>(data), size);
  file.close();
  if (!file) {
    std::error_code ignored;
    std::filesystem::remove(temporary.c_str(), ignored);
    return -static_cast<i32>(SysError::eIO);
  }
  std::error_code error;
  std::filesystem::rename(temporary.c_str(), path.c_str(), error);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(temporary.c_str(), ignored);
  }
  return error ? -static_cast<i32>(SysError::eIO) : 0;
}
i32 Remove(const base::String& root, const base::String& path) {
  std::lock_guard lock(g_mutex);
  if (!Confined(root, path))
    return -static_cast<i32>(SysError::eACCES);
  std::error_code error;
  if (std::filesystem::is_directory(path.c_str(), error))
    return -static_cast<i32>(SysError::eISDIR);
  error.clear();
  const bool removed = std::filesystem::remove(path.c_str(), error);
  if (error)
    return -static_cast<i32>(SysError::eIO);
  return removed ? 0 : -static_cast<i32>(SysError::eNOENT);
}

i32 RemoveSave(const base::String& root) {
  std::lock_guard lock(g_mutex);
  std::error_code error;
  const auto status = std::filesystem::symlink_status(root.c_str(), error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(status)))
    return -static_cast<i32>(SysError::eNOENT);
  if (error)
    return -static_cast<i32>(SysError::eIO);
  if (!std::filesystem::is_directory(status))
    return -static_cast<i32>(SysError::eINVAL);
  const auto canonical = std::filesystem::canonical(root.c_str(), error);
  if (error)
    return -static_cast<i32>(SysError::eIO);
  for (const auto& [file, host] : g_open_files)
    if (host == canonical)
      return -static_cast<i32>(SysError::eBUSY);
  const auto backup = root + ".backup", retiring = root + ".delete";
  for (const auto& path : {retiring, root + ".tmp", root + ".old",
                           backup + ".tmp", backup + ".old"}) {
    const auto status = std::filesystem::symlink_status(path.c_str(), error);
    if (!error && std::filesystem::exists(status))
      return -static_cast<i32>(SysError::eBUSY);
    if (error && error != std::errc::no_such_file_or_directory)
      return -static_cast<i32>(SysError::eIO);
    error.clear();
  }
  const auto backup_status =
      std::filesystem::symlink_status(backup.c_str(), error);
  const bool has_backup = !error && std::filesystem::exists(backup_status);
  if (error && error != std::errc::no_such_file_or_directory)
    return -static_cast<i32>(SysError::eIO);
  if (has_backup) {
    if (!std::filesystem::is_directory(backup_status))
      return -static_cast<i32>(SysError::eACCES);
    error.clear();
    const auto backup_root =
        std::filesystem::canonical(backup.c_str(), error);
    if (error)
      return -static_cast<i32>(SysError::eIO);
    for (const auto& [file, host] : g_open_files)
      if (host == backup_root)
        return -static_cast<i32>(SysError::eBUSY);
  }
  error.clear();
  if (!std::filesystem::create_directory(retiring.c_str(), error))
    return -static_cast<i32>(error ? SysError::eIO : SysError::eBUSY);
  std::filesystem::rename(root.c_str(), (retiring + "/save").c_str(), error);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(retiring.c_str(), ignored);
    return -static_cast<i32>(SysError::eIO);
  }
  if (has_backup) {
    std::filesystem::rename(backup.c_str(), (retiring + "/backup").c_str(),
                            error);
    if (error) {
      std::error_code ignored;
      std::filesystem::rename((retiring + "/save").c_str(), root.c_str(),
                              ignored);
      std::filesystem::remove(retiring.c_str(), ignored);
      return -static_cast<i32>(SysError::eIO);
    }
  }
  std::filesystem::remove_all(retiring.c_str(), error);
  return 0;
}

namespace {
i32 StorageError(const std::error_code& error) {
  return error == std::errc::no_space_on_device
             ? -static_cast<i32>(SysError::eNOSPC)
             : -static_cast<i32>(SysError::eIO);
}
i32 ValidateSave(const base::String& root) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(root.c_str(), error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(status)))
    return -static_cast<i32>(SysError::eNOENT);
  if (error)
    return StorageError(error);
  if (!std::filesystem::is_directory(status))
    return -static_cast<i32>(SysError::eACCES);
  std::filesystem::recursive_directory_iterator it(root.c_str(), error), end;
  for (; !error && it != end; it.increment(error)) {
    const auto entry = it->symlink_status(error);
    if (error)
      break;
    if (!std::filesystem::is_directory(entry) &&
        !std::filesystem::is_regular_file(entry))
      return -static_cast<i32>(SysError::eACCES);
  }
  if (error)
    return StorageError(error);
  u64 total, free;
  return InfoUnlocked(root, total, free);
}
i32 CopySave(const base::String& source, const base::String& destination) {
  if (const auto result = ValidateSave(source))
    return result;
  std::error_code error;
  const auto canonical =
      std::filesystem::weakly_canonical(destination.c_str(), error);
  if (error)
    return StorageError(error);
  const auto canonical_source =
      std::filesystem::canonical(source.c_str(), error);
  if (error)
    return StorageError(error);
  for (const auto& [file, host] : g_open_files)
    if (host == canonical || host == canonical_source)
      return -static_cast<i32>(SysError::eBUSY);
  const auto staging = destination + ".tmp", previous = destination + ".old";
  for (const auto& path : {staging, previous}) {
    const auto status = std::filesystem::symlink_status(path.c_str(), error);
    if (!error && std::filesystem::exists(status))
      return -static_cast<i32>(SysError::eBUSY);
    if (error && error != std::errc::no_such_file_or_directory)
      return StorageError(error);
    error.clear();
  }
  const auto status =
      std::filesystem::symlink_status(destination.c_str(), error);
  const bool exists = !error && std::filesystem::exists(status);
  if (error && error != std::errc::no_such_file_or_directory)
    return StorageError(error);
  if (exists) {
    if (const auto result = ValidateSave(destination))
      return result;
  }
  error.clear();
  std::filesystem::copy(source.c_str(), staging.c_str(),
                        std::filesystem::copy_options::recursive, error);
  if (!error && exists)
    std::filesystem::rename(destination.c_str(), previous.c_str(), error);
  if (!error)
    std::filesystem::rename(staging.c_str(), destination.c_str(), error);
  if (error) {
    const auto result = StorageError(error);
    std::error_code ignored;
    if (exists && !std::filesystem::exists(destination.c_str(), ignored))
      std::filesystem::rename(previous.c_str(), destination.c_str(), ignored);
    std::filesystem::remove_all(staging.c_str(), ignored);
    return result;
  }
  if (exists)
    std::filesystem::remove_all(previous.c_str(), error);
  return 0;
}
}  // namespace

i32 Backup(const base::String& root) {
  std::lock_guard lock(g_mutex);
  return CopySave(root, root + ".backup");
}
i32 CheckBackup(const base::String& root) {
  std::lock_guard lock(g_mutex);
  return ValidateSave(root + ".backup");
}
i32 RestoreBackup(const base::String& root) {
  std::lock_guard lock(g_mutex);
  return CopySave(root + ".backup", root);
}

i32 Rename(const base::String& root,
           const base::String& from,
           const base::String& to) {
  std::lock_guard lock(g_mutex);
  if (!Confined(root, from) || !Confined(root, to))
    return -static_cast<i32>(SysError::eACCES);
  std::error_code error;
  std::filesystem::rename(from.c_str(), to.c_str(), error);
  if (error == std::errc::no_such_file_or_directory)
    return -static_cast<i32>(SysError::eNOENT);
  return error ? -static_cast<i32>(SysError::eIO) : 0;
}

bool MakeDir(const base::String& root, const base::String& path) {
  std::lock_guard lock(g_mutex);
  if (!Confined(root, path))
    return false;
  std::error_code error;
  std::filesystem::create_directories(path.c_str(), error);
  return !error;
}
}  // namespace kern::save_storage
