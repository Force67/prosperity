#include "libSceSaveData.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>

#include "kern/vfs.h"

namespace {

// Mount modes (SCE_SAVE_DATA_MOUNT_MODE_*).
constexpr uint32_t kModeRdOnly = 1;
constexpr uint32_t kModeRdWr = 2;
constexpr uint32_t kModeCreate = 4;
constexpr uint32_t kModeCreate2 = 32;

// Errors (SCE_SAVE_DATA_ERROR_*).
constexpr int kOk = 0;
constexpr int kErrParameter = static_cast<int>(0x809F0000u);
constexpr int kErrExists = static_cast<int>(0x809F0007u);
constexpr int kErrNotFound = static_cast<int>(0x809F0008u);

std::mutex g_mtx;
int g_nextSlot = 0;

bool g_trace() {
  static const bool on = std::getenv("DELTA_SAVE_TRACE") != nullptr;
  return on;
}

// Host directory that holds every title's saves. Games address a save by its
// dirName (e.g. "PTSaveData"); one host subdirectory per dirName.
std::string saveRoot() {
  if (const char *e = std::getenv("DELTA_SAVEDATA_DIR"))
    return e;
  const char *home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.prosperity/savedata";
}

bool hostDirExists(const std::string &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}

// Shared mount core. `dir_name` is the save's directory name; `mode` the mount
// mode bits; `result` an OrbisSaveDataMountResult (may be null). Returns an SCE
// error/OK. Registers a writable VFS mount at /savedataN -> host save dir so the
// game's subsequent file I/O under the mount point lands on the host.
int doMount(const char *dir_name, uint32_t mode, void *result) {
  if (!dir_name || !dir_name[0])
    return kErrParameter;
  const std::string host = saveRoot() + "/" + dir_name;
  const bool exists = hostDirExists(host);
  const bool create = (mode & (kModeCreate | kModeCreate2)) != 0;

  if (!exists && !create)
    return kErrNotFound;  // e.g. a read-only "does a save exist?" check
  if (exists && (mode & kModeCreate))
    return kErrExists;  // strict CREATE requires the save not to exist yet

  std::lock_guard<std::mutex> lk(g_mtx);
  char mount_point[16];
  std::snprintf(mount_point, sizeof(mount_point), "/savedata%d", g_nextSlot++);
  // Registers the mount and creates the host directory if absent.
  krnl::vfs::mountWritable(mount_point, host.c_str());

  if (result) {
    auto *r = static_cast<uint8_t *>(result);
    std::memset(r, 0, 64);
    std::snprintf(reinterpret_cast<char *>(r), 16, "%s", mount_point);
    const uint32_t status = exists ? 0u : 1u;  // 1 = SAVE_DATA_CREATED
    std::memcpy(r + 28, &status, 4);
  }
  if (g_trace())
    std::fprintf(stderr,
                 "[savedata] mount dir='%s' mode=%#x -> %s (host=%s, %s)\n",
                 dir_name, mode, mount_point, host.c_str(),
                 exists ? "existing" : "created");
  return kOk;
}

// Read the char[] a DirName* points at (the struct is just { char data[32] }).
const char *dirNameOf(const void *dir_name_ptr) {
  return static_cast<const char *>(dir_name_ptr);
}

}  // namespace

int PS4ABI sceSaveDataInitialize(void *) { return kOk; }
int PS4ABI sceSaveDataInitialize2(void *) { return kOk; }
int PS4ABI sceSaveDataInitialize3(void *) { return kOk; }
int PS4ABI sceSaveDataTerminate() { return kOk; }

int PS4ABI sceSaveDataMount2(const void *mount, void *result) {
  if (!mount)
    return kErrParameter;
  const auto *p = static_cast<const uint8_t *>(mount);
  const void *dir_name_ptr = nullptr;
  uint32_t mode = 0;
  std::memcpy(&dir_name_ptr, p + 8, sizeof(dir_name_ptr));
  std::memcpy(&mode, p + 24, 4);
  return doMount(dirNameOf(dir_name_ptr), mode, result);
}

int PS4ABI sceSaveDataMount(const void *mount, void *result) {
  if (!mount)
    return kErrParameter;
  const auto *p = static_cast<const uint8_t *>(mount);
  const void *dir_name_ptr = nullptr;
  uint32_t mode = 0;
  std::memcpy(&dir_name_ptr, p + 16, sizeof(dir_name_ptr));  // Mount v1: +16
  std::memcpy(&mode, p + 40, 4);                             // Mount v1: +40
  return doMount(dirNameOf(dir_name_ptr), mode, result);
}

int PS4ABI sceSaveDataMount5(const void *mount, void *result) {
  return sceSaveDataMount2(mount, result);  // same leading layout for our use
}

int PS4ABI sceSaveDataUmount(const void *) { return kOk; }
int PS4ABI sceSaveDataUmountWithBackup(const void *) { return kOk; }

int PS4ABI sceSaveDataGetMountInfo(const void *, void *info) {
  if (info) {
    auto *i = static_cast<uint8_t *>(info);
    std::memset(i, 0, 48);
    const uint64_t blocks = 1u << 20;  // ~32 GiB of 32 KiB blocks: plenty free
    std::memcpy(i + 0, &blocks, 8);    // total blocks
    std::memcpy(i + 8, &blocks, 8);    // free blocks
  }
  return kOk;
}

int PS4ABI sceSaveDataDirNameSearch(const void *, void *result) {
  // Report no matches (hitNum / dirNamesNum / setNum = 0). Titles that load a
  // known save mount it directly; enumeration returning empty just means "no
  // prior saves", which is correct for a fresh host save directory.
  if (result) {
    auto *r = static_cast<uint8_t *>(result);
    const uint32_t zero = 0;
    std::memcpy(r + 0, &zero, 4);   // hitNum
    std::memcpy(r + 16, &zero, 4);  // dirNamesNum
    std::memcpy(r + 20, &zero, 4);  // setNum
  }
  return kOk;
}

int PS4ABI sceSaveDataGetParam(const void *, int, void *buf, uint64_t size,
                               uint64_t *result) {
  if (buf && size)
    std::memset(buf, 0, size);
  if (result)
    *result = 0;
  return kOk;
}
int PS4ABI sceSaveDataSetParam(const void *, int, const void *, uint64_t) {
  return kOk;
}
int PS4ABI sceSaveDataSaveIcon(const void *, const void *) { return kOk; }
