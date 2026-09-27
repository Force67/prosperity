/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceSaveData backed by a writable host directory. See lib_sce_save_data.h
 * for the client/server rationale and the Orbis struct layouts referenced here.
 */

#include "guest_abi.h"
#include <base/environment_variables.h>
#include "base/arch.h"
#include <base/logging.h>
#include "lib_sce_save_data.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kern/vfs.h"
#include <options/options.h>
#include <base/algorithm.h>
#include <base/containers/vector.h>
#include <base/strings/xstring.h>
#include <base/threading/lock_guard.h>
#include <base/threading/mutex.h>

namespace {
DELTA_OPTION(const char *, kSavedataDir, "DELTA_SAVEDATA_DIR", nullptr);
DELTA_OPTION(bool, kSaveTrace, "DELTA_SAVE_TRACE", false);
DELTA_OPTION(bool, kSavedataTrace, "DELTA_SAVEDATA_TRACE", false);
}  // namespace

namespace {

// DELTA_SAVEDATA_TRACE: which save-data entry points the title actually calls.
// A menu that waits on save enumeration gives no other sign of it, so "was this
// even called" is the first thing worth knowing.
static void sdTrace(const char* fn) {
  if (kSavedataTrace)
    BASE_LOGI("savedata", "call {}", fn);
}

// Mount modes (SCE_SAVE_DATA_MOUNT_MODE_*).
constexpr u32 kModeRdOnly = 1;
constexpr u32 kModeRdWr = 2;
constexpr u32 kModeCreate = 4;
constexpr u32 kModeCreate2 = 32;

// Errors (SCE_SAVE_DATA_ERROR_*).
constexpr int kOk = 0;
constexpr int kErrParameter = static_cast<int>(0x809F0000u);
constexpr int kErrNotMounted = static_cast<int>(0x809F0004u);
constexpr int kErrExists = static_cast<int>(0x809F0007u);
constexpr int kErrNotFound = static_cast<int>(0x809F0008u);

// 1 savedata block = 32 KiB. Report a quota with room to spare, but NOT a round
// power of two: a title that converts blocks to bytes in 32 bits wraps any
// multiple of 4 GiB to exactly zero and reads that as "no free space".
// Minecraft's world creation is gated on exactly that check.
constexpr u64 kTotalBlocks = 60000;  // ~1.83 GiB
constexpr u64 kFreeBlocks = 50000;   // ~1.53 GiB

// OrbisSaveDataParam sidecar layout.
constexpr size_t kParamSize = 1328;
constexpr size_t kOffTitle = 0;      // char[128]
constexpr size_t kOffSubTitle = 128; // char[128]
constexpr size_t kOffDetail = 256;   // char[1024]
constexpr size_t kOffUserParam = 1280;
constexpr size_t kOffMtime = 1288;

// OrbisSaveDataParamType.
constexpr u32 kParamAll = 0;
constexpr u32 kParamTitle = 1;
constexpr u32 kParamSubTitle = 2;
constexpr u32 kParamDetail = 3;
constexpr u32 kParamUserParam = 4;
constexpr u32 kParamMtime = 5;

base::Mutex g_mtx;
int g_nextSlot = 0;
int g_nextTransactionResource = 1;

struct Slot {
  base::String point;  // "/savedataN"
  base::String host;   // host directory
  bool readOnly = false;
};
base::Vector<Slot> g_slots;  // guarded by g_mtx

bool g_trace() {
  return kSaveTrace;
}

bool hostDirExists(const base::String &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}

// mkdir -p on the host.
void makeHostDirs(const base::String &path) {
  base::String p = path;
  for (size_t i = 1; i < p.size(); i++) {
    if (p[i] == '/') {
      p[i] = 0;
      ::mkdir(p.c_str(), 0755);
      p[i] = '/';
    }
  }
  ::mkdir(p.c_str(), 0755);
}

// Remove a directory tree on the host (like `rm -rf`).
void removeTree(const base::String &path) {
  DIR *d = ::opendir(path.c_str());
  if (d) {
    while (dirent *e = ::readdir(d)) {
      if (!std::strcmp(e->d_name, ".") || !std::strcmp(e->d_name, ".."))
        continue;
      base::String child = path + "/" + e->d_name;
      struct stat st;
      if (::stat(child.c_str(), &st) == 0 && (st.st_mode & S_IFDIR))
        removeTree(child);
      else
        ::remove(child.c_str());
    }
    ::closedir(d);
  }
  ::rmdir(path.c_str());
}

// Host directory that holds every title's saves. Matches the pre-existing
// behaviour ($DELTA_SAVEDATA_DIR, else ~/.prosperity/savedata) so saves written
// before this module gained per-title roots stay reachable.
base::String saveRoot() {
  if (const char *e = kSavedataDir)
    return e;
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  return base::String(home.empty() ? "." : (const char *)home.c_str()) +
         "/.prosperity/savedata";
}

// The booted title's tag for the save root. dcore parses TITLE_ID from the pkg's
// (outer) param.sfo; fall back to "SAVEDATA" when it can't be determined so
// per-title layout still works and never produces an empty path component.
const base::String &titleTag() {
  static const base::String tag = [] {
    base::String t = kern::vfs::TitleId();
    if (g_trace())
      BASE_LOGI("savedata", "title id = {}",
                t.empty() ? "(fallback SAVEDATA)" : t.c_str());
    return t.empty() ? base::String("SAVEDATA") : t;
  }();
  return tag;
}

base::String titleRoot() { return saveRoot() + "/" + titleTag(); }

// The host dir for a save. Prefer the per-title path; but if it doesn't exist
// yet and a legacy dirName-only save does (written before per-title roots),
// keep using the legacy path so those saves are not orphaned. `existing` is set
// to whether a save already lives at the chosen path.
base::String chooseHost(const char *dirName, bool &existing) {
  const base::String perTitle = titleRoot() + "/" + dirName;
  if (hostDirExists(perTitle)) {
    existing = true;
    return perTitle;
  }
  const base::String legacy = saveRoot() + "/" + dirName;
  if (hostDirExists(legacy)) {
    existing = true;
    return legacy;  // migrate-in-place: reuse the pre-per-title save
  }
  existing = false;
  return perTitle;  // brand-new save -> per-title root
}

// Look up the host directory a mount point ("/savedataN") maps to.
base::String hostForPoint(const void *mountPoint) {
  if (!mountPoint)
    return {};
  const char *point = static_cast<const char *>(mountPoint);
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (auto &s : g_slots)
    if (s.point == point)
      return s.host;
  return {};
}

int unmountPoint(const void *mountPoint) {
  if (!mountPoint)
    return kErrParameter;
  const char *point = static_cast<const char *>(mountPoint);
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (auto it = g_slots.begin(); it != g_slots.end(); ++it) {
    if (it->point == point) {
      kern::vfs::Unmount(point);
      g_slots.erase(it);
      return kOk;
    }
  }
  return kErrNotMounted;
}

// Read a guest pointer stored at byte offset `off` in `base`.
const void *ptrAt(const void *base, size_t off) {
  const void *p = nullptr;
  std::memcpy(&p, static_cast<const u8 *>(base) + off, sizeof(p));
  return p;
}
u32 u32At(const void *base, size_t off) {
  u32 v = 0;
  std::memcpy(&v, static_cast<const u8 *>(base) + off, 4);
  return v;
}
u64 u64At(const void *base, size_t off) {
  u64 v = 0;
  std::memcpy(&v, static_cast<const u8 *>(base) + off, 8);
  return v;
}

// The char[] a DirName* / MountPoint* points at.
const char *cstrOf(const void *p) { return static_cast<const char *>(p); }

// -------- param sidecar (.sce_param.bin) --------

base::String paramPath(const base::String &host) {
  return host + "/.sce_param.bin";
}

void loadParam(const base::String &host, u8 *out /*kParamSize*/) {
  std::memset(out, 0, kParamSize);
  if (FILE *f = std::fopen(paramPath(host).c_str(), "rb")) {
    size_t got = std::fread(out, 1, kParamSize, f);
    (void)got;
    std::fclose(f);
  }
}

void storeParam(const base::String &host, const u8 *blob /*kParamSize*/) {
  if (FILE *f = std::fopen(paramPath(host).c_str(), "wb")) {
    std::fwrite(blob, 1, kParamSize, f);
    std::fclose(f);
  }
}

// -------- SaveDataMemory backing --------

base::String memoryPath(u32 slotId) {
  base::String dir = titleRoot() + "/sce_sdmemory";
  makeHostDirs(dir);
  char name[64];
  std::snprintf(name, sizeof(name), "/memory%u.bin", slotId);
  return dir + name;
}

int memorySetup(u64 memorySize, u32 slotId, void *result) {
  const base::String path = memoryPath(slotId);
  u64 existed = 0;
  struct stat st;
  if (::stat(path.c_str(), &st) == 0)
    existed = static_cast<u64>(st.st_size);
  // Create/extend the blob to memorySize (fill with zeros).
  if (existed < memorySize) {
    if (FILE *f = std::fopen(path.c_str(), existed ? "rb+" : "wb")) {
      if (std::fseek(f, static_cast<long>(memorySize) - 1, SEEK_SET) == 0) {
        const u8 z = 0;
        std::fwrite(&z, 1, 1, f);
      }
      std::fclose(f);
    }
  }
  if (result) {
    std::memset(result, 0, 24);
    std::memcpy(result, &existed, 8);  // existedMemorySize@0
  }
  if (g_trace())
    BASE_LOGI("savedata", "memory setup slot={} size={} (existed={})", slotId,
              (unsigned long long)memorySize, (unsigned long long)existed);
  return kOk;
}

int memoryRead(u32 slotId, void *buf, u64 bufSize, i64 offset) {
  if (buf && bufSize)
    std::memset(buf, 0, bufSize);
  if (!buf)
    return kOk;
  if (FILE *f = std::fopen(memoryPath(slotId).c_str(), "rb")) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) == 0) {
      size_t got = std::fread(buf, 1, bufSize, f);
      (void)got;
    }
    std::fclose(f);
  }
  return kOk;
}

int memoryWrite(u32 slotId, const void *buf, u64 bufSize,
                i64 offset) {
  if (!buf || !bufSize)
    return kOk;
  const base::String path = memoryPath(slotId);
  FILE *f = std::fopen(path.c_str(), "rb+");
  if (!f)
    f = std::fopen(path.c_str(), "wb");
  if (f) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) == 0)
      std::fwrite(buf, 1, bufSize, f);
    std::fclose(f);
  }
  return kOk;
}

// Shared mount core.
int doMount(const char *dir_name, u32 mode, void *result) {
  if (!dir_name || !dir_name[0])
    return kErrParameter;
  bool exists = false;
  const base::String host = chooseHost(dir_name, exists);
  const bool create = (mode & (kModeCreate | kModeCreate2)) != 0;
  const bool readOnly = (mode & kModeRdOnly) && !(mode & kModeRdWr);

  if (!exists && !create)
    return kErrNotFound;  // e.g. a read-only "does a save exist?" probe
  if (exists && (mode & kModeCreate))
    return kErrExists;  // strict CREATE requires the save not to exist yet

  base::LockGuard<base::Mutex> lk(g_mtx);
  char point[16];
  std::snprintf(point, sizeof(point), "/savedata%d", g_nextSlot++);
  kern::vfs::MountWritable(point, host.c_str());  // creates the host dir
  g_slots.push_back({point, host, readOnly});

  if (result) {
    auto *r = static_cast<u8 *>(result);
    std::memset(r, 0, 64);
    std::snprintf(reinterpret_cast<char *>(r), 16, "%s", point);
    const u32 status = exists ? 0u : 1u;  // 1 = SAVE_DATA_CREATED
    std::memcpy(r + 28, &status, 4);            // mount_status@28
  }
  if (g_trace())
    BASE_LOGI("savedata", "mount dir='{}' mode={:#x} -> {} (host={}, {})",
              dir_name, mode, point, host.c_str(),
              exists ? "existing" : "created");
  return kOk;
}

}  // namespace

// ---------------- init / term ----------------

int PS4ABI sceSaveDataInitialize(void *) {
  sdTrace("sceSaveDataInitialize"); return kOk; }
int PS4ABI sceSaveDataInitialize2(void *) {
  sdTrace("sceSaveDataInitialize2"); return kOk; }
int PS4ABI sceSaveDataInitialize3(void *) {
  sdTrace("sceSaveDataInitialize3"); return kOk; }
int PS4ABI sceSaveDataTerminate() {
  sdTrace("sceSaveDataTerminate"); return kOk; }

// ---------------- mount ----------------

int PS4ABI sceSaveDataMount2(const void *mount, void *result) {
  sdTrace("sceSaveDataMount2");
  if (!mount)
    return kErrParameter;
  return doMount(cstrOf(ptrAt(mount, 8)), u32At(mount, 24), result);
}

int PS4ABI sceSaveDataMount(const void *mount, void *result) {
  sdTrace("sceSaveDataMount");
  if (!mount)
    return kErrParameter;
  return doMount(cstrOf(ptrAt(mount, 16)), u32At(mount, 40), result);
}

int PS4ABI sceSaveDataMount3(const void *mount, void *result) {
  sdTrace("sceSaveDataMount3");
  if (!mount)
    return kErrParameter;
  return doMount(cstrOf(ptrAt(mount, 8)), u32At(mount, 32), result);
}

int PS4ABI sceSaveDataMount5(const void *mount, void *result) {
  sdTrace("sceSaveDataMount5");
  return sceSaveDataMount2(mount, result);  // same leading layout for our use
}

int PS4ABI sceSaveDataUmount(const void *mountPoint) {
  sdTrace("sceSaveDataUmount"); return unmountPoint(mountPoint); }
int PS4ABI sceSaveDataUmountWithBackup(const void *mountPoint) {
  sdTrace("sceSaveDataUmountWithBackup"); return unmountPoint(mountPoint); }

int PS4ABI sceSaveDataUmount2(u32, const void *mountPoint) {
  sdTrace("sceSaveDataUmount2"); return unmountPoint(mountPoint); }

int PS4ABI sceSaveDataGetMountInfo(const void *, void *info) {
  sdTrace("sceSaveDataGetMountInfo");
  if (info) {
    auto *i = static_cast<u8 *>(info);
    std::memset(i, 0, 48);
    std::memcpy(i + 0, &kTotalBlocks, 8);  // total blocks
    std::memcpy(i + 8, &kFreeBlocks, 8);   // free blocks
  }
  return kOk;
}

int PS4ABI sceSaveDataCreateTransactionResource(u32) {
  sdTrace("sceSaveDataCreateTransactionResource");
  base::LockGuard<base::Mutex> lk(g_mtx);
  return g_nextTransactionResource++;
}

int PS4ABI sceSaveDataDeleteTransactionResource(i32) {
  sdTrace("sceSaveDataDeleteTransactionResource");
  return kOk;
}

int PS4ABI sceSaveDataPrepare(const void *mountPoint, const void *param) {
  sdTrace("sceSaveDataPrepare");
  return mountPoint && param ? kOk : kErrParameter;
}

int PS4ABI sceSaveDataCommit(const void *param) {
  sdTrace("sceSaveDataCommit");
  return param ? kOk : kErrParameter;
}

// ---------------- enumerate / delete / backup ----------------

int PS4ABI sceSaveDataDirNameSearch(const void *cond, void *result) {
  sdTrace("sceSaveDataDirNameSearch");
  if (!result)
    return kErrParameter;
  auto *r = static_cast<u8 *>(result);
  base::String searchRoot = titleRoot();
  if (cond) {
    const void *TitleId = ptrAt(cond, 8);
    if (TitleId && cstrOf(TitleId)[0])
      searchRoot = saveRoot() + "/" +
                   base::String(cstrOf(TitleId), strnlen(cstrOf(TitleId), 10));
  }
  // dirNamesNum@16 is the caller's array capacity (input).
  const u32 capacity = u32At(result, 16);
  auto *dirNames = static_cast<u8 *>(const_cast<void *>(ptrAt(result, 8)));
  auto *params = static_cast<u8 *>(const_cast<void *>(ptrAt(result, 24)));
  auto *infos = static_cast<u8 *>(const_cast<void *>(ptrAt(result, 32)));

  // Optional name filter from the search condition.
  const char *filter = nullptr;
  if (cond) {
    const void *dn = ptrAt(cond, 16);
    if (dn && cstrOf(dn)[0])
      filter = cstrOf(dn);
  }

  // Enumerate existing save directories under the title root.
  base::Vector<base::String> hits;
  if (DIR *d = ::opendir(searchRoot.c_str())) {
    while (dirent *e = ::readdir(d)) {
      if (e->d_name[0] == '.' || !std::strncmp(e->d_name, "sce_", 4))
        continue;
      base::String child = searchRoot + "/" + e->d_name;
      struct stat st;
      if (::stat(child.c_str(), &st) != 0 || !(st.st_mode & S_IFDIR))
        continue;
      if (filter && !std::strstr(e->d_name, filter))
        continue;
      hits.push_back(e->d_name);
    }
    ::closedir(d);
  }
  base::Sort(hits.begin(), hits.end());
  // order@28: 1 = DESCENT.
  if (cond && u32At(cond, 28) == 1)
    base::Reverse(hits.begin(), hits.end());

  const u32 setNum =
      capacity < hits.size() ? capacity : static_cast<u32>(hits.size());
  for (u32 i = 0; i < setNum; i++) {
    if (dirNames) {
      char *slot = reinterpret_cast<char *>(dirNames) + i * 32;
      std::memset(slot, 0, 32);
      std::snprintf(slot, 32, "%s", hits[i].c_str());
    }
    if (params) {
      u8 *pslot = params + i * kParamSize;
      loadParam(searchRoot + "/" + hits[i], pslot);
    }
    if (infos) {
      u8 *islot = infos + i * 48;  // SearchInfo { u64 blocks; u64 free; }
      std::memset(islot, 0, 48);
      std::memcpy(islot + 0, &kTotalBlocks, 8);
      std::memcpy(islot + 8, &kFreeBlocks, 8);
    }
  }
  const u32 hitNum = static_cast<u32>(hits.size());
  std::memcpy(r + 0, &hitNum, 4);   // hitNum
  std::memcpy(r + 20, &setNum, 4);  // setNum
  // dirNamesNum stays the caller's capacity value; leave it untouched.
  if (g_trace())
    BASE_LOGI("savedata", "dirNameSearch -> {} hit(s), {} returned", hitNum,
              setNum);
  return kOk;
}

int PS4ABI sceSaveDataDelete(const void *del) {
  sdTrace("sceSaveDataDelete");
  if (!del)
    return kErrParameter;
  const void *dn = ptrAt(del, 16);
  if (!dn || !cstrOf(dn)[0])
    return kErrParameter;
  bool exists = false;
  const base::String host = chooseHost(cstrOf(dn), exists);
  if (exists)
    removeTree(host);
  if (g_trace())
    BASE_LOGI("savedata", "delete dir='{}'", cstrOf(dn));
  return kOk;
}

int PS4ABI sceSaveDataCheckBackupData(const void *check) {
  sdTrace("sceSaveDataCheckBackupData");
  if (!check)
    return kErrParameter;
  // We keep no separate backup image (the save dir itself persists across
  // umount), so there is never a backup to restore: report NOT_FOUND, which is
  // the correct "no backup" answer and makes titles fall back to a normal load.
  return kErrNotFound;
}

int PS4ABI sceSaveDataRestoreBackupData(const void *restore) {
  sdTrace("sceSaveDataRestoreBackupData");
  if (!restore)
    return kErrParameter;
  return kErrNotFound;  // no backup image (see sceSaveDataCheckBackupData)
}

// ---------------- param / icon ----------------

int PS4ABI sceSaveDataGetParam(const void *mountPoint, u32 paramType,
                               void *buf, u64 size, u64 *result) {
  sdTrace("sceSaveDataGetParam");
  if (buf && size)
    std::memset(buf, 0, size);
  if (result)
    *result = 0;
  const base::String host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (!buf || !size)
    return kOk;

  u8 blob[kParamSize];
  loadParam(host, blob);
  size_t off = 0, len = 0;
  switch (paramType) {
  case kParamAll:
    off = 0;
    len = kParamSize;
    break;
  case kParamTitle:
    off = kOffTitle;
    len = 128;
    break;
  case kParamSubTitle:
    off = kOffSubTitle;
    len = 128;
    break;
  case kParamDetail:
    off = kOffDetail;
    len = 1024;
    break;
  case kParamUserParam:
    off = kOffUserParam;
    len = 4;
    break;
  case kParamMtime:
    off = kOffMtime;
    len = 8;
    break;
  default:
    return kErrParameter;
  }
  const size_t n = size < len ? static_cast<size_t>(size) : len;
  std::memcpy(buf, blob + off, n);
  if (result)
    *result = n;
  return kOk;
}

int PS4ABI sceSaveDataSetParam(const void *mountPoint, u32 paramType,
                               const void *buf, u64 size) {
  sdTrace("sceSaveDataSetParam");
  const base::String host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (!buf || !size)
    return kOk;

  u8 blob[kParamSize];
  loadParam(host, blob);
  size_t off = 0, len = 0;
  switch (paramType) {
  case kParamAll:
    off = 0;
    len = kParamSize;
    break;
  case kParamTitle:
    off = kOffTitle;
    len = 128;
    break;
  case kParamSubTitle:
    off = kOffSubTitle;
    len = 128;
    break;
  case kParamDetail:
    off = kOffDetail;
    len = 1024;
    break;
  case kParamUserParam:
    off = kOffUserParam;
    len = 4;
    break;
  case kParamMtime:
    off = kOffMtime;
    len = 8;
    break;
  default:
    return kErrParameter;
  }
  const size_t n = size < len ? static_cast<size_t>(size) : len;
  std::memcpy(blob + off, buf, n);
  storeParam(host, blob);
  if (g_trace())
    BASE_LOGI("savedata", "setParam type={} size={} -> {}", paramType,
              (unsigned long long)size, host.c_str());
  return kOk;
}

int PS4ABI sceSaveDataSaveIcon(const void *mountPoint, const void *icon) {
  sdTrace("sceSaveDataSaveIcon");
  const base::String host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (icon) {
    const void *iconBuf = ptrAt(icon, 0);
    const u64 bufSize = u64At(icon, 8);
    const u64 dataSize = u64At(icon, 16);
    const u64 n = dataSize < bufSize ? dataSize : bufSize;
    if (iconBuf && n) {
      if (FILE *f = std::fopen((host + "/.sce_icon.bin").c_str(), "wb")) {
        std::fwrite(iconBuf, 1, static_cast<size_t>(n), f);
        std::fclose(f);
      }
    }
  }
  if (g_trace())
    BASE_LOGI("savedata", "saveIcon -> {}", host.c_str());
  return kOk;
}

int PS4ABI sceSaveDataLoadIcon(const void *mountPoint, void *icon) {
  sdTrace("sceSaveDataLoadIcon");
  const base::String host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (!icon)
    return kErrParameter;

  const base::String path = host + "/.sce_icon.bin";
  struct stat st;
  if (::stat(path.c_str(), &st) != 0)
    return kErrNotFound;

  void *iconBuf = const_cast<void *>(ptrAt(icon, 0));
  const u64 bufSize = u64At(icon, 8);
  const u64 dataSize = static_cast<u64>(st.st_size);
  if (iconBuf && bufSize) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
      return kErrNotFound;
    const u64 copySize = dataSize < bufSize ? dataSize : bufSize;
    std::fread(iconBuf, 1, static_cast<size_t>(copySize), f);
    std::fclose(f);
  }
  std::memcpy(static_cast<u8 *>(icon) + 16, &dataSize, sizeof(dataSize));
  return kOk;
}

// ---------------- SaveDataMemory ----------------

int PS4ABI sceSaveDataSetupSaveDataMemory(u32, u64 memorySize,
                                          void *) {
  sdTrace("sceSaveDataSetupSaveDataMemory");
  return memorySetup(memorySize, 0, nullptr);
}

int PS4ABI sceSaveDataSetupSaveDataMemory2(const void *setupParam,
                                           void *result) {
  sdTrace("sceSaveDataSetupSaveDataMemory2");
  if (!setupParam) {
    if (result)
      std::memset(result, 0, 24);
    return kErrParameter;
  }
  const u64 memorySize = u64At(setupParam, 8);
  const u32 slotId = u32At(setupParam, 40);
  return memorySetup(memorySize, slotId, result);
}

int PS4ABI sceSaveDataGetSaveDataMemory(u32, void *buf, u64 bufSize,
                                        i64 offset) {
  sdTrace("sceSaveDataGetSaveDataMemory");
  return memoryRead(0, buf, bufSize, offset);
}

int PS4ABI sceSaveDataGetSaveDataMemory2(void *getParam) {
  sdTrace("sceSaveDataGetSaveDataMemory2");
  if (!getParam)
    return kErrParameter;
  const u32 slotId = u32At(getParam, 32);
  const void *data = ptrAt(getParam, 8);
  if (!data)
    return kErrParameter;
  void *buf = const_cast<void *>(ptrAt(data, 0));
  const u64 bufSize = u64At(data, 8);
  const i64 offset = static_cast<i64>(u64At(data, 16));
  return memoryRead(slotId, buf, bufSize, offset);
}

int PS4ABI sceSaveDataSetSaveDataMemory(u32, const void *buf,
                                        u64 bufSize, i64 offset) {
  sdTrace("sceSaveDataSetSaveDataMemory");
  return memoryWrite(0, buf, bufSize, offset);
}

int PS4ABI sceSaveDataSetSaveDataMemory2(const void *setParam) {
  sdTrace("sceSaveDataSetSaveDataMemory2");
  if (!setParam)
    return kErrParameter;
  const u32 slotId = u32At(setParam, 36);
  const void *data = ptrAt(setParam, 8);
  if (!data)
    return kOk;
  const void *buf = ptrAt(data, 0);
  const u64 bufSize = u64At(data, 8);
  const i64 offset = static_cast<i64>(u64At(data, 16));
  return memoryWrite(slotId, buf, bufSize, offset);
}

int PS4ABI sceSaveDataSyncSaveDataMemory(void *) {
  sdTrace("sceSaveDataSyncSaveDataMemory"); return kOk; }
int PS4ABI sceSaveDataRestoreLoadSaveDataMemory(const void *) {
  sdTrace("sceSaveDataRestoreLoadSaveDataMemory"); return kOk; }
