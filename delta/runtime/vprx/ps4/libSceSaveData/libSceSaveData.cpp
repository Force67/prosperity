/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceSaveData backed by a writable host directory. See libSceSaveData.h
 * for the client/server rationale and the Orbis struct layouts referenced here.
 */

#include "libSceSaveData.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "kern/vfs.h"

namespace {

// DELTA_SAVEDATA_TRACE: which save-data entry points the title actually calls.
// A menu that waits on save enumeration gives no other sign of it, so "was this
// even called" is the first thing worth knowing.
static void sdTrace(const char* fn) {
  static const bool on = std::getenv("DELTA_SAVEDATA_TRACE") != nullptr;
  if (on)
    std::fprintf(stderr, "[savedata] call %s\n", fn);
}

// Mount modes (SCE_SAVE_DATA_MOUNT_MODE_*).
constexpr uint32_t kModeRdOnly = 1;
constexpr uint32_t kModeRdWr = 2;
constexpr uint32_t kModeCreate = 4;
constexpr uint32_t kModeCreate2 = 32;

// Errors (SCE_SAVE_DATA_ERROR_*).
constexpr int kOk = 0;
constexpr int kErrParameter = static_cast<int>(0x809F0000u);
constexpr int kErrNotMounted = static_cast<int>(0x809F0004u);
constexpr int kErrExists = static_cast<int>(0x809F0007u);
constexpr int kErrNotFound = static_cast<int>(0x809F0008u);

// 1 savedata block = 32 KiB; report ~32 GiB of free space to titles that ask.
constexpr uint64_t kTotalBlocks = 1u << 20;

// OrbisSaveDataParam sidecar layout.
constexpr size_t kParamSize = 1328;
constexpr size_t kOffTitle = 0;      // char[128]
constexpr size_t kOffSubTitle = 128; // char[128]
constexpr size_t kOffDetail = 256;   // char[1024]
constexpr size_t kOffUserParam = 1280;
constexpr size_t kOffMtime = 1288;

// OrbisSaveDataParamType.
constexpr uint32_t kParamAll = 0;
constexpr uint32_t kParamTitle = 1;
constexpr uint32_t kParamSubTitle = 2;
constexpr uint32_t kParamDetail = 3;
constexpr uint32_t kParamUserParam = 4;
constexpr uint32_t kParamMtime = 5;

std::mutex g_mtx;
int g_nextSlot = 0;

struct Slot {
  std::string point;  // "/savedataN"
  std::string host;   // host directory
  bool readOnly = false;
};
std::vector<Slot> g_slots;  // guarded by g_mtx

bool g_trace() {
  static const bool on = std::getenv("DELTA_SAVE_TRACE") != nullptr;
  return on;
}

bool hostDirExists(const std::string &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}

// mkdir -p on the host.
void makeHostDirs(const std::string &path) {
  std::string p = path;
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
void removeTree(const std::string &path) {
  DIR *d = ::opendir(path.c_str());
  if (d) {
    while (dirent *e = ::readdir(d)) {
      if (!std::strcmp(e->d_name, ".") || !std::strcmp(e->d_name, ".."))
        continue;
      std::string child = path + "/" + e->d_name;
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
std::string saveRoot() {
  if (const char *e = std::getenv("DELTA_SAVEDATA_DIR"))
    return e;
  const char *home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.prosperity/savedata";
}

// The booted title's tag for the save root. dcore parses TITLE_ID from the pkg's
// (outer) param.sfo; fall back to "SAVEDATA" when it can't be determined so
// per-title layout still works and never produces an empty path component.
const std::string &titleTag() {
  static const std::string tag = [] {
    std::string t = krnl::vfs::titleId();
    if (g_trace())
      std::fprintf(stderr, "[savedata] title id = %s\n",
                   t.empty() ? "(fallback SAVEDATA)" : t.c_str());
    return t.empty() ? std::string("SAVEDATA") : t;
  }();
  return tag;
}

std::string titleRoot() { return saveRoot() + "/" + titleTag(); }

// The host dir for a save. Prefer the per-title path; but if it doesn't exist
// yet and a legacy dirName-only save does (written before per-title roots),
// keep using the legacy path so those saves are not orphaned. `existing` is set
// to whether a save already lives at the chosen path.
std::string chooseHost(const char *dirName, bool &existing) {
  const std::string perTitle = titleRoot() + "/" + dirName;
  if (hostDirExists(perTitle)) {
    existing = true;
    return perTitle;
  }
  const std::string legacy = saveRoot() + "/" + dirName;
  if (hostDirExists(legacy)) {
    existing = true;
    return legacy;  // migrate-in-place: reuse the pre-per-title save
  }
  existing = false;
  return perTitle;  // brand-new save -> per-title root
}

// Look up the host directory a mount point ("/savedataN") maps to.
std::string hostForPoint(const void *mountPoint) {
  if (!mountPoint)
    return {};
  const char *point = static_cast<const char *>(mountPoint);
  std::lock_guard<std::mutex> lk(g_mtx);
  for (auto &s : g_slots)
    if (s.point == point)
      return s.host;
  return {};
}

// Read a guest pointer stored at byte offset `off` in `base`.
const void *ptrAt(const void *base, size_t off) {
  const void *p = nullptr;
  std::memcpy(&p, static_cast<const uint8_t *>(base) + off, sizeof(p));
  return p;
}
uint32_t u32At(const void *base, size_t off) {
  uint32_t v = 0;
  std::memcpy(&v, static_cast<const uint8_t *>(base) + off, 4);
  return v;
}
uint64_t u64At(const void *base, size_t off) {
  uint64_t v = 0;
  std::memcpy(&v, static_cast<const uint8_t *>(base) + off, 8);
  return v;
}

// The char[] a DirName* / MountPoint* points at.
const char *cstrOf(const void *p) { return static_cast<const char *>(p); }

// -------- param sidecar (.sce_param.bin) --------

std::string paramPath(const std::string &host) {
  return host + "/.sce_param.bin";
}

void loadParam(const std::string &host, uint8_t *out /*kParamSize*/) {
  std::memset(out, 0, kParamSize);
  if (FILE *f = std::fopen(paramPath(host).c_str(), "rb")) {
    size_t got = std::fread(out, 1, kParamSize, f);
    (void)got;
    std::fclose(f);
  }
}

void storeParam(const std::string &host, const uint8_t *blob /*kParamSize*/) {
  if (FILE *f = std::fopen(paramPath(host).c_str(), "wb")) {
    std::fwrite(blob, 1, kParamSize, f);
    std::fclose(f);
  }
}

// -------- SaveDataMemory backing --------

std::string memoryPath(uint32_t slotId) {
  std::string dir = titleRoot() + "/sce_sdmemory";
  makeHostDirs(dir);
  char name[64];
  std::snprintf(name, sizeof(name), "/memory%u.bin", slotId);
  return dir + name;
}

int memorySetup(uint64_t memorySize, uint32_t slotId, void *result) {
  const std::string path = memoryPath(slotId);
  uint64_t existed = 0;
  struct stat st;
  if (::stat(path.c_str(), &st) == 0)
    existed = static_cast<uint64_t>(st.st_size);
  // Create/extend the blob to memorySize (fill with zeros).
  if (existed < memorySize) {
    if (FILE *f = std::fopen(path.c_str(), existed ? "rb+" : "wb")) {
      if (std::fseek(f, static_cast<long>(memorySize) - 1, SEEK_SET) == 0) {
        const uint8_t z = 0;
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
    std::fprintf(stderr,
                 "[savedata] memory setup slot=%u size=%llu (existed=%llu)\n",
                 slotId, (unsigned long long)memorySize,
                 (unsigned long long)existed);
  return kOk;
}

int memoryRead(uint32_t slotId, void *buf, uint64_t bufSize, int64_t offset) {
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

int memoryWrite(uint32_t slotId, const void *buf, uint64_t bufSize,
                int64_t offset) {
  if (!buf || !bufSize)
    return kOk;
  const std::string path = memoryPath(slotId);
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
int doMount(const char *dir_name, uint32_t mode, void *result) {
  if (!dir_name || !dir_name[0])
    return kErrParameter;
  bool exists = false;
  const std::string host = chooseHost(dir_name, exists);
  const bool create = (mode & (kModeCreate | kModeCreate2)) != 0;
  const bool readOnly = (mode & kModeRdOnly) && !(mode & kModeRdWr);

  if (!exists && !create)
    return kErrNotFound;  // e.g. a read-only "does a save exist?" probe
  if (exists && (mode & kModeCreate))
    return kErrExists;  // strict CREATE requires the save not to exist yet

  std::lock_guard<std::mutex> lk(g_mtx);
  char point[16];
  std::snprintf(point, sizeof(point), "/savedata%d", g_nextSlot++);
  krnl::vfs::mountWritable(point, host.c_str());  // creates the host dir
  g_slots.push_back({point, host, readOnly});

  if (result) {
    auto *r = static_cast<uint8_t *>(result);
    std::memset(r, 0, 64);
    std::snprintf(reinterpret_cast<char *>(r), 16, "%s", point);
    const uint32_t status = exists ? 0u : 1u;  // 1 = SAVE_DATA_CREATED
    std::memcpy(r + 28, &status, 4);            // mount_status@28
  }
  if (g_trace())
    std::fprintf(stderr,
                 "[savedata] mount dir='%s' mode=%#x -> %s (host=%s, %s)\n",
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

int PS4ABI sceSaveDataMount5(const void *mount, void *result) {
  sdTrace("sceSaveDataMount5");
  return sceSaveDataMount2(mount, result);  // same leading layout for our use
}

int PS4ABI sceSaveDataUmount(const void *) {
  sdTrace("sceSaveDataUmount"); return kOk; }
int PS4ABI sceSaveDataUmountWithBackup(const void *) {
  sdTrace("sceSaveDataUmountWithBackup"); return kOk; }

int PS4ABI sceSaveDataGetMountInfo(const void *, void *info) {
  sdTrace("sceSaveDataGetMountInfo");
  if (info) {
    auto *i = static_cast<uint8_t *>(info);
    std::memset(i, 0, 48);
    std::memcpy(i + 0, &kTotalBlocks, 8);  // total blocks
    std::memcpy(i + 8, &kTotalBlocks, 8);  // free blocks
  }
  return kOk;
}

// ---------------- enumerate / delete / backup ----------------

int PS4ABI sceSaveDataDirNameSearch(const void *cond, void *result) {
  sdTrace("sceSaveDataDirNameSearch");
  if (!result)
    return kErrParameter;
  auto *r = static_cast<uint8_t *>(result);
  // dirNamesNum@16 is the caller's array capacity (input).
  const uint32_t capacity = u32At(result, 16);
  auto *dirNames = static_cast<uint8_t *>(const_cast<void *>(ptrAt(result, 8)));
  auto *params = static_cast<uint8_t *>(const_cast<void *>(ptrAt(result, 24)));
  auto *infos = static_cast<uint8_t *>(const_cast<void *>(ptrAt(result, 32)));

  // Optional name filter from the search condition.
  const char *filter = nullptr;
  if (cond) {
    const void *dn = ptrAt(cond, 16);
    if (dn && cstrOf(dn)[0])
      filter = cstrOf(dn);
  }

  // Enumerate existing save directories under the title root.
  std::vector<std::string> hits;
  if (DIR *d = ::opendir(titleRoot().c_str())) {
    while (dirent *e = ::readdir(d)) {
      if (e->d_name[0] == '.' || !std::strncmp(e->d_name, "sce_", 4))
        continue;
      std::string child = titleRoot() + "/" + e->d_name;
      struct stat st;
      if (::stat(child.c_str(), &st) != 0 || !(st.st_mode & S_IFDIR))
        continue;
      if (filter && !std::strstr(e->d_name, filter))
        continue;
      hits.push_back(e->d_name);
    }
    ::closedir(d);
  }
  std::sort(hits.begin(), hits.end());
  // order@28: 1 = DESCENT.
  if (cond && u32At(cond, 28) == 1)
    std::reverse(hits.begin(), hits.end());

  const uint32_t setNum =
      capacity < hits.size() ? capacity : static_cast<uint32_t>(hits.size());
  for (uint32_t i = 0; i < setNum; i++) {
    if (dirNames) {
      char *slot = reinterpret_cast<char *>(dirNames) + i * 32;
      std::memset(slot, 0, 32);
      std::snprintf(slot, 32, "%s", hits[i].c_str());
    }
    if (params) {
      uint8_t *pslot = params + i * kParamSize;
      loadParam(titleRoot() + "/" + hits[i], pslot);
    }
    if (infos) {
      uint8_t *islot = infos + i * 48;  // SearchInfo { u64 blocks; u64 free; }
      std::memset(islot, 0, 48);
      std::memcpy(islot + 0, &kTotalBlocks, 8);
      std::memcpy(islot + 8, &kTotalBlocks, 8);
    }
  }
  const uint32_t hitNum = static_cast<uint32_t>(hits.size());
  std::memcpy(r + 0, &hitNum, 4);   // hitNum
  std::memcpy(r + 20, &setNum, 4);  // setNum
  // dirNamesNum stays the caller's capacity value; leave it untouched.
  if (g_trace())
    std::fprintf(stderr, "[savedata] dirNameSearch -> %u hit(s), %u returned\n",
                 hitNum, setNum);
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
  const std::string host = chooseHost(cstrOf(dn), exists);
  if (exists)
    removeTree(host);
  if (g_trace())
    std::fprintf(stderr, "[savedata] delete dir='%s'\n", cstrOf(dn));
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

int PS4ABI sceSaveDataGetParam(const void *mountPoint, uint32_t paramType,
                               void *buf, uint64_t size, uint64_t *result) {
  sdTrace("sceSaveDataGetParam");
  if (buf && size)
    std::memset(buf, 0, size);
  if (result)
    *result = 0;
  const std::string host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (!buf || !size)
    return kOk;

  uint8_t blob[kParamSize];
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

int PS4ABI sceSaveDataSetParam(const void *mountPoint, uint32_t paramType,
                               const void *buf, uint64_t size) {
  sdTrace("sceSaveDataSetParam");
  const std::string host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (!buf || !size)
    return kOk;

  uint8_t blob[kParamSize];
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
    std::fprintf(stderr, "[savedata] setParam type=%u size=%llu -> %s\n",
                 paramType, (unsigned long long)size, host.c_str());
  return kOk;
}

int PS4ABI sceSaveDataSaveIcon(const void *mountPoint, const void *icon) {
  sdTrace("sceSaveDataSaveIcon");
  const std::string host = hostForPoint(mountPoint);
  if (host.empty())
    return kErrNotMounted;
  if (icon) {
    const void *iconBuf = ptrAt(icon, 0);
    const uint64_t bufSize = u64At(icon, 8);
    const uint64_t dataSize = u64At(icon, 16);
    const uint64_t n = dataSize < bufSize ? dataSize : bufSize;
    if (iconBuf && n) {
      if (FILE *f = std::fopen((host + "/.sce_icon.bin").c_str(), "wb")) {
        std::fwrite(iconBuf, 1, static_cast<size_t>(n), f);
        std::fclose(f);
      }
    }
  }
  if (g_trace())
    std::fprintf(stderr, "[savedata] saveIcon -> %s\n", host.c_str());
  return kOk;
}

// ---------------- SaveDataMemory ----------------

int PS4ABI sceSaveDataSetupSaveDataMemory(uint32_t, uint64_t memorySize,
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
  const uint64_t memorySize = u64At(setupParam, 8);
  const uint32_t slotId = u32At(setupParam, 40);
  return memorySetup(memorySize, slotId, result);
}

int PS4ABI sceSaveDataGetSaveDataMemory(uint32_t, void *buf, uint64_t bufSize,
                                        int64_t offset) {
  sdTrace("sceSaveDataGetSaveDataMemory");
  return memoryRead(0, buf, bufSize, offset);
}

int PS4ABI sceSaveDataGetSaveDataMemory2(void *getParam) {
  sdTrace("sceSaveDataGetSaveDataMemory2");
  if (!getParam)
    return kErrParameter;
  const uint32_t slotId = u32At(getParam, 32);
  const void *data = ptrAt(getParam, 8);
  if (!data)
    return kErrParameter;
  void *buf = const_cast<void *>(ptrAt(data, 0));
  const uint64_t bufSize = u64At(data, 8);
  const int64_t offset = static_cast<int64_t>(u64At(data, 16));
  return memoryRead(slotId, buf, bufSize, offset);
}

int PS4ABI sceSaveDataSetSaveDataMemory(uint32_t, const void *buf,
                                        uint64_t bufSize, int64_t offset) {
  sdTrace("sceSaveDataSetSaveDataMemory");
  return memoryWrite(0, buf, bufSize, offset);
}

int PS4ABI sceSaveDataSetSaveDataMemory2(const void *setParam) {
  sdTrace("sceSaveDataSetSaveDataMemory2");
  if (!setParam)
    return kErrParameter;
  const uint32_t slotId = u32At(setParam, 36);
  const void *data = ptrAt(setParam, 8);
  if (!data)
    return kOk;
  const void *buf = ptrAt(data, 0);
  const uint64_t bufSize = u64At(data, 8);
  const int64_t offset = static_cast<int64_t>(u64At(data, 16));
  return memoryWrite(slotId, buf, bufSize, offset);
}

int PS4ABI sceSaveDataSyncSaveDataMemory(void *) {
  sdTrace("sceSaveDataSyncSaveDataMemory"); return kOk; }
int PS4ABI sceSaveDataRestoreLoadSaveDataMemory(const void *) {
  sdTrace("sceSaveDataRestoreLoadSaveDataMemory"); return kOk; }
