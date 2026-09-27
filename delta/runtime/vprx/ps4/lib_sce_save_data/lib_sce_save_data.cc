/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceSaveData backed by a writable host directory. See
 * lib_sce_save_data.h for the client/server rationale and the Orbis struct
 * layouts referenced here.
 */

#include "runtime/vprx/ps4/lib_sce_save_data/lib_sce_save_data.h"
#include "base/arch.h"
#include "base/environment_variables.h"
#include "base/logging.h"
#include "guest_abi.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kSavedataDir, "DELTA_SAVEDATA_DIR", nullptr);
DELTA_OPTION(bool, kSaveTrace, "DELTA_SAVE_TRACE", false);
DELTA_OPTION(bool, kSavedataTrace, "DELTA_SAVEDATA_TRACE", false);
}  // namespace

namespace {

// DELTA_SAVEDATA_TRACE: which save-data entry points the title actually calls.
// A menu that waits on save enumeration gives no other sign of it, so "was this
// even called" is the first thing worth knowing.
static void SdTrace(const char* fn) {
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
constexpr size_t kOffTitle = 0;       // char[128]
constexpr size_t kOffSubTitle = 128;  // char[128]
constexpr size_t kOffDetail = 256;    // char[1024]
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
int g_next_slot = 0;
int g_next_transaction_resource = 1;

struct Slot {
  base::String point;  // "/savedataN"
  base::String host;   // host directory
  bool read_only = false;
};
base::Vector<Slot> g_slots;  // guarded by g_mtx

bool Trace() {
  return kSaveTrace;
}

bool HostDirExists(const base::String& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}

// mkdir -p on the host.
void MakeHostDirs(const base::String& path) {
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
void RemoveTree(const base::String& path) {
  DIR* d = ::opendir(path.c_str());
  if (d) {
    while (dirent* e = ::readdir(d)) {
      if (!std::strcmp(e->d_name, ".") || !std::strcmp(e->d_name, ".."))
        continue;
      base::String child = path + "/" + e->d_name;
      struct stat st;
      if (::stat(child.c_str(), &st) == 0 && (st.st_mode & S_IFDIR))
        RemoveTree(child);
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
base::String SaveRoot() {
  if (const char* e = kSavedataDir)
    return e;
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  return base::String(home.empty() ? "." : (const char*)home.c_str()) +
         "/.prosperity/savedata";
}

// The booted title's tag for the save root. the launcher parses TITLE_ID from
// the pkg's (outer) param.sfo; fall back to "SAVEDATA" when it can't be
// determined so per-title layout still works and never produces an empty path
// component.
const base::String& TitleTag() {
  static const base::String kTag = [] {
    base::String t = kern::vfs::TitleId();
    if (Trace())
      BASE_LOGI("savedata", "title id = {}",
                t.empty() ? "(fallback SAVEDATA)" : t.c_str());
    return t.empty() ? base::String("SAVEDATA") : t;
  }();
  return kTag;
}

base::String TitleRoot() {
  return SaveRoot() + "/" + TitleTag();
}

// The host dir for a save. Prefer the per-title path; but if it doesn't exist
// yet and a legacy dirName-only save does (written before per-title roots),
// keep using the legacy path so those saves are not orphaned. `existing` is set
// to whether a save already lives at the chosen path.
base::String ChooseHost(const char* dir_name, bool& existing) {
  const base::String per_title = TitleRoot() + "/" + dir_name;
  if (HostDirExists(per_title)) {
    existing = true;
    return per_title;
  }
  const base::String legacy = SaveRoot() + "/" + dir_name;
  if (HostDirExists(legacy)) {
    existing = true;
    return legacy;  // migrate-in-place: reuse the pre-per-title save
  }
  existing = false;
  return per_title;  // brand-new save -> per-title root
}

// Look up the host directory a mount point ("/savedataN") maps to.
base::String HostForPoint(const void* mount_point) {
  if (!mount_point)
    return {};
  const char* point = static_cast<const char*>(mount_point);
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (auto& s : g_slots)
    if (s.point == point)
      return s.host;
  return {};
}

int UnmountPoint(const void* mount_point) {
  if (!mount_point)
    return kErrParameter;
  const char* point = static_cast<const char*>(mount_point);
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
const void* PtrAt(const void* base, size_t off) {
  const void* p = nullptr;
  std::memcpy(&p, static_cast<const u8*>(base) + off, sizeof(p));
  return p;
}
u32 U32At(const void* base, size_t off) {
  u32 v = 0;
  std::memcpy(&v, static_cast<const u8*>(base) + off, 4);
  return v;
}
u64 U64At(const void* base, size_t off) {
  u64 v = 0;
  std::memcpy(&v, static_cast<const u8*>(base) + off, 8);
  return v;
}

// The char[] a DirName* / MountPoint* points at.
const char* CstrOf(const void* p) {
  return static_cast<const char*>(p);
}

// -------- param sidecar (.sce_param.bin) --------

base::String ParamPath(const base::String& host) {
  return host + "/.sce_param.bin";
}

void LoadParam(const base::String& host, u8* out /*kParamSize*/) {
  std::memset(out, 0, kParamSize);
  if (FILE* f = std::fopen(ParamPath(host).c_str(), "rb")) {
    size_t got = std::fread(out, 1, kParamSize, f);
    (void)got;
    std::fclose(f);
  }
}

void StoreParam(const base::String& host, const u8* blob /*kParamSize*/) {
  if (FILE* f = std::fopen(ParamPath(host).c_str(), "wb")) {
    std::fwrite(blob, 1, kParamSize, f);
    std::fclose(f);
  }
}

// -------- SaveDataMemory backing --------

base::String MemoryPath(u32 slot_id) {
  base::String dir = TitleRoot() + "/sce_sdmemory";
  MakeHostDirs(dir);
  char name[64];
  std::snprintf(name, sizeof(name), "/memory%u.bin", slot_id);
  return dir + name;
}

int MemorySetup(u64 memory_size, u32 slot_id, void* result) {
  const base::String path = MemoryPath(slot_id);
  u64 existed = 0;
  struct stat st;
  if (::stat(path.c_str(), &st) == 0)
    existed = static_cast<u64>(st.st_size);
  // Create/extend the blob to memorySize (fill with zeros).
  if (existed < memory_size) {
    if (FILE* f = std::fopen(path.c_str(), existed ? "rb+" : "wb")) {
      if (std::fseek(f, static_cast<long>(memory_size) - 1, SEEK_SET) == 0) {
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
  if (Trace())
    BASE_LOGI("savedata", "memory setup slot={} size={} (existed={})", slot_id,
              (unsigned long long)memory_size, (unsigned long long)existed);
  return kOk;
}

int MemoryRead(u32 slot_id, void* buf, u64 buf_size, i64 offset) {
  if (buf && buf_size)
    std::memset(buf, 0, buf_size);
  if (!buf)
    return kOk;
  if (FILE* f = std::fopen(MemoryPath(slot_id).c_str(), "rb")) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) == 0) {
      size_t got = std::fread(buf, 1, buf_size, f);
      (void)got;
    }
    std::fclose(f);
  }
  return kOk;
}

int MemoryWrite(u32 slot_id, const void* buf, u64 buf_size, i64 offset) {
  if (!buf || !buf_size)
    return kOk;
  const base::String path = MemoryPath(slot_id);
  FILE* f = std::fopen(path.c_str(), "rb+");
  if (!f)
    f = std::fopen(path.c_str(), "wb");
  if (f) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) == 0)
      std::fwrite(buf, 1, buf_size, f);
    std::fclose(f);
  }
  return kOk;
}

// Shared mount core.
int DoMount(const char* dir_name, u32 mode, void* result) {
  if (!dir_name || !dir_name[0])
    return kErrParameter;
  bool exists = false;
  const base::String host = ChooseHost(dir_name, exists);
  const bool create = (mode & (kModeCreate | kModeCreate2)) != 0;
  const bool read_only = (mode & kModeRdOnly) && !(mode & kModeRdWr);

  if (!exists && !create)
    return kErrNotFound;  // e.g. a read-only "does a save exist?" probe
  if (exists && (mode & kModeCreate))
    return kErrExists;  // strict CREATE requires the save not to exist yet

  base::LockGuard<base::Mutex> lk(g_mtx);
  char point[16];
  std::snprintf(point, sizeof(point), "/savedata%d", g_next_slot++);
  kern::vfs::MountWritable(point, host.c_str());  // creates the host dir
  g_slots.push_back({point, host, read_only});

  if (result) {
    auto* r = static_cast<u8*>(result);
    std::memset(r, 0, 64);
    std::snprintf(reinterpret_cast<char*>(r), 16, "%s", point);
    const u32 status = exists ? 0u : 1u;  // 1 = SAVE_DATA_CREATED
    std::memcpy(r + 28, &status, 4);      // mount_status@28
  }
  if (Trace())
    BASE_LOGI("savedata", "mount dir='{}' mode={:#x} -> {} (host={}, {})",
              dir_name, mode, point, host.c_str(),
              exists ? "existing" : "created");
  return kOk;
}

}  // namespace

// ---------------- init / term ----------------

int PS4ABI sceSaveDataInitialize(void*) {
  SdTrace("sceSaveDataInitialize");
  return kOk;
}
int PS4ABI sceSaveDataInitialize2(void*) {
  SdTrace("sceSaveDataInitialize2");
  return kOk;
}
int PS4ABI sceSaveDataInitialize3(void*) {
  SdTrace("sceSaveDataInitialize3");
  return kOk;
}
int PS4ABI sceSaveDataTerminate() {
  SdTrace("sceSaveDataTerminate");
  return kOk;
}

// ---------------- mount ----------------

int PS4ABI sceSaveDataMount2(const void* mount, void* result) {
  SdTrace("sceSaveDataMount2");
  if (!mount)
    return kErrParameter;
  return DoMount(CstrOf(PtrAt(mount, 8)), U32At(mount, 24), result);
}

int PS4ABI sceSaveDataMount(const void* mount, void* result) {
  SdTrace("sceSaveDataMount");
  if (!mount)
    return kErrParameter;
  return DoMount(CstrOf(PtrAt(mount, 16)), U32At(mount, 40), result);
}

int PS4ABI sceSaveDataMount3(const void* mount, void* result) {
  SdTrace("sceSaveDataMount3");
  if (!mount)
    return kErrParameter;
  return DoMount(CstrOf(PtrAt(mount, 8)), U32At(mount, 32), result);
}

int PS4ABI sceSaveDataMount5(const void* mount, void* result) {
  SdTrace("sceSaveDataMount5");
  return sceSaveDataMount2(mount, result);  // same leading layout for our use
}

int PS4ABI sceSaveDataUmount(const void* mount_point) {
  SdTrace("sceSaveDataUmount");
  return UnmountPoint(mount_point);
}
int PS4ABI sceSaveDataUmountWithBackup(const void* mount_point) {
  SdTrace("sceSaveDataUmountWithBackup");
  return UnmountPoint(mount_point);
}

int PS4ABI sceSaveDataUmount2(u32, const void* mount_point) {
  SdTrace("sceSaveDataUmount2");
  return UnmountPoint(mount_point);
}

int PS4ABI sceSaveDataGetMountInfo(const void*, void* info) {
  SdTrace("sceSaveDataGetMountInfo");
  if (info) {
    auto* i = static_cast<u8*>(info);
    std::memset(i, 0, 48);
    std::memcpy(i + 0, &kTotalBlocks, 8);  // total blocks
    std::memcpy(i + 8, &kFreeBlocks, 8);   // free blocks
  }
  return kOk;
}

int PS4ABI sceSaveDataCreateTransactionResource(u32) {
  SdTrace("sceSaveDataCreateTransactionResource");
  base::LockGuard<base::Mutex> lk(g_mtx);
  return g_next_transaction_resource++;
}

int PS4ABI sceSaveDataDeleteTransactionResource(i32) {
  SdTrace("sceSaveDataDeleteTransactionResource");
  return kOk;
}

int PS4ABI sceSaveDataPrepare(const void* mount_point, const void* param) {
  SdTrace("sceSaveDataPrepare");
  return mount_point && param ? kOk : kErrParameter;
}

int PS4ABI sceSaveDataCommit(const void* param) {
  SdTrace("sceSaveDataCommit");
  return param ? kOk : kErrParameter;
}

// ---------------- enumerate / delete / backup ----------------

int PS4ABI sceSaveDataDirNameSearch(const void* cond, void* result) {
  SdTrace("sceSaveDataDirNameSearch");
  if (!result)
    return kErrParameter;
  auto* r = static_cast<u8*>(result);
  base::String search_root = TitleRoot();
  if (cond) {
    const void* title_id = PtrAt(cond, 8);
    if (title_id && CstrOf(title_id)[0])
      search_root =
          SaveRoot() + "/" +
          base::String(CstrOf(title_id), strnlen(CstrOf(title_id), 10));
  }
  // dirNamesNum@16 is the caller's array capacity (input).
  const u32 capacity = U32At(result, 16);
  auto* dir_names = static_cast<u8*>(const_cast<void*>(PtrAt(result, 8)));
  auto* params = static_cast<u8*>(const_cast<void*>(PtrAt(result, 24)));
  auto* infos = static_cast<u8*>(const_cast<void*>(PtrAt(result, 32)));

  // Optional name filter from the search condition.
  const char* filter = nullptr;
  if (cond) {
    const void* dn = PtrAt(cond, 16);
    if (dn && CstrOf(dn)[0])
      filter = CstrOf(dn);
  }

  // Enumerate existing save directories under the title root.
  base::Vector<base::String> hits;
  if (DIR* d = ::opendir(search_root.c_str())) {
    while (dirent* e = ::readdir(d)) {
      if (e->d_name[0] == '.' || !std::strncmp(e->d_name, "sce_", 4))
        continue;
      base::String child = search_root + "/" + e->d_name;
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
  if (cond && U32At(cond, 28) == 1)
    base::Reverse(hits.begin(), hits.end());

  const u32 set_num =
      capacity < hits.size() ? capacity : static_cast<u32>(hits.size());
  for (u32 i = 0; i < set_num; i++) {
    if (dir_names) {
      char* slot = reinterpret_cast<char*>(dir_names) + i * 32;
      std::memset(slot, 0, 32);
      std::snprintf(slot, 32, "%s", hits[i].c_str());
    }
    if (params) {
      u8* pslot = params + i * kParamSize;
      LoadParam(search_root + "/" + hits[i], pslot);
    }
    if (infos) {
      u8* islot = infos + i * 48;  // SearchInfo { u64 blocks; u64 free; }
      std::memset(islot, 0, 48);
      std::memcpy(islot + 0, &kTotalBlocks, 8);
      std::memcpy(islot + 8, &kFreeBlocks, 8);
    }
  }
  const u32 hit_num = static_cast<u32>(hits.size());
  std::memcpy(r + 0, &hit_num, 4);   // hitNum
  std::memcpy(r + 20, &set_num, 4);  // setNum
  // dirNamesNum stays the caller's capacity value; leave it untouched.
  if (Trace())
    BASE_LOGI("savedata", "dirNameSearch -> {} hit(s), {} returned", hit_num,
              set_num);
  return kOk;
}

int PS4ABI sceSaveDataDelete(const void* del) {
  SdTrace("sceSaveDataDelete");
  if (!del)
    return kErrParameter;
  const void* dn = PtrAt(del, 16);
  if (!dn || !CstrOf(dn)[0])
    return kErrParameter;
  bool exists = false;
  const base::String host = ChooseHost(CstrOf(dn), exists);
  if (exists)
    RemoveTree(host);
  if (Trace())
    BASE_LOGI("savedata", "delete dir='{}'", CstrOf(dn));
  return kOk;
}

int PS4ABI sceSaveDataCheckBackupData(const void* check) {
  SdTrace("sceSaveDataCheckBackupData");
  if (!check)
    return kErrParameter;
  // We keep no separate backup image (the save dir itself persists across
  // umount), so there is never a backup to restore: report NOT_FOUND, which is
  // the correct "no backup" answer and makes titles fall back to a normal load.
  return kErrNotFound;
}

int PS4ABI sceSaveDataRestoreBackupData(const void* restore) {
  SdTrace("sceSaveDataRestoreBackupData");
  if (!restore)
    return kErrParameter;
  return kErrNotFound;  // no backup image (see sceSaveDataCheckBackupData)
}

// ---------------- param / icon ----------------

int PS4ABI sceSaveDataGetParam(const void* mount_point,
                               u32 param_type,
                               void* buf,
                               u64 size,
                               u64* result) {
  SdTrace("sceSaveDataGetParam");
  if (buf && size)
    std::memset(buf, 0, size);
  if (result)
    *result = 0;
  const base::String host = HostForPoint(mount_point);
  if (host.empty())
    return kErrNotMounted;
  if (!buf || !size)
    return kOk;

  u8 blob[kParamSize];
  LoadParam(host, blob);
  size_t off = 0, len = 0;
  switch (param_type) {
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

int PS4ABI sceSaveDataSetParam(const void* mount_point,
                               u32 param_type,
                               const void* buf,
                               u64 size) {
  SdTrace("sceSaveDataSetParam");
  const base::String host = HostForPoint(mount_point);
  if (host.empty())
    return kErrNotMounted;
  if (!buf || !size)
    return kOk;

  u8 blob[kParamSize];
  LoadParam(host, blob);
  size_t off = 0, len = 0;
  switch (param_type) {
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
  StoreParam(host, blob);
  if (Trace())
    BASE_LOGI("savedata", "setParam type={} size={} -> {}", param_type,
              (unsigned long long)size, host.c_str());
  return kOk;
}

int PS4ABI sceSaveDataSaveIcon(const void* mount_point, const void* icon) {
  SdTrace("sceSaveDataSaveIcon");
  const base::String host = HostForPoint(mount_point);
  if (host.empty())
    return kErrNotMounted;
  if (icon) {
    const void* icon_buf = PtrAt(icon, 0);
    const u64 buf_size = U64At(icon, 8);
    const u64 data_size = U64At(icon, 16);
    const u64 n = data_size < buf_size ? data_size : buf_size;
    if (icon_buf && n) {
      if (FILE* f = std::fopen((host + "/.sce_icon.bin").c_str(), "wb")) {
        std::fwrite(icon_buf, 1, static_cast<size_t>(n), f);
        std::fclose(f);
      }
    }
  }
  if (Trace())
    BASE_LOGI("savedata", "saveIcon -> {}", host.c_str());
  return kOk;
}

int PS4ABI sceSaveDataLoadIcon(const void* mount_point, void* icon) {
  SdTrace("sceSaveDataLoadIcon");
  const base::String host = HostForPoint(mount_point);
  if (host.empty())
    return kErrNotMounted;
  if (!icon)
    return kErrParameter;

  const base::String path = host + "/.sce_icon.bin";
  struct stat st;
  if (::stat(path.c_str(), &st) != 0)
    return kErrNotFound;

  void* icon_buf = const_cast<void*>(PtrAt(icon, 0));
  const u64 buf_size = U64At(icon, 8);
  const u64 data_size = static_cast<u64>(st.st_size);
  if (icon_buf && buf_size) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
      return kErrNotFound;
    const u64 copy_size = data_size < buf_size ? data_size : buf_size;
    std::fread(icon_buf, 1, static_cast<size_t>(copy_size), f);
    std::fclose(f);
  }
  std::memcpy(static_cast<u8*>(icon) + 16, &data_size, sizeof(data_size));
  return kOk;
}

// ---------------- SaveDataMemory ----------------

int PS4ABI sceSaveDataSetupSaveDataMemory(u32, u64 memory_size, void*) {
  SdTrace("sceSaveDataSetupSaveDataMemory");
  return MemorySetup(memory_size, 0, nullptr);
}

int PS4ABI sceSaveDataSetupSaveDataMemory2(const void* setup_param,
                                           void* result) {
  SdTrace("sceSaveDataSetupSaveDataMemory2");
  if (!setup_param) {
    if (result)
      std::memset(result, 0, 24);
    return kErrParameter;
  }
  const u64 memory_size = U64At(setup_param, 8);
  const u32 slot_id = U32At(setup_param, 40);
  return MemorySetup(memory_size, slot_id, result);
}

int PS4ABI sceSaveDataGetSaveDataMemory(u32,
                                        void* buf,
                                        u64 buf_size,
                                        i64 offset) {
  SdTrace("sceSaveDataGetSaveDataMemory");
  return MemoryRead(0, buf, buf_size, offset);
}

int PS4ABI sceSaveDataGetSaveDataMemory2(void* get_param) {
  SdTrace("sceSaveDataGetSaveDataMemory2");
  if (!get_param)
    return kErrParameter;
  const u32 slot_id = U32At(get_param, 32);
  const void* data = PtrAt(get_param, 8);
  if (!data)
    return kErrParameter;
  void* buf = const_cast<void*>(PtrAt(data, 0));
  const u64 buf_size = U64At(data, 8);
  const i64 offset = static_cast<i64>(U64At(data, 16));
  return MemoryRead(slot_id, buf, buf_size, offset);
}

int PS4ABI sceSaveDataSetSaveDataMemory(u32,
                                        const void* buf,
                                        u64 buf_size,
                                        i64 offset) {
  SdTrace("sceSaveDataSetSaveDataMemory");
  return MemoryWrite(0, buf, buf_size, offset);
}

int PS4ABI sceSaveDataSetSaveDataMemory2(const void* set_param) {
  SdTrace("sceSaveDataSetSaveDataMemory2");
  if (!set_param)
    return kErrParameter;
  const u32 slot_id = U32At(set_param, 36);
  const void* data = PtrAt(set_param, 8);
  if (!data)
    return kOk;
  const void* buf = PtrAt(data, 0);
  const u64 buf_size = U64At(data, 8);
  const i64 offset = static_cast<i64>(U64At(data, 16));
  return MemoryWrite(slot_id, buf, buf_size, offset);
}

int PS4ABI sceSaveDataSyncSaveDataMemory(void*) {
  SdTrace("sceSaveDataSyncSaveDataMemory");
  return kOk;
}
int PS4ABI sceSaveDataRestoreLoadSaveDataMemory(const void*) {
  SdTrace("sceSaveDataRestoreLoadSaveDataMemory");
  return kOk;
}
