/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE override for libSceAppContent. The real .sprx answers these
 * through a system service we don't host, so its calls fail: a title that reads
 * "cannot tell" as "no space" or "trial" locks the features neither gets.
 * Minecraft (PPSA17221) asks for the free space of its temporary-data and
 * download-data areas and, on failure, greys out "create new world" and puts up
 * an out-of-storage modal.
 *
 * These six entry points are exactly the ones Minecraft imports; the rest of
 * libSceAppContent stays LLE.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>
#include <cstring>
#include <string>
#include <sys/stat.h>

#include "kern/vfs.h"

namespace {
// SCE_APP_CONTENT_APPPARAM_ID_SKU_FLAG == 0; 0 = full game, 1 = trial.
constexpr uint32_t kSkuFlagFull = 0;

// Free space reported for both areas, in KiB (32 GiB).
constexpr uint64_t kAvailableKb = 32ull * 1024 * 1024;

constexpr char kTempPoint[] = "/temp0";

std::string tempHostDir() {
  const char *home = std::getenv("HOME");
  std::string root = std::string(home ? home : ".") + "/.prosperity/appcontent";
  const std::string &title = krnl::vfs::titleId();
  return root + "/" + (title.empty() ? std::string("APPCONTENT") : title) +
         "/temp0";
}

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

int PS4ABI appContentInitialize(const void *, uint32_t *bootParam) {
  if (bootParam)
    *bootParam = 0;
  return 0;
}

int PS4ABI appContentAppParamGetInt(uint32_t /*paramId*/, int32_t *value) {
  if (!value)
    return -1;
  *value = static_cast<int32_t>(kSkuFlagFull);
  return 0;
}

// SceAppContentMountPoint is a char[16] the caller uses as a path prefix.
int PS4ABI appContentTemporaryDataMount2(uint32_t /*option*/, void *mountPoint) {
  if (!mountPoint)
    return -1;
  const std::string host = tempHostDir();
  makeHostDirs(host);
  krnl::vfs::mountWritable(kTempPoint, host.c_str());
  std::memset(mountPoint, 0, 16);
  std::memcpy(mountPoint, kTempPoint, sizeof(kTempPoint));
  return 0;
}

int PS4ABI appContentTemporaryDataUnmount(const void *) { return 0; }

int PS4ABI appContentGetAvailableSpaceKb(const void *, uint64_t *availableKb) {
  if (!availableKb)
    return -1;
  *availableKb = kAvailableKb;
  return 0;
}
}  // namespace

static const runtime::funcInfo functions[] = {
    {0x47D940F363AB68DB, (void *)&appContentInitialize},
    {0xF7D6FCD88297A47E, (void *)&appContentAppParamGetInt},
    {0x6EE61B78B3865A60, (void *)&appContentTemporaryDataMount2},
    {0x6DCA255CC9A9EAA4, (void *)&appContentTemporaryDataUnmount},
    {0x49A2A26F6520D322, (void *)&appContentGetAvailableSpaceKb},
    {0x1A5EB0E62D09A246, (void *)&appContentGetAvailableSpaceKb},
};

MODULE_INIT_PS5(libSceAppContent);

extern "C" int vprx_anchor_ps5_libSceAppContent = 1;
