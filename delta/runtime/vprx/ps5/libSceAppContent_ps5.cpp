/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE override for libSceAppContent. The real .sprx answers through a
 * system service we don't host, so every call fails and a title that reads
 * "cannot tell" as "trial" or "no space" locks features neither state gets.
 *
 * These six entry points are exactly the ones Minecraft (PPSA17221) imports;
 * the rest of libSceAppContent stays LLE.
 */

#include <base/environment_variables.h>
#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

#include "kern/vfs.h"

namespace {
// SCE_APP_CONTENT_APPPARAM_ID_SKU_FLAG == 0; 0 = full game, 1 = trial.
constexpr uint32_t kSkuFlagFull = 0;

// Free space reported for both areas, in KiB. Deliberately 1 GiB and not
// something larger: a caller that converts KiB to bytes in 32 bits wraps to
// exactly 0 for any value that is a multiple of 4 GiB, which reads as "no
// space". 1 GiB is 0x40000000 bytes -- positive even as a signed 32-bit count,
// and far more than a new world needs.
constexpr uint64_t kAvailableKb = 1024ull * 1024;

constexpr char kTempPoint[] = "/temp0";

std::string tempHostDir() {
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  std::string root =
      std::string(home.empty() ? "." : (const char *)home.c_str()) +
      "/.prosperity/appcontent";
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

// paramId 1..4 are the title's own userDefinedParamN, which on Prospero live in
// /app0/sce_sys/param.json. Scrape the one key rather than pulling in a JSON
// parser: the file is a flat object of "key": value pairs.
int32_t userDefinedParam(uint32_t n) {
  static const std::string json = [] {
    utl::File f = krnl::vfs::openRead("/app0/sce_sys/param.json");
    if (!f.IsOpen())
      return std::string();
    std::string s(static_cast<size_t>(f.GetSize()), '\0');
    f.Read(s.data(), s.size());
    return s;
  }();
  char key[32];
  std::snprintf(key, sizeof(key), "\"userDefinedParam%u\"", n);
  const size_t at = json.find(key);
  if (at == std::string::npos)
    return 0;
  const size_t colon = json.find(':', at);
  return colon == std::string::npos
             ? 0
             : static_cast<int32_t>(std::strtol(json.c_str() + colon + 1,
                                                nullptr, 10));
}

int PS4ABI appContentAppParamGetInt(uint32_t paramId, int32_t *value) {
  if (!value)
    return -1;
  *value = paramId == 0 ? static_cast<int32_t>(kSkuFlagFull)
                        : userDefinedParam(paramId);
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
