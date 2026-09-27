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

#include "base/arch.h"
#include "base/environment_variables.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"  // PS4ABI (via <guest_abi.h>), MODULE_INIT_PS5

#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/strings/xstring.h"
#include "kern/vfs.h"

namespace {
// SCE_APP_CONTENT_APPPARAM_ID_SKU_FLAG == 0; 0 = full game, 1 = trial.
constexpr u32 kSkuFlagFull = 0;

// Free space for both areas, in KiB, deliberately 1 GiB: KiB->bytes in 32 bits
// wraps to exactly 0 for any multiple of 4 GiB (reads as "no space"); 1 GiB =
// 0x40000000 bytes, positive as a signed 32-bit count, and plenty for a new
// world.
constexpr u64 kAvailableKb = 1024ull * 1024;

constexpr char kTempPoint[] = "/temp0";

base::String TempHostDir() {
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  base::String root =
      base::String(home.empty() ? "." : (const char*)home.c_str()) +
      "/.prosperity/appcontent";
  const base::String& title = kern::vfs::TitleId();
  return root + "/" + (title.empty() ? base::String("APPCONTENT") : title) +
         "/temp0";
}

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

int PS4ABI AppContentInitialize(const void*, u32* boot_param) {
  if (boot_param)
    *boot_param = 0;
  return 0;
}

// paramId 1..4 are the title's own userDefinedParamN, which on Prospero live in
// /app0/sce_sys/param.json. Scrape the one key rather than pulling in a JSON
// parser: the file is a flat object of "key": value pairs.
i32 UserDefinedParam(u32 n) {
  static const base::String kJson = [] {
    io::File f = kern::vfs::OpenRead("/app0/sce_sys/param.json");
    if (!f.IsOpen())
      return base::String();
    base::String s(static_cast<size_t>(f.GetSize()), '\0');
    f.Read(s.data(), s.size());
    return s;
  }();
  char key[32];
  std::snprintf(key, sizeof(key), "\"userDefinedParam%u\"", n);
  const size_t at = kJson.find(key);
  if (at == base::String::npos)
    return 0;
  const size_t colon = kJson.find(':', at);
  return colon == base::String::npos
             ? 0
             : static_cast<i32>(
                   std::strtol(kJson.c_str() + colon + 1, nullptr, 10));
}

int PS4ABI AppContentAppParamGetInt(u32 param_id, i32* value) {
  if (!value)
    return -1;
  *value = param_id == 0 ? static_cast<i32>(kSkuFlagFull)
                         : UserDefinedParam(param_id);
  return 0;
}

// SceAppContentMountPoint is a char[16] the caller uses as a path prefix.
int PS4ABI AppContentTemporaryDataMount2(u32 /*option*/, void* mount_point) {
  if (!mount_point)
    return -1;
  const base::String host = TempHostDir();
  MakeHostDirs(host);
  kern::vfs::MountWritable(kTempPoint, host.c_str());
  std::memset(mount_point, 0, 16);
  std::memcpy(mount_point, kTempPoint, sizeof(kTempPoint));
  return 0;
}

int PS4ABI AppContentTemporaryDataUnmount(const void*) {
  return 0;
}

int PS4ABI AppContentGetAvailableSpaceKb(const void*, u64* available_kb) {
  if (!available_kb)
    return -1;
  *available_kb = kAvailableKb;
  return 0;
}

}  // namespace

static const runtime::FuncInfo functions[] = {
    {0x47D940F363AB68DB, (void*)&AppContentInitialize},
    {0xF7D6FCD88297A47E, (void*)&AppContentAppParamGetInt},
    {0x6EE61B78B3865A60, (void*)&AppContentTemporaryDataMount2},
    {0x6DCA255CC9A9EAA4, (void*)&AppContentTemporaryDataUnmount},
    {0x49A2A26F6520D322, (void*)&AppContentGetAvailableSpaceKb},
    {0x1A5EB0E62D09A246, (void*)&AppContentGetAvailableSpaceKb},
};

MODULE_INIT_PS5(libSceAppContent);

extern "C" int g_vprx_anchor_ps5_lib_sce_app_content = 1;
