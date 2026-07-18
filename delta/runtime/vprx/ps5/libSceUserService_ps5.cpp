/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 (Prospero) copy of the HLE libSceUserService. The real Prospero .sprx spins
 * in sceUserServiceInitialize waiting on the SceUserService IPMI daemon we don't
 * host. This is a dedicated PS5 copy (separate state + functions) so its behaviour
 * can diverge from the PS4 HLE without any risk to PS4 titles. Registered in the
 * PS5-only registry (MODULE_INIT_PS5); the ps5Layout import resolver force-routes
 * libSceUserService here.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>
#include <cstring>

namespace {
constexpr int32_t kUserId = 1;
constexpr int32_t kInvalidUserId = -1;
constexpr int kNoEvent = 0x80960007;  // SCE_USER_SERVICE_ERROR_NO_EVENT
bool g_loginDelivered = false;

int PS4ABI userServiceInitialize(const void *) { return 0; }
int PS4ABI userServiceInitialize2(uint32_t, int64_t, const void *) { return 0; }
int PS4ABI userServiceTerminate() { return 0; }

int PS4ABI userServiceGetEvent(void *eventOut) {
  struct Event {
    int32_t eventType;  // 0 = LOGIN, 1 = LOGOUT
    int32_t userId;
  };
  auto *e = static_cast<Event *>(eventOut);
  if (!e)
    return -1;
  if (!g_loginDelivered) {
    g_loginDelivered = true;
    e->eventType = 0;  // LOGIN
    e->userId = kUserId;
    return 0;
  }
  return kNoEvent;  // drained
}

int PS4ABI userServiceGetLoginUserIdList(void *listOut) {
  struct List {
    int32_t userId[4];
  };
  auto *l = static_cast<List *>(listOut);
  if (!l)
    return -1;
  l->userId[0] = kUserId;
  l->userId[1] = l->userId[2] = l->userId[3] = kInvalidUserId;
  return 0;
}

int PS4ABI userServiceGetInitialUser(int32_t *userId) {
  if (userId)
    *userId = kUserId;
  return 0;
}

int PS4ABI userServiceGetForegroundUser(int32_t *userId) {
  if (userId)
    *userId = kUserId;
  return 0;
}

int PS4ABI userServiceGetUserName(int32_t, char *name, uint64_t size) {
  if (name && size) {
    std::strncpy(name, "Player", size - 1);
    name[size - 1] = '\0';
  }
  return 0;
}
}  // namespace

static const runtime::funcInfo functions[] = {
    {0x8F760CBB531534DA, (void *)&userServiceInitialize},
    {0x6B3FF447A7AF899D, (void *)&userServiceInitialize2},
    {0x6F01634BE6D7F660, (void *)&userServiceTerminate},
    {0xC87D7B43A356B558, (void *)&userServiceGetEvent},
    {0x7CF87298A36F2BF0, (void *)&userServiceGetLoginUserIdList},
    {0x09D5A9D281D61ABD, (void *)&userServiceGetInitialUser},
    {0x78D6F9DCB4099883, (void *)&userServiceGetForegroundUser},
    {0xD71C5C3221AED9FA, (void *)&userServiceGetUserName},
};

MODULE_INIT_PS5(libSceUserService);

extern "C" int vprx_anchor_ps5_libSceUserService = 1;
