#include "libSceUserService.h"

#include <cstdint>
#include <cstring>

// A single fixed local user. The PS4 user service normally tracks PSN/local
// users; the title queries the login list and the foreground/initial user to
// associate a controller and proceed past the "press start" sign-in. We report
// one user (id 1) logged in and deliver exactly one LOGIN event.
namespace {
constexpr int32_t kUserId = 1;
constexpr int32_t kInvalidUserId = -1;
constexpr int kNoEvent = 0x80960013;  // SCE_USER_SERVICE_ERROR_NO_EVENT
bool g_loginDelivered = false;
}  // namespace

int PS4ABI sceUserServiceGetEvent(void *eventOut) {
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

int PS4ABI sceUserServiceGetLoginUserIdList(void *listOut) {
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

int PS4ABI sceUserServiceGetInitialUser(int32_t *userId) {
  if (userId)
    *userId = kUserId;
  return 0;
}

int PS4ABI sceUserServiceGetForegroundUser(int32_t *userId) {
  if (userId)
    *userId = kUserId;
  return 0;
}

int PS4ABI sceUserServiceGetUserName(int32_t userId, char *name, uint64_t size) {
  if (name && size) {
    std::strncpy(name, "Player", size - 1);
    name[size - 1] = '\0';
  }
  return 0;
}
