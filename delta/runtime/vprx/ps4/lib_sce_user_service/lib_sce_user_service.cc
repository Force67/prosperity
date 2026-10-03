#include "runtime/vprx/ps4/lib_sce_user_service/lib_sce_user_service.h"
#include "base/arch.h"
#include "guest/session.h"
#include "guest_abi.h"

#include <cstring>

// A single fixed local user. The PS4 user service normally tracks PSN/local
// users; the title queries the login list and the foreground/initial user to
// associate a controller and proceed past the "press start" sign-in. We report
// one user (id 1) logged in and deliver exactly one LOGIN event.
namespace {
constexpr i32 kUserId = 1;
constexpr i32 kInvalidUserId = -1;
constexpr int kNoEvent = 0x80960007;  // SCE_USER_SERVICE_ERROR_NO_EVENT
bool g_login_delivered = false;

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_login_delivered);
});
}  // namespace

}  // namespace

// Fully take over init/teardown so the LLE UserService never sets up its IPMI
// client (whose login round-trip to a non-existent system daemon spins forever
// once a controller appears).
int PS4ABI sceUserServiceInitialize(const void* params) {
  return 0;
}
int PS4ABI sceUserServiceInitialize2(u32 a, i64 b, const void* c) {
  return 0;
}
int PS4ABI sceUserServiceTerminate() {
  return 0;
}

int PS4ABI sceUserServiceGetEvent(void* event_out) {
  struct Event {
    i32 event_type;  // 0 = LOGIN, 1 = LOGOUT
    i32 user_id;
  };
  auto* e = static_cast<Event*>(event_out);
  if (!e)
    return -1;
  if (!g_login_delivered) {
    g_login_delivered = true;
    e->event_type = 0;  // LOGIN
    e->user_id = kUserId;
    return 0;
  }
  return kNoEvent;  // drained
}

int PS4ABI sceUserServiceGetLoginUserIdList(void* list_out) {
  struct List {
    i32 user_id[4];
  };
  auto* l = static_cast<List*>(list_out);
  if (!l)
    return -1;
  l->user_id[0] = kUserId;
  l->user_id[1] = l->user_id[2] = l->user_id[3] = kInvalidUserId;
  return 0;
}

int PS4ABI sceUserServiceGetInitialUser(i32* user_id) {
  if (user_id)
    *user_id = kUserId;
  return 0;
}

int PS4ABI sceUserServiceGetForegroundUser(i32* user_id) {
  if (user_id)
    *user_id = kUserId;
  return 0;
}

int PS4ABI sceUserServiceGetUserName(i32 user_id, char* name, u64 size) {
  if (name && size) {
    std::strncpy(name, "Player", size - 1);
    name[size - 1] = '\0';
  }
  return 0;
}

// The one local user is logged in before the title starts and never changes,
// so a registered login/logout callback has nothing to deliver.
int PS4ABI sceUserServiceRegisterCallbackForNpToolkit(void* func, void* arg) {
  return 0;
}

int PS4ABI sceUserServiceUnregisterCallbackForNpToolkit(void* func) {
  return 0;
}
