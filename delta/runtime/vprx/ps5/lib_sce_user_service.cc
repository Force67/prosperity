/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 (Prospero) copy of the HLE libSceUserService. The real Prospero .sprx
 * spins in sceUserServiceInitialize waiting on the SceUserService IPMI daemon
 * we don't host. This is a dedicated PS5 copy (separate state + functions) so
 * its behaviour can diverge from the PS4 HLE without any risk to PS4 titles.
 * Registered in the PS5-only registry (MODULE_INIT_PS5); the ps5Layout import
 * resolver force-routes libSceUserService here.
 */

#include "base/arch.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"  // PS4ABI (via <guest_abi.h>), MODULE_INIT_PS5

#include <cstring>

namespace {
constexpr i32 kUserId = 1;
constexpr i32 kInvalidUserId = -1;
constexpr int kNoEvent = 0x80960007;  // SCE_USER_SERVICE_ERROR_NO_EVENT
bool g_login_delivered = false;

int PS4ABI UserServiceInitialize(const void*) {
  return 0;
}
int PS4ABI UserServiceInitialize2(u32, i64, const void*) {
  return 0;
}
int PS4ABI UserServiceTerminate() {
  return 0;
}

int PS4ABI UserServiceGetEvent(void* event_out) {
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

int PS4ABI UserServiceGetLoginUserIdList(void* list_out) {
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

int PS4ABI UserServiceGetInitialUser(i32* user_id) {
  if (user_id)
    *user_id = kUserId;
  return 0;
}

int PS4ABI UserServiceGetForegroundUser(i32* user_id) {
  if (user_id)
    *user_id = kUserId;
  return 0;
}

int PS4ABI UserServiceGetUserName(i32, char* name, u64 size) {
  if (name && size) {
    std::strncpy(name, "Player", size - 1);
    name[size - 1] = '\0';
  }
  return 0;
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_login_delivered);
});
}  // namespace

}  // namespace

static const runtime::vprx::ExportEntry kExports[] = {
    {0x8F760CBB531534DA, (void*)&UserServiceInitialize},
    {0x6B3FF447A7AF899D, (void*)&UserServiceInitialize2},
    {0x6F01634BE6D7F660, (void*)&UserServiceTerminate},
    {0xC87D7B43A356B558, (void*)&UserServiceGetEvent},
    {0x7CF87298A36F2BF0, (void*)&UserServiceGetLoginUserIdList},
    {0x09D5A9D281D61ABD, (void*)&UserServiceGetInitialUser},
    {0x78D6F9DCB4099883, (void*)&UserServiceGetForegroundUser},
    {0xD71C5C3221AED9FA, (void*)&UserServiceGetUserName},
};

MODULE_INIT_PS5(libSceUserService);

extern "C" int g_vprx_anchor_ps5_lib_sce_user_service = 1;
