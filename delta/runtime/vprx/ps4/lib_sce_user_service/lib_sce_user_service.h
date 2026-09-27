#pragma once

#include "guest_abi.h"
#include "../../vprx.h"
#include "base/arch.h"

// Minimal libSceUserService HLE: report one logged-in local user. Only the
// login-state queries are overridden; everything else falls through to the LLE
// libSceUserService.sprx. This is the dependency that lets a connected
// controller associate with a user (without it the title loops on login).
int PS4ABI sceUserServiceInitialize(const void *params);
int PS4ABI sceUserServiceInitialize2(u32 a, i64 b, const void *c);
int PS4ABI sceUserServiceTerminate();
int PS4ABI sceUserServiceGetEvent(void *eventOut);
int PS4ABI sceUserServiceGetLoginUserIdList(void *listOut);
int PS4ABI sceUserServiceGetInitialUser(i32 *userId);
int PS4ABI sceUserServiceGetForegroundUser(i32 *userId);
int PS4ABI sceUserServiceGetUserName(i32 userId, char *name, u64 size);

// libSceUserServiceForNpToolkit shares the .sprx (and its "initialized" flag)
// with the library above, so shimming one and not the other leaves NpToolkit2's
// login-event registration reporting NOT_INITIALIZED. See the same pairing in
// libSceNetCtl.
int PS4ABI sceUserServiceRegisterCallbackForNpToolkit(void *func, void *arg);
int PS4ABI sceUserServiceUnregisterCallbackForNpToolkit(void *func);
