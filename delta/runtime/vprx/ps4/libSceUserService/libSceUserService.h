#pragma once

#include "../../vprx.h"

// Minimal libSceUserService HLE: report one logged-in local user. Only the
// login-state queries are overridden; everything else falls through to the LLE
// libSceUserService.sprx. This is the dependency that lets a connected
// controller associate with a user (without it the title loops on login).
int PS4ABI sceUserServiceInitialize(const void *params);
int PS4ABI sceUserServiceInitialize2(uint32_t a, int64_t b, const void *c);
int PS4ABI sceUserServiceTerminate();
int PS4ABI sceUserServiceGetEvent(void *eventOut);
int PS4ABI sceUserServiceGetLoginUserIdList(void *listOut);
int PS4ABI sceUserServiceGetInitialUser(int32_t *userId);
int PS4ABI sceUserServiceGetForegroundUser(int32_t *userId);
int PS4ABI sceUserServiceGetUserName(int32_t userId, char *name, uint64_t size);

// libSceUserServiceForNpToolkit shares the .sprx (and its "initialized" flag)
// with the library above, so shimming one and not the other leaves NpToolkit2's
// login-event registration reporting NOT_INITIALIZED. See the same pairing in
// libSceNetCtl.
int PS4ABI sceUserServiceRegisterCallbackForNpToolkit(void *func, void *arg);
int PS4ABI sceUserServiceUnregisterCallbackForNpToolkit(void *func);
