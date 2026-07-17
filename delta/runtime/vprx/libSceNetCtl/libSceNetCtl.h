#pragma once

#include "../vprx.h"

// libSceNetCtl HLE: report a fully configured wired network (state IP_OBTAINED
// with a static LAN config). The LLE libSceNetCtl.sprx asks the system's net
// daemon over IPMI, which doesn't exist in our env, so its state never leaves
// DISCONNECTED -- titles that gate boot on connectivity (PT polls GetState for
// up to 10s, then continues down a broken init path) stall or break without
// this.
int PS4ABI sceNetCtlInit();
int PS4ABI sceNetCtlTerm();
int PS4ABI sceNetCtlGetState(int32_t *state);
int PS4ABI sceNetCtlGetInfo(int32_t code, void *info);
int PS4ABI sceNetCtlRegisterCallback(void *func, void *arg, int32_t *cid);
int PS4ABI sceNetCtlUnregisterCallback(int32_t cid);
int PS4ABI sceNetCtlCheckCallback();
