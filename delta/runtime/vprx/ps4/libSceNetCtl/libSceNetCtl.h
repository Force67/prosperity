#pragma once

#include "../../vprx.h"
#include "base/arch.h"

// libSceNetCtl HLE: report a fully configured wired network (IP_OBTAINED, static LAN).
// The LLE .sprx asks the system net daemon over IPMI, which doesn't exist here, so its
// state never leaves DISCONNECTED; titles gating boot on connectivity (PT polls
// GetState up to 10s, then a broken init path) stall or break without this.
int PS4ABI sceNetCtlInit();
int PS4ABI sceNetCtlTerm();
int PS4ABI sceNetCtlGetState(i32 *state);
int PS4ABI sceNetCtlGetInfo(i32 code, void *info);
int PS4ABI sceNetCtlRegisterCallback(void *func, void *arg, i32 *cid);
int PS4ABI sceNetCtlUnregisterCallback(i32 cid);
int PS4ABI sceNetCtlCheckCallback();

// libSceNetCtlForNpToolkit is a second entry set into the same .sprx and checks the
// same internal initialized flag, which HLE-ing the plain library leaves unset.
// NpToolkit2 registers a link-state callback there during initialize() and treats
// NOT_INITIALIZED as fatal (what takes GTA:SA down).
int PS4ABI sceNetCtlRegisterCallbackForNpToolkit(void *func, void *arg,
                                                 i32 *cid);
int PS4ABI sceNetCtlUnregisterCallbackForNpToolkit(i32 cid);
int PS4ABI sceNetCtlCheckCallbackForNpToolkit();
