#pragma once

/*
 * HLE libSceNpTrophy: the real module needs an NP service backend we don't run,
 * so its calls fail with NOT_INITIALIZED (0x80551601). Titles that init
 * trophies for the logged-in user then loop forever (e.g. GameMaker compares
 * the context's user id against the current login user and re-inits when they
 * differ). We model the context/handle lifecycle synchronously so trophy init
 * succeeds; we don't persist real trophies (unlocks are accepted and dropped,
 * queries report an empty set), which is enough to get titles past startup.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"

extern "C" {

int PS4ABI sceNpTrophyCreateContext(i32* context,
                                    i32 user_id,
                                    u32 service_label,
                                    u64 options);
int PS4ABI sceNpTrophyCreateHandle(i32* handle);
int PS4ABI sceNpTrophyDestroyContext(i32 context);
int PS4ABI sceNpTrophyDestroyHandle(i32 handle);
int PS4ABI sceNpTrophyAbortHandle(i32 handle);
int PS4ABI sceNpTrophyRegisterContext(i32 context, i32 handle, u64 options);
int PS4ABI sceNpTrophyUnlockTrophy(i32 context,
                                   i32 handle,
                                   i32 trophy_id,
                                   i32* platinum_id);
int PS4ABI sceNpTrophyGetTrophyUnlockState(i32 context,
                                           i32 handle,
                                           void* flags,
                                           u32* count);
int PS4ABI sceNpTrophyGetGameInfo(i32 context,
                                  i32 handle,
                                  void* details,
                                  void* data);
int PS4ABI sceNpTrophyGetTrophyInfo(i32 context,
                                    i32 handle,
                                    i32 trophy_id,
                                    void* details,
                                    void* data);
int PS4ABI sceNpTrophyGetGroupInfo(i32 context,
                                   i32 handle,
                                   i32 group_id,
                                   void* details,
                                   void* data);
int PS4ABI sceNpTrophyGetGameIcon(i32 context,
                                  i32 handle,
                                  void* buffer,
                                  u64* size);
int PS4ABI sceNpTrophyGetGroupIcon(i32 context,
                                   i32 handle,
                                   i32 group_id,
                                   void* buffer,
                                   u64* size);
int PS4ABI sceNpTrophyGetTrophyIcon(i32 context,
                                    i32 handle,
                                    i32 trophy_id,
                                    void* buffer,
                                    u64* size);
int PS4ABI sceNpTrophyCaptureScreenshot(i32 a, void* b, void* c);
int PS4ABI sceNpTrophyShowTrophyList(i32 context, i32 handle);
}  // extern "C"
