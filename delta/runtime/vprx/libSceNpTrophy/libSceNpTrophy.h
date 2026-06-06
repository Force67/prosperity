#pragma once

/*
 * HLE libSceNpTrophy: the real module needs an NP service backend we don't run,
 * so its calls fail with NOT_INITIALIZED (0x80551601). Titles that init trophies
 * for the logged-in user then loop forever (e.g. GameMaker compares the context's
 * user id against the current login user and re-inits when they differ). We model
 * the context/handle lifecycle synchronously so trophy init succeeds; we don't
 * persist real trophies (unlocks are accepted and dropped, queries report an
 * empty set), which is enough to get titles past startup.
 */

#include "../vprx.h"

#include <cstdint>

extern "C" {

int PS4ABI sceNpTrophyCreateContext(int32_t *context, int32_t userId,
                                    uint32_t serviceLabel, uint64_t options);
int PS4ABI sceNpTrophyCreateHandle(int32_t *handle);
int PS4ABI sceNpTrophyDestroyContext(int32_t context);
int PS4ABI sceNpTrophyDestroyHandle(int32_t handle);
int PS4ABI sceNpTrophyAbortHandle(int32_t handle);
int PS4ABI sceNpTrophyRegisterContext(int32_t context, int32_t handle,
                                      uint64_t options);
int PS4ABI sceNpTrophyUnlockTrophy(int32_t context, int32_t handle,
                                   int32_t trophyId, int32_t *platinumId);
int PS4ABI sceNpTrophyGetTrophyUnlockState(int32_t context, int32_t handle,
                                           void *flags, uint32_t *count);
int PS4ABI sceNpTrophyGetGameInfo(int32_t context, int32_t handle,
                                  void *details, void *data);
int PS4ABI sceNpTrophyGetTrophyInfo(int32_t context, int32_t handle,
                                    int32_t trophyId, void *details,
                                    void *data);
int PS4ABI sceNpTrophyGetGroupInfo(int32_t context, int32_t handle,
                                   int32_t groupId, void *details, void *data);
int PS4ABI sceNpTrophyGetGameIcon(int32_t context, int32_t handle, void *buffer,
                                  uint64_t *size);
int PS4ABI sceNpTrophyGetGroupIcon(int32_t context, int32_t handle,
                                   int32_t groupId, void *buffer,
                                   uint64_t *size);
int PS4ABI sceNpTrophyGetTrophyIcon(int32_t context, int32_t handle,
                                    int32_t trophyId, void *buffer,
                                    uint64_t *size);
int PS4ABI sceNpTrophyCaptureScreenshot(int32_t a, void *b, void *c);
int PS4ABI sceNpTrophyShowTrophyList(int32_t context, int32_t handle);
}  // extern "C"
