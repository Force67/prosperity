#pragma once

/*
 * HLE libSceAvPlayer: the real module decodes H.264 video + Atrac9 audio (via
 * libSceAjm) which we don't run, so its decode threads crash on a buffer the
 * backend never filled. Titles use it for non-interactive intro/cutscene movies.
 * We stub the whole player to report "playback finished immediately": Init hands
 * back a dummy handle, AddSource/PostInit/Close succeed, and IsActive returns
 * false so the title's `while (sceAvPlayerIsActive()) { drawFrame }` loop is
 * skipped and it proceeds straight to the menu. No frames are produced.
 */

#include "../vprx.h"

#include <cstdint>

extern "C" {

int64_t PS4ABI sceAvPlayerInit(void *initData);
int64_t PS4ABI sceAvPlayerInitEx(const void *initData, int64_t *handleOut);
int PS4ABI sceAvPlayerPostInit(int64_t handle, void *postInitData);
int PS4ABI sceAvPlayerAddSource(int64_t handle, const char *filename);
int PS4ABI sceAvPlayerAddSourceEx(int64_t handle, uint32_t type, void *source);
int PS4ABI sceAvPlayerStart(int64_t handle);
int PS4ABI sceAvPlayerStop(int64_t handle);
int PS4ABI sceAvPlayerClose(int64_t handle);
bool PS4ABI sceAvPlayerIsActive(int64_t handle);
bool PS4ABI sceAvPlayerGetVideoData(int64_t handle, void *frameInfo);
bool PS4ABI sceAvPlayerGetVideoDataEx(int64_t handle, void *frameInfo);
bool PS4ABI sceAvPlayerGetAudioData(int64_t handle, void *frameInfo);
uint64_t PS4ABI sceAvPlayerCurrentTime(int64_t handle);
int PS4ABI sceAvPlayerSetLooping(int64_t handle, bool loop);
int PS4ABI sceAvPlayerStreamCount(int64_t handle);
int PS4ABI sceAvPlayerGetStreamInfo(int64_t handle, uint32_t streamId,
                                    void *info);
}  // extern "C"
