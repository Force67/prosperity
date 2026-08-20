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

#include "../../vprx.h"
#include "base/arch.h"


extern "C" {

i64 PS4ABI sceAvPlayerInit(void *initData);
i64 PS4ABI sceAvPlayerInitEx(const void *initData, i64 *handleOut);
int PS4ABI sceAvPlayerPostInit(i64 handle, void *postInitData);
int PS4ABI sceAvPlayerAddSource(i64 handle, const char *filename);
int PS4ABI sceAvPlayerAddSourceEx(i64 handle, u32 type, void *source);
int PS4ABI sceAvPlayerStart(i64 handle);
int PS4ABI sceAvPlayerStop(i64 handle);
int PS4ABI sceAvPlayerClose(i64 handle);
bool PS4ABI sceAvPlayerIsActive(i64 handle);
bool PS4ABI sceAvPlayerGetVideoData(i64 handle, void *frameInfo);
bool PS4ABI sceAvPlayerGetVideoDataEx(i64 handle, void *frameInfo);
bool PS4ABI sceAvPlayerGetAudioData(i64 handle, void *frameInfo);
u64 PS4ABI sceAvPlayerCurrentTime(i64 handle);
int PS4ABI sceAvPlayerSetLooping(i64 handle, bool loop);
int PS4ABI sceAvPlayerStreamCount(i64 handle);
int PS4ABI sceAvPlayerGetStreamInfo(i64 handle, u32 streamId,
                                    void *info);

// The remaining entry points a title reaches (stream enable/disable, seek,
// pause/resume, sync mode). Each is a thin `int f(handle, ...)` in the real
// .sprx that opens with `mov rdi,[rdi]`, so leaving any of them LLE means the
// module dereferences the sentinel handle above and faults. They only steer
// playback that never happens, so they all report success.
int PS4ABI sceAvPlayerEnableStream(i64 handle, u32 streamId);
int PS4ABI sceAvPlayerDisableStream(i64 handle, u32 streamId);
int PS4ABI sceAvPlayerPause(i64 handle);
int PS4ABI sceAvPlayerResume(i64 handle);
int PS4ABI sceAvPlayerJumpToTime(i64 handle, u64 offsetMs);
int PS4ABI sceAvPlayerSetAvSyncMode(i64 handle, u32 mode);
int PS4ABI sceAvPlayerSetTrickSpeed(i64 handle, int speed);
}  // extern "C"
