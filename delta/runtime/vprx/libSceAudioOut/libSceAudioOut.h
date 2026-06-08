#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceAudioOut: PS4 audio output. The real module talks to the audio DSP /
 * a kernel audio device we don't emulate (same situation as libSceVideoOut, which
 * is also HLE'd), so we bridge sceAudioOutOpen/Output to a host SDL3 audio device
 * (delta_gfx's gfx_audio). This is the audio analogue of the graphics HLE
 * exception to the keep-PRX-LLE rule.
 */

#include "../vprx.h"

#include <cstdint>

extern "C" {

int PS4ABI sceAudioOutInit();
int PS4ABI sceAudioOutOpen(int32_t userId, int32_t type, int32_t index,
                           uint32_t length, uint32_t freq, uint32_t param);
int PS4ABI sceAudioOutOutput(int32_t handle, const void *ptr);
int PS4ABI sceAudioOutClose(int32_t handle);
int PS4ABI sceAudioOutSetVolume(int32_t handle, int32_t flag, int32_t *vol);
int PS4ABI sceAudioOutOutputs(void *params, uint32_t num);
int PS4ABI sceAudioOutGetPortState(int32_t handle, void *state);
int64_t PS4ABI sceAudioOutGetLastOutputTime(int32_t handle);
int PS4ABI sceAudioOutSetVolumeDc(int32_t handle, void *p);
int PS4ABI sceAudioOutInitIpmiGetSession(int32_t arg);

}  // extern "C"
