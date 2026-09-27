#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceAudioOut: PS4 audio output. The real module talks to the audio DSP
 * / a kernel audio device we don't emulate (same situation as libSceVideoOut,
 * which is also HLE'd), so we bridge sceAudioOutOpen/Output to a host SDL3
 * audio device (delta_host's gfx_audio). This is the audio analogue of the
 * graphics HLE exception to the keep-PRX-LLE rule.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"

extern "C" {

int PS4ABI sceAudioOutInit();
int PS4ABI sceAudioOutOpen(i32 user_id,
                           i32 type,
                           i32 index,
                           u32 length,
                           u32 freq,
                           u32 param);
int PS4ABI sceAudioOutOutput(i32 handle, const void* ptr);
int PS4ABI sceAudioOutClose(i32 handle);
int PS4ABI sceAudioOutSetVolume(i32 handle, i32 flag, i32* vol);
int PS4ABI sceAudioOutOutputs(void* params, u32 num);
int PS4ABI sceAudioOutGetPortState(i32 handle, void* state);
i64 PS4ABI sceAudioOutGetLastOutputTime(i32 handle);
int PS4ABI sceAudioOutSetVolumeDc(i32 handle, void* p);
int PS4ABI sceAudioOutInitIpmiGetSession(i32 arg);

}  // extern "C"
