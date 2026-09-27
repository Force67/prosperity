#pragma once

/*
 * HLE libSceAudioIn: enough of the microphone API for titles that probe pad or
 * camera audio input. We do not capture host microphone audio; reads return
 * silence and status calls report an open, idle device.
 */

#include "guest_abi.h"
#include "../../vprx.h"
#include "base/arch.h"


extern "C" {

int PS4ABI sceAudioInInit();
int PS4ABI sceAudioInOpen(i32 userId, i32 type, i32 index,
                          u32 length, u32 freq, u32 param);
int PS4ABI sceAudioInInput(i32 handle, void *ptr);
int PS4ABI sceAudioInClose(i32 handle);
int PS4ABI sceAudioInGetStatus(i32 handle, void *status);
int PS4ABI sceAudioInSetConnections(i32 handle, i32 connections);
int PS4ABI sceAudioInGetHandleStatus(i32 handle, void *status);

int PS4ABI sceAudioInDeviceOpen(i32 userId, i32 type, i32 index,
                                u32 length, u32 freq,
                                u32 param);
int PS4ABI sceAudioInDeviceHqOpen(i32 userId, i32 type, i32 index,
                                  u32 length, u32 freq,
                                  u32 param);
int PS4ABI sceAudioInDeviceRead(i32 handle, void *ptr);
int PS4ABI sceAudioInDeviceClose(i32 handle);
int PS4ABI sceAudioInDeviceState(i32 handle, void *state);

}  // extern "C"
