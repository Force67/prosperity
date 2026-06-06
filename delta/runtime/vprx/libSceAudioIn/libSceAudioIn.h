#pragma once

/*
 * HLE libSceAudioIn: enough of the microphone API for titles that probe pad or
 * camera audio input. We do not capture host microphone audio; reads return
 * silence and status calls report an open, idle device.
 */

#include "../vprx.h"

#include <cstdint>

extern "C" {

int PS4ABI sceAudioInInit();
int PS4ABI sceAudioInOpen(int32_t userId, int32_t type, int32_t index,
                          uint32_t length, uint32_t freq, uint32_t param);
int PS4ABI sceAudioInInput(int32_t handle, void *ptr);
int PS4ABI sceAudioInClose(int32_t handle);
int PS4ABI sceAudioInGetStatus(int32_t handle, void *status);
int PS4ABI sceAudioInSetConnections(int32_t handle, int32_t connections);
int PS4ABI sceAudioInGetHandleStatus(int32_t handle, void *status);

int PS4ABI sceAudioInDeviceOpen(int32_t userId, int32_t type, int32_t index,
                                uint32_t length, uint32_t freq,
                                uint32_t param);
int PS4ABI sceAudioInDeviceHqOpen(int32_t userId, int32_t type, int32_t index,
                                  uint32_t length, uint32_t freq,
                                  uint32_t param);
int PS4ABI sceAudioInDeviceRead(int32_t handle, void *ptr);
int PS4ABI sceAudioInDeviceClose(int32_t handle);
int PS4ABI sceAudioInDeviceState(int32_t handle, void *state);

}  // extern "C"
