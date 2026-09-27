#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Host audio output bridge (SDL3). The libSceAudioOut HLE (delta_runtime) drives
 * these to play the game's PCM. Mirrors the videoout flip bridge: the HLE module
 * stays free of SDL, the device lives here in delta_host (which links SDL3). On
 * Android (no SDL) these are no-ops.
 */

#include "base/arch.h"

namespace host {

// Open a playback port: freq Hz, channels (1/2/8), isFloat (0 = S16, 1 = F32).
// Returns a handle >= 0, or -1 on failure.
int OpenAudioPort(u32 freq, u32 channels, int isFloat);

// Queue `frames` interleaved frames (frames*channels samples) for playback.
// Returns frames queued, or -1 on a bad handle. Bounds latency internally.
int QueueAudio(int handle, const void *samples, u32 frames);

// Set a port's linear gain (0..1), applied on output. flag selects channels;
// we apply it uniformly.
void SetAudioPortVolume(int handle, float gain);

void CloseAudioPort(int handle);

}  // namespace host
