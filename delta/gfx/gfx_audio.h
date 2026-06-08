#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Host audio output bridge (SDL3). The libSceAudioOut HLE (delta_runtime) drives
 * these to play the game's PCM. Mirrors the videoout flip bridge: the HLE module
 * stays free of SDL, the device lives here in delta_gfx (which links SDL3). On
 * Android (no SDL in the gfx build) these are no-ops for now.
 */

#include <cstdint>

extern "C" {

// Open a playback port: freq Hz, channels (1/2/8), isFloat (0 = S16, 1 = F32).
// Returns a handle >= 0, or -1 on failure.
int prosperity_audio_open(uint32_t freq, uint32_t channels, int isFloat);

// Queue `frames` interleaved frames (frames*channels samples) for playback.
// Returns frames queued, or -1 on a bad handle. Bounds latency internally.
int prosperity_audio_output(int handle, const void *samples, uint32_t frames);

// Set a port's linear gain (0..1), applied on output. flag selects channels;
// we apply it uniformly.
void prosperity_audio_volume(int handle, float gain);

void prosperity_audio_close(int handle);

}  // extern "C"
