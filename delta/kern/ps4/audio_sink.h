/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Where the audio daemon sends decoded PCM.
//
// The device itself is an SDL3 stream and lives in delta_gfx, which is the
// module that links SDL. kern may not name it, so the daemon is handed these
// four operations by the composition root instead. Unset (the default in a
// test or a headless tool) means the daemon runs and discards, which is what
// makes it constructible without a graphics stack.

#include "base/arch.h"

namespace krnl::ps4 {

struct AudioSink {
  int (*open)(u32 freq, u32 channels, int isFloat) = nullptr;
  int (*output)(int handle, const void *samples, u32 frames) = nullptr;
  void (*volume)(int handle, float gain) = nullptr;
  void (*close)(int handle) = nullptr;
};

void setAudioSink(const AudioSink &sink);
const AudioSink &audioSink();

}  // namespace krnl::ps4
