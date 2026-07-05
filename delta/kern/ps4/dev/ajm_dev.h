#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "device.h"

namespace krnl {
class proc;

// /dev/ajm: the PS4 audio-decoder (AJM) management device. libSceAjm opens it
// during init (FMOD on PS4 routes its decoding through AJM). We don't decode any
// audio, but the open must SUCCEED -- a soft ENOENT here makes libSceAjm fail,
// which makes FMOD's System::init return FMOD_ERR_INTERNAL, which makes Doom64
// treat audio init as fatal and exit its main thread before it ever renders.
// Open + benign ioctls (a non-zero AJM context handle, zeroed batch results) let
// init complete; no sound plays.
class ajmDevice : public device {
public:
  ajmDevice(proc *);
  int32_t ioctl(uint32_t command, void *args) override;
};
} // namespace krnl
