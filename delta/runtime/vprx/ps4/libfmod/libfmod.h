#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libfmod (the game's bundled FMOD .prx). FMOD's real PS4 init registers an
 * ATRAC9 codec through the AJM hardware decoder (/dev/ajm), which we don't
 * emulate -- the failure makes Doom64 (KEX) treat audio init as fatal and abort
 * before it ever renders. We stub FMOD's API to "succeed" with null audio so the
 * game boots; no sound plays. See libfmod.cpp.
 */

#include "../../vprx.h"

#include <cstdint>

extern "C" {
// Generic FMOD export stub: returns FMOD_OK (0). Used for every imported NID
// during discovery; out-param-bearing calls get specific handlers (see .cpp).
uint64_t PS4ABI fmodStub();
}  // extern "C"
