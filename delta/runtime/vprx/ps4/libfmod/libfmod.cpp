/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libfmod. See libfmod.h. FMOD_OK == 0, and almost every FMOD entry point
 * returns an FMOD_RESULT, so a blanket "return 0" makes the game's audio layer
 * believe each call succeeded. Functions that hand back an object via an
 * out-pointer get specific handlers in libfmod_api.cpp.
 */

#include "libfmod.h"

extern "C" {

uint64_t PS4ABI fmodStub() { return 0; /* FMOD_OK */ }

}  // extern "C"
