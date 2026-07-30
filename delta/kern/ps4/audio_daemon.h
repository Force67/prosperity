#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Stand-in for the console's system audio daemon, for the case where the REAL
 * libSceAudioOut.sprx runs (DELTA_LLE=libSceAudioOut) instead of our HLE shim.
 *
 * The real module never touches a device node. It creates POSIX shm regions,
 * mixes into them, and waits on a named event flag for the daemon to say "block
 * taken". Nothing in kern/ps4/dev can see that, so with no daemon the title runs
 * fine and is silent. This subsystem is that missing consumer: it notices the
 * regions being created, decodes the control block, and pumps the blocks into
 * the same host sink (gfx_audio -> SDL) the HLE shim uses.
 *
 * It cannot collide with the HLE shim: only the real module ever creates
 * "/shm_<pid>_C", and when the shim is in play (the default) the module is never
 * loaded, so the daemon never starts and never opens an SDL stream.
 */

#include <cstddef>
#include <cstdint>

namespace krnl {

// Called from the POSIX-shm syscalls whenever a named region gets or changes its
// host backing (`base` = null means the name went away). Names that are not part
// of the libSceAudioOut protocol are ignored; the first control region starts the
// daemon thread.
void audioDaemonNoticeShm(const char *name, uint8_t *base, size_t size);

}  // namespace krnl
