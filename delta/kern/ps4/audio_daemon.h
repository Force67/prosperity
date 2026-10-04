#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * System audio consumer for the real PS4 audio output client shared-memory
 * protocol. Decoded PCM is sent through the host AudioSink callbacks.
 * PS5 AudioOut2 container queues currently use a paced null sink.
 */

#include <cstddef>
#include "base/arch.h"

namespace kern {

// Called from the POSIX-shm syscalls whenever a named region gets or changes
// its host backing (`base` = null means the name went away). Names that are not
// part of the audio output client protocol are ignored; the first control region
// starts the daemon thread.
void AudioDaemonNoticeShm(const char* name, u8* base, size_t size);

// Stop consuming audio regions before the guest releases their mapping.
// Waits for any consumer access in progress to finish.
void AudioDaemonForgetRange(const void* base, size_t size);

}  // namespace kern
