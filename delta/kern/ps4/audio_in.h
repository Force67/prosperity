#pragma once

#include <cstddef>
#include "base/arch.h"

namespace kern::ps4 {
void AudioInNoticeShm(const char* name, u8* base, size_t size);
void AudioInForgetRange(const void* base, size_t size);
i32 SetAudioInPort(u32 pid, u32 handle, bool open);
}  // namespace kern::ps4
