#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include <cstddef>
#include <cstdint>
#include "base/arch.h"

namespace gpu::ps4 {

void SetPs4NeoMode(bool enabled);

using WriteWatchCallback = void (*)(uintptr_t, size_t, unsigned, bool, bool);
void SetWriteWatchCallback(WriteWatchCallback callback);

void SubmitDcb(const void* dcb, u32 size_bytes);
void SubmitRingDcb(const void* dcb, u32 size_bytes);

void SubmitCcb(const void* ccb, u32 size_bytes);

void SetUnknownWaitHook(void (*hook)(u64 address));

void EndFrame(u64 scanout_base);

void WaitForFrame();

void StopRenderQueue();

}  // namespace gpu::ps4
