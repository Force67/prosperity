#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * End-of-pipe fence writes (EOP, RELEASE_MEM, EOS, query dumps) land once the
 * GPU has run the work recorded ahead of them, not when the renderer records
 * it. Shaders read guest memory where it lies, so a title that sees its fence
 * early reuses that memory under a draw the GPU has not run yet.
 */

#include "base/arch.h"
#include "base/functional/function.h"

namespace gpu::ps4 {

// Renderer thread: runs `write` once the GPU is past everything recorded so
// far. `address`/`bytes` name the word it writes and `value` its value, for a
// WAIT_REG_MEM that polls it meanwhile; `bytes` 0 when there is no such word.
void WriteAfterGpu(base::Function<void()> write,
                   u64 address = 0,
                   u32 bytes = 0,
                   u64 value = 0);
// Renderer thread, at the end of a command buffer: submits the work the held
// writes wait for.
void SubmitHeldWrites();
// Any thread: the value a held write puts at `address`, which the GPU sees in
// queue order before the guest does.
bool HeldWriteValue(u64 address, u32* value);
// Whether a held write covers [address, address+bytes).
bool HeldWriteOverlaps(u64 address, u64 bytes);

}  // namespace gpu::ps4
