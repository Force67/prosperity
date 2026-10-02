#pragma once
#include "base/arch.h"

namespace gpu::render {
// DELTA_GPU_DIRECT: shaders read the title's direct memory where it lives.
// The memfd that backs it is mapped once more, out of the guest's reach, and
// imported 2 MiB at a time as the GPU first touches it; a table of 64 KiB
// guest blocks gives each block's device address. Nothing is copied, and the
// CPU never resolves what a shader reads.
constexpr u64 kDirectBase = 0x300000000ull;
constexpr u64 kDirectSize = 0x1000000000ull;
constexpr u32 kDirectBlockShift = 16;
constexpr u32 kDirectMissSlots = 63;

// Sets up the table and publishes it to the recompiler; false when the
// option is off or the device cannot import host memory.
bool InitGuestDirect();
// Before a submit: imports what the GPU reported missing, and follows remaps.
void SyncGuestDirect();
// Inside the second mapping of direct memory. Only the GPU reads through it,
// so it aliases nothing the write tracker has to worry about.
bool InDirectAlias(u64 va);
// The 64 KiB block at `block` reads from `device_address` instead, until
// restored. Restoring is async-signal-safe.
void DirectRedirect(u64 block, u64 device_address);
void DirectRestore(u64 block);
}  // namespace gpu::render
