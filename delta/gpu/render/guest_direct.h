#pragma once
#include "base/arch.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
// DELTA_GPU_DIRECT: shaders read the title's direct memory where it lives.
// The memfd that backs it is mapped once more, out of the guest's reach, and
// imported 2 MiB at a time as the GPU first touches it; a table of 64 KiB
// guest blocks gives each block's device address. Nothing is copied, and the
// CPU never resolves what a shader reads.
using gcn::GuestDirectIndex;
using gcn::kDirectBase;
using gcn::kDirectBlocks;
using gcn::kDirectBlockShift;
using gcn::kDirectMissSlots;
using gcn::kDirectPs5Base;
using gcn::kDirectSize;

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
// The import of direct memory holding the guest byte at `address`, through
// the second mapping and so whatever the guest's protection: `offset` is the
// byte's place in it and `bytes` how many follow contiguously. Null where no
// view of direct memory holds the address.
rhi::Buffer* DirectImportAt(u64 address, u64* offset, u64* bytes);
// One imported buffer around the guest bytes at `address` for a vertex or
// index bind, so the fetcher reads them where they lie: `offset` is the
// span's place in it. The whole span has to stay inside one view. Null
// where it does not or the import failed; the caller copies then.
rhi::Buffer* DirectBindAt(u64 address, u64 bytes, u64* offset);
}  // namespace gpu::render
