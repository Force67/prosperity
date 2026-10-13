#pragma once
#include "base/arch.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
// DELTA_GPU_DIRECT: shaders resolve guest addresses on the GPU. Direct-memory
// pools import through a second view, outside the guest's protection changes.
// Other mappings use page tables, importing writable pages and uploading
// read-only pages once. Remaps invalidate their cached addresses and copies.
using gcn::GuestDirectIndex;
using gcn::kDirectBase;
using gcn::kDirectBlocks;
using gcn::kDirectBlockShift;
using gcn::kDirectMissSlots;
using gcn::kDirectSize;

// Sets up the table and publishes it to the recompiler; false when the
// option is off or the device cannot import host memory.
bool InitGuestDirect();
// Before a submit: imports what the GPU reported missing, and follows remaps.
void SyncGuestDirect();
// Any thread: a remap or protection change can invalidate copied host pages.
void NoteGuestDirectRemap(u64 base, u64 bytes);
// Inside the second mapping of direct memory. Only the GPU reads through it,
// so it aliases nothing the write tracker has to worry about.
bool InDirectAlias(u64 va);
// The 64 KiB block at `block` reads from `device_address` instead, until
// restored. Restoring is async-signal-safe.
void DirectRedirect(u64 block, u64 device_address);
void DirectRestore(u64 block);
// A GPU buffer holding the guest byte at `address`: direct pools use their
// second mapping, other writable pages use an import, and read-only pages a
// cached upload. `offset` locates the byte and `bytes` its contiguous span.
rhi::Buffer* DirectImportAt(u64 address, u64* offset, u64* bytes);
// One imported buffer around the guest bytes at `address` for a vertex or
// index bind, so the fetcher reads them where they lie: `offset` is the
// span's place in it. The whole span has to stay inside one view. Null
// where it does not or the import failed; the caller copies then.
rhi::Buffer* DirectBindAt(u64 address, u64 bytes, u64* offset);
}  // namespace gpu::render
