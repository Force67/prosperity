#pragma once
#include "base/arch.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
using gcn::GuestDirectIndex;
using gcn::kDirectBase;
using gcn::kDirectBlocks;
using gcn::kDirectBlockShift;
using gcn::kDirectMissSlots;
using gcn::kDirectSize;

bool InitGuestDirect();
void SyncGuestDirect();
void NoteGuestDirectRemap(u64 base, u64 bytes);
bool InDirectAlias(u64 va);
void DirectRedirect(u64 block, u64 device_address);
void DirectRestore(u64 block);
rhi::Buffer* DirectImportAt(u64 address, u64* offset, u64* bytes);
rhi::Buffer* DirectBindAt(u64 address, u64 bytes, u64* offset);
}  // namespace gpu::render
