/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

bool CreateDevice();

void DrawCheckpoint(rhi::CommandList* list, u32 frame, u32 draw, bool after);
void DispatchCheckpoint(rhi::CommandList* list, u64 cs_addr, bool after);

}  // namespace gpu::render
