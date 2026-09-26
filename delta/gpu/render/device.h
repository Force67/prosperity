/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Bring-up of the device the renderer runs on (on the backend DELTA_GPU_BACKEND
// names), and the progress markers that locate a device loss.

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

// Create the device, the frame ring and the upload rings.
bool CreateDevice();

void DrawCheckpoint(rhi::CommandList* list, u32 frame, u32 draw, bool after);
void DispatchCheckpoint(rhi::CommandList* list, u64 cs_addr, bool after);

}  // namespace gpu::render
