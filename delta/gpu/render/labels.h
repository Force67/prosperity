/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Command labels for capture tools and the validation layer, keyed by GUEST
// addresses so a capture lines up against DELTA_GPU_* logs by eye. Nothing is
// formatted unless the device reports that labels reach a tool.

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

// Whether to ask the device for labels at all: with no consumer every label
// is a formatted string for nobody (~0.3 ms per Isaac frame). True only with
// RenderDoc injected, the validation layer on, or DELTA_GPU_MARKERS=1.
bool WantDebugUtils();

void CmdBeginLabel(rhi::CommandList* list, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
void CmdEndLabel(rhi::CommandList* list);
void CmdInsertLabel(rhi::CommandList* list, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

}  // namespace gpu::render
