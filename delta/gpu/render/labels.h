/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

bool WantDebugUtils();

void CmdBeginLabel(rhi::CommandList* list, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
void CmdEndLabel(rhi::CommandList* list);
void CmdInsertLabel(rhi::CommandList* list, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

}  // namespace gpu::render
