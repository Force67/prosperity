/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

namespace gpu::render {

void CsSyncReport(double frames);
void PushStageSample();
void DrawPerfOverlay(u8* pixels, u32 w, u32 h, bool rgba = false);
void ReportFps();

}  // namespace gpu::render
