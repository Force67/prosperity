#pragma once

#include "base/arch.h"

namespace ui {

enum class MemoryOverlayMode { kOff, kCompact, kFull };

MemoryOverlayMode GetMemoryOverlayMode();
void SetMemoryOverlayMode(MemoryOverlayMode mode);
void MemoryOverlayToggle();
bool MemoryOverlayFull();
bool MemoryOverlayBlocksInput();
bool MemoryOverlayEscape(bool pressed, bool gamepad = false);
void MemoryOverlayBuild(u32 width, u32 height);
void MemoryOverlayReset();

}  // namespace ui
