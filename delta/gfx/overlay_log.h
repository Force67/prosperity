#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The live log panel: the tail of the log drawn over the game, bottom right.
 *
 * A log sink keeps the last few lines in a fixed ring of fixed-length slots, so
 * a title logging thousands of lines a second costs one bounded memcpy per line
 * and nothing per frame. The panel goes straight into the overlay's foreground
 * draw list (one rect plus one AddText per line, clipped rather than measured),
 * so its cost does not depend on how much is being logged.
 *
 * It shows what goes through the shared logger, which is LOG_* plus the base
 * channels routed into it. Most of the emulator's console output is a direct
 * fprintf to stderr and does not pass through here, so the panel is a subset of
 * the terminal, not a mirror of it.
 */

#include "base/arch.h"

namespace gfx {

// Start capturing log lines. Idempotent; overlayEnsureImGui does it.
void overlayLogAttach();

// Draw the panel over a display of `w` by `h`. No-op while hidden or empty.
void overlayLogBuild(u32 w, u32 h);

// Bound to F2 by the window event pump. Visible by default.
void overlayLogToggle();

}  // namespace gfx
