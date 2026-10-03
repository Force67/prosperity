#pragma once

#include "base/arch.h"
#include "base/strings/xstring.h"

namespace ui {
void PauseMenuSetGameTitle(const base::String& title);
void PauseMenuGameReady();
void PauseMenuToggle();
bool PauseMenuVisible();
bool PauseMenuExitRequested();
void PauseMenuRequestExit();
bool PauseMenuReturnRequested();
void PauseMenuRequestReturn();
void PauseMenuBuild(u32 width, u32 height);
void PauseMenuReset();
}  // namespace ui
