#pragma once

#include "base/arch.h"
#include "guest/controller_input.h"

namespace kern {
struct ControllerPort {
  u32 user, type, index, flags, pid;
};

i32 AllocateControllerHandle();
bool IsControllerHandle(u32 handle);
bool ReleaseControllerHandle(u32 handle);
i32 BindControllerPort(u32 handle, const ControllerPort& port);
i32 UnbindControllerPort(u32 handle, u32 pid);
bool GetControllerPort(u32 handle, ControllerPort& port);
void SetControllerSource(bool (*poll)(guest::ControllerState&));
void SetControllerOpen(void (*open)(int user, int type, int index));
bool ReadControllerPort(u32 handle, guest::ControllerState& state);
bool ControllerPortConnected(u32 handle);
bool HasControllerSource();
void SetControllerRumble(void (*rumble)(u8 large, u8 small));
i32 VibrateControllerPort(u32 handle, u8 large, u8 small);
void SetControllerLight(void (*light)(u8 red, u8 green, u8 blue));
i32 LightControllerPort(u32 handle, u8 red, u8 green, u8 blue);
i32 ResetControllerLight(u32 handle);
}  // namespace kern
