#pragma once

#include "guest/controller_input.h"

namespace host {
void OpenControllerInput(int user, int type, int index);
bool PollControllerInput(guest::ControllerState& state);
}  // namespace host
