#pragma once

#include "ui/home_screen.h"

namespace cli {

base::String AddHomeGame(const base::String& path);
base::String ShowHomeScreen(const base::Vector<ui::HomeGame>& games);

}  // namespace cli
