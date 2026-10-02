#pragma once

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "ui/home_screen.h"

namespace cli {

base::Vector<ui::HomeGame> ReadRecentGames();
void RememberGame(const base::String& path,
                  const base::String& name,
                  const base::String& title_id,
                  bool is_ps5,
                  const base::Vector<u8>& icon,
                  const base::Vector<u8>& artwork);

}  // namespace cli
