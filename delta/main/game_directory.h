#pragma once

#include "base/strings/xstring.h"

namespace cli {
struct GameDirectory {
  base::String root;
  base::String main_module;
};

GameDirectory FindGameDirectory(const base::String& path);
}  // namespace cli
