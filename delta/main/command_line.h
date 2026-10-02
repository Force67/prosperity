#pragma once

#include "base/containers/vector.h"
#include "base/strings/xstring.h"

namespace cli {

struct CommandLine {
  base::String game;
  base::String configure_ps4_fw;
  base::String configure_ps5_fw;
  base::Vector<base::String> guest_args;
  base::Vector<base::String> options;
  bool dump_options = false;
  bool exit = false;
  int exit_code = 0;
};

CommandLine Parse(int argc, char** argv);

}  // namespace cli
