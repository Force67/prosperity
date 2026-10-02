#pragma once

#include "main/command_line.h"

namespace cli {

bool FirmwareModulesReady(bool ps5);

bool ConfigureFirmware(const CommandLine& command);

}  // namespace cli
