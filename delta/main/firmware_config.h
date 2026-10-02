#pragma once

#include "main/command_line.h"

namespace cli {

bool FirmwareModulesReady(bool ps5);
u32 Ps5FirmwareVersion();

bool ConfigureFirmware(const CommandLine& command);

}  // namespace cli
