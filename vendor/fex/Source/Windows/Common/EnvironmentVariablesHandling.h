// SPDX-License-Identifier: MIT
#pragma once
#include <winternl.h>

namespace FEX::Windows {
void SetupEnvironmentVariableValues(HMODULE NtDll);
}
