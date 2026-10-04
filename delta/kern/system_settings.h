#pragma once

#include <cstddef>
#include "base/arch.h"

namespace kern {
bool GetRegistryInt(u64 key, i32& value);
bool SetRegistryInt(u64 key, i32 value);
void InitializeShellCoreSharedMemory(u8* memory, size_t size);
}  // namespace kern
