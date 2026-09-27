#pragma once

// Copyright (C) Force67 2019

#include <guest_abi.h>
#include "base/arch.h"

namespace krnl {
int PS4ABI sys_ioctl(u32 fd, u32 cmd, void *data);
}