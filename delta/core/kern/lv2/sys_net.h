#pragma once

// Copyright (C) Force67 2019

#include <base.h>

namespace krnl {
int PS4ABI sys_netcontrol(uint32_t fd, uint32_t op, void *buffer,
                          uint32_t size);
int PS4ABI sys_socketex(const char *name, int32_t domain, int32_t type,
                        int32_t protocol);

} // namespace krnl