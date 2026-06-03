#pragma once

// Copyright (C) Force67 2019

#include <base.h>

namespace krnl {
int PS4ABI sys_netcontrol(uint32_t fd, uint32_t op, void *buffer,
                          uint32_t size);
int PS4ABI sys_socketex(const char *name, int32_t domain, int32_t type,
                        int32_t protocol);
int PS4ABI sys_socket(int32_t domain, int32_t type, int32_t protocol);
int PS4ABI sys_connect(int32_t fd, const void *addr, uint32_t addrlen);
int PS4ABI sys_recvmsg(int32_t fd, void *msg, int32_t flags);

} // namespace krnl