#pragma once

// Copyright (C) Force67 2019

#include <base.h>

namespace krnl {
int PS4ABI sys_netcontrol(uint32_t fd, uint32_t op, void *buffer,
                          uint32_t size);
int PS4ABI sys_socketex(const char *name, int32_t domain, int32_t type,
                        int32_t protocol);
int PS4ABI sys_socket(int32_t domain, int32_t type, int32_t protocol);
int PS4ABI sys_bind(int32_t fd, const void *addr, uint32_t addrlen);
int PS4ABI sys_getsockname(int32_t fd, void *addr, uint32_t *addrlen);
int PS4ABI sys_socketclose(int32_t fd);
int PS4ABI sys_connect(int32_t fd, const void *addr, uint32_t addrlen);
int PS4ABI sys_recvmsg(int32_t fd, void *msg, int32_t flags);
int64_t PS4ABI sys_sendto(int32_t fd, const void *buf, size_t len,
                          int32_t flags, const void *to, uint32_t tolen);
int64_t PS4ABI sys_recvfrom(int32_t fd, void *buf, size_t len, int32_t flags,
                            void *from, uint32_t *fromlen);

} // namespace krnl