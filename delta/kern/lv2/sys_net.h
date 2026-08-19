#pragma once

// Copyright (C) Force67 2019

#include <base.h>
#include "base/arch.h"

namespace krnl {
int PS4ABI sys_netcontrol(u32 fd, u32 op, void *buffer,
                          u32 size);
int PS4ABI sys_socketex(const char *name, i32 domain, i32 type,
                        i32 protocol);
int PS4ABI sys_socket(i32 domain, i32 type, i32 protocol);
int PS4ABI sys_netgetsockinfo(i32 fd, void *info, i32 n, i32 flags);
int PS4ABI sys_bind(i32 fd, const void *addr, u32 addrlen);
int PS4ABI sys_getsockname(i32 fd, void *addr, u32 *addrlen);
int PS4ABI sys_socketclose(i32 fd);
int PS4ABI sys_connect(i32 fd, const void *addr, u32 addrlen);
int PS4ABI sys_recvmsg(i32 fd, void *msg, i32 flags);
i64 PS4ABI sys_sendto(i32 fd, const void *buf, size_t len,
                          i32 flags, const void *to, u32 tolen);
i64 PS4ABI sys_recvfrom(i32 fd, void *buf, size_t len, i32 flags,
                            void *from, u32 *fromlen);

} // namespace krnl