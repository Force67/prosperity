/*
 * PS4Delta : PS4 emulation and research project
 *
 * Orbis syscall handlers with no shared counterpart. Each follows the PS4
 * kernel's sysent; lv2/ps4/table.cc routes to them.
 */

#pragma once

#include <cstddef>
#include "base/arch.h"
#include "guest_abi.h"

namespace kern {

// Answers shared by many rows.
int PS4ABI Ps4Ok();
int PS4ABI Ps4Nosys();
int PS4ABI Ps4Eperm();
int PS4ABI Ps4Einval();
int PS4ABI Ps4Eopnotsupp();

// sockets
int PS4ABI Ps4Accept(int fd, void* addr, u32* addrlen);
int PS4ABI Ps4Netabort(int fd, int flags);

// memory
int PS4ABI Ps4Mtypeprotect(u8* addr, size_t len, int mtype, int prot);

}  // namespace kern
