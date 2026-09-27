/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero system-information syscalls. A PS5 reports different hardware than
 * a PS4 does, so the handlers a title reads its machine's shape from diverge
 * from the Orbis ones rather than sharing them.
 */

#pragma once

#include <cstddef>
#include "base/arch.h"
#include "guest_abi.h"

namespace krnl {

int PS4ABI Ps5CpusetGetaffinity(int level,
                                int which,
                                i64 id,
                                size_t cpusetsize,
                                void* mask);

int PS4ABI Ps5Sysctl(int* name,
                     u32 namelen,
                     void* oldp,
                     size_t* oldlenp,
                     const void* newp,
                     size_t newlen);

}  // namespace krnl
