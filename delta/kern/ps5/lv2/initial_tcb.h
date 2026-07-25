#pragma once

#include <cstdint>

// PS5 (FreeBSD 11 / Prospero) initial-thread setup. Kept out of the PS4 tree so
// the FreeBSD 9 paths stay untouched.
namespace krnl::ps5 {

// Build the TCB a PS5 process's initial thread starts life with, and return the
// value its fs base should hold. 0 if it could not be allocated.
uint64_t makeInitialTcb();

} // namespace krnl::ps5
