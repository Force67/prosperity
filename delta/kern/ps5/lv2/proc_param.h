#pragma once

#include <cstddef>

// PS5 (FreeBSD 11 / Prospero) process-parameter handling. Kept out of the PS4
// tree so the FreeBSD 9 paths stay untouched.
namespace krnl::ps5 {

// The proc param sys_dynlib_get_proc_param hands a PS5 title. Returns `appPP`
// unchanged unless the title needs the sceLibcParam heap override; `appSize` is
// the app's own proc-param size.
void *procParam(void *appPP, size_t appSize);

} // namespace krnl::ps5
