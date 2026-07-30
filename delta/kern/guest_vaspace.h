#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * See guest_vaspace.cpp. Claims the fixed virtual-address ranges a PS4 title
 * and its firmware modules MAP_FIXED into, before anything host-side can be
 * handed them.
 */

#include <cstddef>
#include <cstdint>

namespace krnl {

// Reserve every known guest-fixed range PROT_NONE. Call once, as early as
// possible -- before the CPU backend reserves its JIT heap and before any guest
// module maps. Safe to call twice (the second call is a no-op).
void reserveGuestVaSpace();

// True when [addr, addr+len) lies wholly inside a range reserved above. A guest
// mmap whose hint lands here must be COMMITTED there rather than relocated: the
// range is guest-owned, and our placeholder is the only thing occupying it.
bool isGuestReservedVa(const void *addr, size_t len);

}  // namespace krnl
