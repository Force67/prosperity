#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstddef>
#include <cstdint>
#include "base/arch.h"

namespace host_memory {
enum class AllocationType { kReserve, kCommit, kReservecommit };

enum class PageProtection : u32 {
  kPriv = 0,
  kR = 1,
  kW = 1 << 1,
  kX = 1 << 2,
  kRx = kR | kX,
  kRwx = kR | kW | kX,
};

inline bool operator&(PageProtection lhs, PageProtection rhs) {
  return (static_cast<u32>(lhs) & static_cast<u32>(rhs));
}

inline PageProtection operator|(PageProtection lhs, PageProtection rhs) {
  return static_cast<PageProtection>(static_cast<u32>(lhs) |
                                     static_cast<u32>(rhs));
}

inline PageProtection& operator|=(PageProtection& lhs, PageProtection rhs) {
  lhs = lhs | rhs;
  return lhs;
}

void* AllocMem(void* preferred_addr,
               size_t len,
               PageProtection,
               AllocationType);
void FreeMem(void* addr);
bool ProtectMem(void* addr, size_t len, PageProtection);
bool IsMemoryRangeMapped(const void* addr, size_t len);

// Identity of backing pages allocated through the guest memory helpers. A
// remap gets a new identity even when its address, size and permissions match.
// Unknown or only partially tracked ranges return zero and cannot be cached.
void TrackMemoryMapping(void* addr, size_t len);
void ForgetMemoryMapping(void* addr, size_t len);
u64 MemoryMappingIdentity(const void* addr, size_t len);

size_t MappedMemoryPrefix(const void* addr, size_t max_len);

size_t GetAvailableMem();
}  // namespace host_memory
