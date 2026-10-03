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
void FreeMem(void* addr, size_t size);
void BeginMemorySession();
size_t EndMemorySession();
size_t SessionMappedBytes();
bool ProtectMem(void* addr, size_t len, PageProtection);
bool IsMemoryRangeMapped(const void* addr, size_t len);

// Identity of backing pages allocated through the guest memory helpers. A
// remap gets a new identity even when its address, size and permissions match.
// Unknown or only partially tracked ranges return zero and cannot be cached.
void TrackMemoryMapping(void* addr, size_t len, bool reserved = false);
void ForgetMemoryMapping(void* addr, size_t len);
u64 MemoryMappingIdentity(const void* addr, size_t len);

size_t MappedMemoryPrefix(const void* addr, size_t max_len);

struct MappingStats {
  u64 mapped = 0;
  u64 reserved = 0;
};
MappingStats GetMappingStats();

size_t GetAvailableMem();

// Pages a cache write-protects to learn when the guest writes them (the GPU's
// write tracker). The kernel's fault handler offers every access fault here
// first; a true return means the page was one of those, is writable again and
// the faulting instruction can resume.
using WriteFaultHandler = bool (*)(uintptr_t addr);
void SetWriteFaultHandler(WriteFaultHandler handler);
bool HandleWriteFault(uintptr_t addr);

// The host kernel does not fault on a protected page, it fails the call
// (read() returns EFAULT). Code that lets the kernel write guest memory calls
// this first; the tracker opens and reports the pages in the range.
using HostWriteHook = void (*)(void* addr, size_t len);
void SetHostWriteHook(HostWriteHook hook);
void BeforeHostWrite(void* addr, size_t len);
}  // namespace host_memory
