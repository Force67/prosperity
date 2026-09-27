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

namespace utl {
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

// Arm a write-watch on a guest range from a layer that cannot reach the kernel
// directly. The kernel owns the SIGSEGV machinery (see krnl::startWriteWatch)
// and registers itself here at startup; the GPU layer needs it because the only
// interesting addresses (the descriptor-table pointer a shader actually read)
// are not known until a draw is being processed, and they move every run.
// Returns false when nothing has registered.
using WriteWatchArmer = void (*)(uintptr_t addr,
                                 size_t bytes,
                                 unsigned every_ms);
void SetWriteWatchArmer(WriteWatchArmer fn);
bool ArmWriteWatch(uintptr_t addr, size_t bytes, unsigned every_ms);

// Report the qword at `addr` on every write-watch fault. A page watch reopens
// its page on the first fault so the guest can proceed, so only the FIRST byte
// touched per re-arm is ever seen: a block memcpy hides every later byte,
// including the one you care about. Watching the value instead pins down which
// fault the change happened across, which names the writer.
void SetWriteWatchValueProbe(uintptr_t addr);
uintptr_t WriteWatchValueProbe();

// Follow the probed word UPSTREAM: when a watch fault looks like a block copy
// into the probe, re-aim the probe at the corresponding word in the copy's
// source and watch that instead. Guest allocations move every run, so the
// address of each hop is only knowable from the previous fault, so chaining is
// the only way to walk a value back to where it was first written. Capped so a
// self-referential copy cannot loop forever.
void SetWriteWatchChase(unsigned hops);
unsigned WriteWatchChaseLeft();
void WriteWatchChaseTook();
size_t MappedMemoryPrefix(const void* addr, size_t max_len);

size_t GetAvailableMem();
}  // namespace utl
