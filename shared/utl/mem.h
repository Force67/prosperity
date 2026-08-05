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
#include <type_traits>

namespace utl {
enum class allocationType { reserve, commit, reservecommit };

enum class pageProtection : uint32_t {
  priv = 0,
  r = 1,
  w = 1 << 1,
  x = 1 << 2,
  rx = r | x,
  rwx = r | w | x,
};

inline bool operator&(pageProtection lhs, pageProtection rhs) {
  return (static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

inline pageProtection operator|(pageProtection lhs, pageProtection rhs) {
  return static_cast<pageProtection>(static_cast<uint32_t>(lhs) |
                                     static_cast<uint32_t>(rhs));
}

inline pageProtection &operator|=(pageProtection &lhs, pageProtection rhs) {
  lhs = lhs | rhs;
  return lhs;
}

void *allocMem(void *preferredAddr, size_t len, pageProtection, allocationType);
void freeMem(void *addr);
bool protectMem(void *addr, size_t len, pageProtection);
bool isMemoryRangeMapped(const void *addr, size_t len);

// Arm a write-watch on a guest range from a layer that cannot reach the kernel
// directly. The kernel owns the SIGSEGV machinery (see krnl::startWriteWatch)
// and registers itself here at startup; the GPU layer needs it because the only
// interesting addresses -- the descriptor-table pointer a shader actually read
// -- are not known until a draw is being processed, and they move every run.
// Returns false when nothing has registered.
using WriteWatchArmer = void (*)(uintptr_t addr, size_t bytes, unsigned everyMs);
void setWriteWatchArmer(WriteWatchArmer fn);
bool armWriteWatch(uintptr_t addr, size_t bytes, unsigned everyMs);

// Report the qword at `addr` on every write-watch fault. A page watch reopens
// its page on the first fault so the guest can proceed, so only the FIRST byte
// touched per re-arm is ever seen -- a block memcpy hides every later byte,
// including the one you care about. Watching the value instead pins down which
// fault the change happened across, which names the writer.
void setWriteWatchValueProbe(uintptr_t addr);
uintptr_t writeWatchValueProbe();

// Follow the probed word UPSTREAM: when a watch fault looks like a block copy
// into the probe, re-aim the probe at the corresponding word in the copy's
// source and watch that instead. Guest allocations move every run, so the
// address of each hop is only knowable from the previous fault -- chaining is
// the only way to walk a value back to where it was first written. Capped so a
// self-referential copy cannot loop forever.
void setWriteWatchChase(unsigned hops);
unsigned writeWatchChaseLeft();
void writeWatchChaseTook();
size_t mappedMemoryPrefix(const void *addr, size_t maxLen);

size_t getAvailableMem();
}
