#pragma once

#include <cstddef>
#include <cstdint>

// Lets a layer that cannot reach the kernel arm a write-watch on a guest range.
// The kernel owns the SIGSEGV machinery (see krnl::StartWriteWatch) and
// registers itself here at startup; the GPU layer needs it because the only
// interesting addresses (the descriptor-table pointer a shader actually read)
// are not known until a draw is being processed, and they move every run.
namespace write_watch {

using Armer = void (*)(uintptr_t addr, size_t bytes, unsigned every_ms);
void SetArmer(Armer fn);
// Returns false when nothing has registered.
bool Arm(uintptr_t addr, size_t bytes, unsigned every_ms);

// Report the qword at `addr` on every write-watch fault. A page watch reopens
// its page on the first fault so the guest can proceed, so only the FIRST byte
// touched per re-arm is ever seen: a block memcpy hides every later byte,
// including the one you care about. Watching the value instead pins down which
// fault the change happened across, which names the writer.
void SetValueProbe(uintptr_t addr);
uintptr_t ValueProbe();

// Follow the probed word UPSTREAM: when a watch fault looks like a block copy
// into the probe, re-aim the probe at the corresponding word in the copy's
// source and watch that instead. Guest allocations move every run, so the
// address of each hop is only knowable from the previous fault, so chaining is
// the only way to walk a value back to where it was first written. Capped so a
// self-referential copy cannot loop forever.
void SetChase(unsigned hops);
unsigned ChaseLeft();
void ChaseTook();

}  // namespace write_watch
