#pragma once

#include <cstdint>

namespace krnl {
class proc;
}

namespace krnl::ps5 {

// DIAGNOSTIC ONLY -- not a fix, and off unless DELTA_PS5_CTOR_PREPEND is set.
//
// Prepend a function in the main module to its legacy .ctors list, so it runs
// before every static initializer. Used to answer "if subsystem X were already
// constructed, how much further would this title boot?" without guessing at the
// real trigger. DELTA_PS5_CTOR_PREPEND=<hex offset into the main module>.
void maybePrependCtor(proc &p);

} // namespace krnl::ps5
