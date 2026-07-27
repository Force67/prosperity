/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Fail-fast invariant checks for the gpu module, in the spirit of
// base/check.h's BASE_BUGCHECK -- but active in every build flavour: the
// vendored macro is stripped repo-wide (BASE_STRIP_BUGCHECK) and BASE_DCHECK's
// CHECK_BREAK is a no-op outside CONFIG_DEBUG, so neither ever fires in the
// Release builds this project actually runs. A violated invariant here means
// the renderer's own bookkeeping is corrupt; dying at the violation beats
// debugging the downstream artifact three frames later.
//
// Only for programmer invariants. Anything a guest title can trigger with bad
// packet data must stay an error path, never a check.

#include <cstdio>
#include <cstdlib>

#define GPU_BUGCHECK(expression, ...)                                         \
  do {                                                                        \
    if (!(expression)) {                                                      \
      std::fprintf(stderr, "[gpu] BUGCHECK %s:%d: %s\n", __FILE__, __LINE__,  \
                   #expression);                                              \
      __VA_OPT__(std::fprintf(stderr, "[gpu]   " __VA_ARGS__);                \
                 std::fputc('\n', stderr);)                                   \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
