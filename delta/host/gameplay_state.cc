/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Whether the title has reached gameplay, as opposed to a menu or a cutscene.
 * The renderer sets it and the pad HLE reads it, so it is defined here, in
 * the module both already link, rather than owned by either of them.
 */

#include "base/atomic.h"
#include "host/window.h"

namespace host {
namespace {
base::Atomic<bool> g_in_gameplay{false};
}

void SetInGameplay(bool v) {
  g_in_gameplay.store(v, base::memory_order_relaxed);
}
bool InGameplay() {
  return g_in_gameplay.load(base::memory_order_relaxed);
}

}  // namespace host
