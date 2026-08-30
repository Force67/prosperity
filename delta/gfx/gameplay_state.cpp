/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Whether the title has reached gameplay, as opposed to a menu or a cutscene.
 * The renderer sets it and the pad HLE reads it, so it is defined here -- in
 * the module both already link -- rather than owned by either of them.
 */

#include "gfx.h"

#include <atomic>

namespace gfx {
namespace {
std::atomic<bool> g_inGameplay{false};
}

void setInGameplay(bool v) { g_inGameplay.store(v, std::memory_order_relaxed); }
bool inGameplay() { return g_inGameplay.load(std::memory_order_relaxed); }

}  // namespace gfx
