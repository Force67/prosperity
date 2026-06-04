/*
 * PS4Delta : PS4 emulation and research project
 *
 * Shared graphics-layer state with no platform dependency. Currently just the
 * gameplay harness signal (see gfx.h), defined here so both the GPU renderer
 * (delta_gpu) and the input layer (libScePad, delta_runtime) link one instance.
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
