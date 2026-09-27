#include "write_watch/write_watch.h"

namespace write_watch {
namespace {
Armer g_armer = nullptr;
uintptr_t g_probe = 0;
unsigned g_chase = 0;
}  // namespace

void SetArmer(Armer fn) {
  g_armer = fn;
}

bool Arm(uintptr_t addr, size_t bytes, unsigned every_ms) {
  if (!g_armer || !addr || !bytes)
    return false;
  g_armer(addr, bytes, every_ms);
  return true;
}

void SetValueProbe(uintptr_t addr) {
  g_probe = addr;
}

uintptr_t ValueProbe() {
  return g_probe;
}

void SetChase(unsigned hops) {
  g_chase = hops;
}

unsigned ChaseLeft() {
  return g_chase;
}

void ChaseTook() {
  if (g_chase)
    g_chase--;
}

}  // namespace write_watch
