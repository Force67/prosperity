/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE aliases for libSceAgcDriver exports that firmware 01.14.00 does not
 * have (newer-SDK titles import them anyway; see vprx/ps5/libkernel_ps5.cpp for the
 * same problem in libc). Both entries here are query functions with a u32
 * out-parameter, reached from the title's varargs GPU-debug-label path. Left on the
 * badcall stub they return "success" without writing the out-param, and the caller
 * allocas that uninitialised stack slot -- Dead Cells (PPSA15552) overflows its
 * stack into a guard page that way.
 *
 * Everything else in libSceAgcDriver stays LLE.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>

namespace {
// libSceAgc "unsupported" status. The title's own SDK wrappers already return this
// code, so its error paths handle it; anything non-zero except 0x8a6c9018 makes the
// debug-label helper bail before the alloca.
constexpr int kAgcUnsupported = 0x8a6c0044;

// f(uint32_t *out): caller reads *out unconditionally and ignores the return.
int PS4ABI agcDriverQueryU32(uint32_t *out) {
  if (out)
    *out = 0;
  return 0;
}

// f(uint32_t *sizeOut): on success the caller allocas *sizeOut. We have no size to
// report, so fail instead -- the caller then returns -1 without touching the stack.
int PS4ABI agcDriverQuerySize(uint32_t *sizeOut) {
  if (sizeOut)
    *sizeOut = 0;
  return kAgcUnsupported;
}

// f(int): return value ignored at its only call site.
int PS4ABI agcDriverSetFlag(int) { return 0; }
}  // namespace

static const runtime::funcInfo functions[] = {
    {0x174657B79AB46530, (void *)&agcDriverQueryU32},   // F0ZXt5q0ZTA
    {0xB89CE246C3839357, (void *)&agcDriverQuerySize},  // uJziRsODk1c
    {0x53DB9EC84852905E, (void *)&agcDriverSetFlag},    // U9ueyEhSkF4
};

MODULE_INIT_PS5(libSceAgcDriver);

extern "C" int vprx_anchor_ps5_libSceAgcDriver = 1;
