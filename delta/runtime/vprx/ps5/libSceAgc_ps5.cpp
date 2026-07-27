/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE aliases for libSceAgc exports missing from firmware 01.14.00.
 * Each is reached through one of the title's own SDK wrappers, which already
 * handle an "unsupported" status (they return 0x8a6c0044 / 0x8a6c000a themselves
 * on their guard paths), so reporting failure is the safe answer: the caller
 * skips the feature instead of consuming out-parameters the badcall stub would
 * have left uninitialised.
 *
 * Everything else in libSceAgc stays LLE.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>

namespace {
constexpr int kAgcUnsupported = 0x8a6c0044;

int PS4ABI agcUnsupported() { return kAgcUnsupported; }

// f(int, int): the wrapper discards the result (`xor eax,eax; ret`).
int PS4ABI agcIgnored(int, int) { return 0; }
}  // namespace

static const runtime::funcInfo functions[] = {
    {0xFCA47359E915D76D, (void *)&agcUnsupported},  // -KRzWekV120
    {0x4FAC6E570D0A509A, (void *)&agcIgnored},      // T6xuVw0KUJo
    {0x000797FD4E7F3F73, (void *)&agcUnsupported},  // AAeX-U5-P3M
    {0x7DDE41A79B464E0A, (void *)&agcUnsupported},  // fd5Bp5tGTgo
};

MODULE_INIT_PS5(libSceAgc);

extern "C" int vprx_anchor_ps5_libSceAgc = 1;
