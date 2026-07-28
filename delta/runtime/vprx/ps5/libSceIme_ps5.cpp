/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE override for sceImeKeyboardOpen. The real .sprx needs the IME
 * service daemon we don't host and fails with 0x80bc0001, which titles don't
 * expect from this call: Minecraft (PPSA17221) asserts on anything but success
 * or 0x80bc0003 / 0x80bc0004, then aborts from its own assert handler.
 *
 * 0x80bc0004 is a status the real sceImeKeyboardOpen returns on its own device
 * paths, and the call site treats it as "no keyboard attached" -- it leaves the
 * keyboard-open flag clear and carries on. Report that rather than faking a
 * keyboard the title would then poll.
 *
 * Everything else in libSceIme stays LLE.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>

namespace {
constexpr int kImeNoKeyboard = 0x80bc0004;

int PS4ABI imeKeyboardOpen(int32_t, const void *) { return kImeNoKeyboard; }
}  // namespace

static const runtime::funcInfo functions[] = {
    {0x79A1578DF26FDF1B, (void *)&imeKeyboardOpen},  // eaFXjfJv3xs
};

MODULE_INIT_PS5(libSceIme);

extern "C" int vprx_anchor_ps5_libSceIme = 1;
