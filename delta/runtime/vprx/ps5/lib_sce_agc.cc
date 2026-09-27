/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE aliases for libSceAgc exports missing from firmware 01.14.00.
 * Each is reached through one of the title's own SDK wrappers, which already
 * handle an "unsupported" status (they return 0x8a6c0044 / 0x8a6c000a
 * themselves on their guard paths), so reporting failure is the safe answer:
 * the caller skips the feature instead of consuming out-parameters the badcall
 * stub would have left uninitialised.
 *
 * Everything else in libSceAgc stays LLE.
 */

#include "guest_abi.h"
#include "runtime/vprx/vprx.h"  // PS4ABI (via <guest_abi.h>), MODULE_INIT_PS5

namespace {
constexpr int kAgcUnsupported = 0x8a6c0044;

int PS4ABI AgcUnsupported() {
  return kAgcUnsupported;
}

// f(int, int): the wrapper discards the result (`xor eax,eax; ret`).
int PS4ABI AgcIgnored(int, int) {
  return 0;
}
}  // namespace

static const runtime::vprx::ExportEntry kExports[] = {
    {0xFCA47359E915D76D, (void*)&AgcUnsupported},  // -KRzWekV120
    {0x4FAC6E570D0A509A, (void*)&AgcIgnored},      // T6xuVw0KUJo
    {0x000797FD4E7F3F73, (void*)&AgcUnsupported},  // AAeX-U5-P3M
    {0x7DDE41A79B464E0A, (void*)&AgcUnsupported},  // fd5Bp5tGTgo
};

MODULE_INIT_PS5(libSceAgc);

extern "C" int g_vprx_anchor_ps5_lib_sce_agc = 1;
