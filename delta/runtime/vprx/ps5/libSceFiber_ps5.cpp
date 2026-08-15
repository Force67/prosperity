/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only no-op shims for the sanitizer fiber hooks libSceFiber imports from
 * libSceDbgThreadSanitizer / libSceDbgAddressSanitizer. Retail firmware ships
 * neither module, so those seven imports land on the badcall stub and every
 * boot logs them as unresolved (Demon's Souls PPSA01342 was the first hit).
 *
 * The real libSceFiber only calls them when sceKernelIsThreadSanitizerEnabled /
 * sceKernelIsAddressSanitizerEnabled report a sanitizer runtime, which never
 * happens in our env, but resolving them here silences the warnings and makes a
 * stray call harmless instead of fatal.
 *
 * The fiber API itself stays LLE: fw 08.40 libSceFiber exports all of it, and
 * its switch primitive is a plain user-space callee-saved register swap that
 * only reads the fs-based TLS fiber slot (never writes fs), so it runs as-is.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5
#include "base/arch.h"


namespace {
void *PS4ABI tsanCreateFiber(u32) { return nullptr; }
void PS4ABI tsanDestroyFiber(void *) {}
void *PS4ABI tsanGetCurrentFiber() { return nullptr; }
void PS4ABI tsanSwitchToFiber(void *, u32) {}

// The caller passes &save on the stack and later forwards the stored value to
// the finish hook; write null so it never forwards stack garbage.
void PS4ABI sanitizerStartSwitchFiber(void **save, const void *, size_t) {
  if (save)
    *save = nullptr;
}

void PS4ABI sanitizerFinishSwitchFiber(void *, const void **bottomOld,
                                       size_t *sizeOld) {
  if (bottomOld)
    *bottomOld = nullptr;
  if (sizeOld)
    *sizeOld = 0;
}

void PS4ABI asanDestroyFakeStack(void *) {}
}  // namespace

static const runtime::funcInfo functions[] = {
    {0x52428EF1EB1400CE, (void *)&tsanCreateFiber},           // UkKO8esUAM4 __tsan_create_fiber
    {0xA8F98DBB1D62524A, (void *)&tsanDestroyFiber},          // qPmNux1iUko __tsan_destroy_fiber
    {0xB5D9DC2DEBB6D28C, (void *)&tsanGetCurrentFiber},       // tdncLeu20ow __tsan_get_current_fiber
    {0x3394726352F54432, (void *)&tsanSwitchToFiber},         // M5RyY1L1RDI __tsan_switch_to_fiber
    {0x00CE5D6D7A77A9B2, (void *)&sanitizerStartSwitchFiber}, // AM5dbXp3qbI __sanitizer_start_switch_fiber
    {0x718958B03418E74D, (void *)&sanitizerFinishSwitchFiber},// cYlYsDQY500 __sanitizer_finish_switch_fiber
    {0x8DA721ADB8E91E73, (void *)&asanDestroyFakeStack},      // jachrbjpHnM __asan_destroy_fake_stack
};

MODULE_INIT_PS5(libSceFiber);

extern "C" int vprx_anchor_ps5_libSceFiber = 1;
