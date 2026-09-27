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

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"  // PS4ABI (via <guest_abi.h>), MODULE_INIT_PS5

namespace {
void* PS4ABI TsanCreateFiber(u32) {
  return nullptr;
}
void PS4ABI TsanDestroyFiber(void*) {}
void* PS4ABI TsanGetCurrentFiber() {
  return nullptr;
}
void PS4ABI TsanSwitchToFiber(void*, u32) {}

// The caller passes &save on the stack and later forwards the stored value to
// the finish hook; write null so it never forwards stack garbage.
void PS4ABI SanitizerStartSwitchFiber(void** save, const void*, size_t) {
  if (save)
    *save = nullptr;
}

void PS4ABI SanitizerFinishSwitchFiber(void*,
                                       const void** bottom_old,
                                       size_t* size_old) {
  if (bottom_old)
    *bottom_old = nullptr;
  if (size_old)
    *size_old = 0;
}

void PS4ABI AsanDestroyFakeStack(void*) {}
}  // namespace

static const runtime::vprx::ExportEntry kExports[] = {
    {0x52428EF1EB1400CE,
     (void*)&TsanCreateFiber},  // UkKO8esUAM4 __tsan_create_fiber
    {0xA8F98DBB1D62524A,
     (void*)&TsanDestroyFiber},  // qPmNux1iUko __tsan_destroy_fiber
    {0xB5D9DC2DEBB6D28C,
     (void*)&TsanGetCurrentFiber},  // tdncLeu20ow __tsan_get_current_fiber
    {0x3394726352F54432,
     (void*)&TsanSwitchToFiber},  // M5RyY1L1RDI __tsan_switch_to_fiber
    {0x00CE5D6D7A77A9B2,
     (void*)&SanitizerStartSwitchFiber},  // AM5dbXp3qbI
                                          // __sanitizer_start_switch_fiber
    {0x718958B03418E74D,
     (void*)&SanitizerFinishSwitchFiber},  // cYlYsDQY500
                                           // __sanitizer_finish_switch_fiber
    {0x8DA721ADB8E91E73,
     (void*)&AsanDestroyFakeStack},  // jachrbjpHnM __asan_destroy_fake_stack
};

MODULE_INIT_PS5(libSceFiber);

extern "C" int g_vprx_anchor_ps5_lib_sce_fiber = 1;
