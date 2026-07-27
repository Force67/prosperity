/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE alias for the one libSceNgs2 export missing from firmware 01.14.00.
 * All 12 call sites build the command struct in place and pass it in -- there is no
 * out-parameter to fill, and the sites that check the result treat non-zero as
 * fatal, so report success.
 *
 * Everything else in libSceNgs2 stays LLE.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstdint>

namespace {
// f(handle, const void *cmds, uint32_t count): command submission, all inputs.
int PS4ABI ngs2SubmitCommands(void *, const void *, uint32_t) { return 0; }
}  // namespace

static const runtime::funcInfo functions[] = {
    {0x01B62F4CE67C3EDB, (void *)&ngs2SubmitCommands},  // AbYvTOZ8Pts
};

MODULE_INIT_PS5(libSceNgs2);

extern "C" int vprx_anchor_ps5_libSceNgs2 = 1;
