/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE override for sceSystemServiceReportAbnormalTermination, matching
 * what the PS4 HLE already does. The real .sprx aborts when handed a NULL report,
 * which is exactly how titles call it from their own fatal handlers, so the
 * crash the emulator sees is the reporter rather than whatever the title was
 * complaining about. Accept the report and let the title's error path continue.
 *
 * sceSystemServiceParamGetInt answers the console's system settings, the system
 * LANGUAGE above all -- see ../sys_params.h.
 *
 * Everything else in libSceSystemService stays LLE.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include "../sys_params.h"

namespace {
int PS4ABI systemServiceReportAbnormalTermination(void *) { return 0; }

int PS4ABI systemServiceParamGetInt(int32_t paramId, int32_t *value) {
  return runtime::sysparam::ParamGetInt(paramId, value);
}
}  // namespace

static const runtime::funcInfo functions[] = {
    {0xDECF1C1E20812811, (void *)&systemServiceReportAbnormalTermination},
    {0x7D9A38F2E9FB2CAE, (void *)&systemServiceParamGetInt},
};

MODULE_INIT_PS5(libSceSystemService);

extern "C" int vprx_anchor_ps5_libSceSystemService = 1;
