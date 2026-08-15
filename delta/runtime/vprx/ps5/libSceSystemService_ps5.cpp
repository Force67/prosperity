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
#include "base/arch.h"

#include "../sys_params.h"

namespace {
int PS4ABI systemServiceReportAbnormalTermination(void *) { return 0; }

int PS4ABI systemServiceParamGetInt(i32 paramId, i32 *value) {
  return runtime::sysparam::ParamGetInt(paramId, value);
}

// sceLncUtilGetAppStatus: {u32 appId, u32, u32 state}. State 4 means "running
// in the foreground" (kern/ipmi/svc_lnc.cpp answers the PS4 query the same
// way); the PS5 LLE path queries a SceLncService method we don't implement and
// reads back a zeroed reply, and sceNpWebApi2Initialize returns 0x8055c102
// unless the state is 4 or 5 (Demon's Souls' Crossgen init verifies it).
int PS4ABI lncUtilGetAppStatus(u32 *status) {
  if (!status)
    return -1;
  status[0] = 0x60000001;  // ipmi kForegroundAppId
  status[1] = 0;
  status[2] = 4;  // running, foreground
  return 0;
}
}  // namespace

static const runtime::funcInfo functions[] = {
    {0xDECF1C1E20812811, (void *)&systemServiceReportAbnormalTermination},
    {0x7D9A38F2E9FB2CAE, (void *)&systemServiceParamGetInt},
    {0x0F14648BB4F6138E, (void *)&lncUtilGetAppStatus},
};

MODULE_INIT_PS5(libSceSystemService);

extern "C" int vprx_anchor_ps5_libSceSystemService = 1;
