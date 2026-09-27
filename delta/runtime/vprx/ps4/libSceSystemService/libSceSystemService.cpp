#include "guest_abi.h"
#include "libSceSystemService.h"
#include "base/arch.h"

#include "../../sys_params.h"

int PS4ABI sceSystemServiceReportAbnormalTermination(void * /*param*/) {
  return 0;  // SCE_OK; telemetry no-op
}

int PS4ABI sceSystemServiceParamGetInt(i32 paramId, i32 *value) {
  return runtime::sysparam::ParamGetInt(paramId, value);
}
