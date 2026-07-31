#include "libSceSystemService.h"

#include "../../sys_params.h"

int PS4ABI sceSystemServiceReportAbnormalTermination(void * /*param*/) {
  return 0;  // SCE_OK; telemetry no-op
}

int PS4ABI sceSystemServiceParamGetInt(int32_t paramId, int32_t *value) {
  return runtime::sysparam::ParamGetInt(paramId, value);
}
