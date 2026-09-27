#include "runtime/vprx/ps4/lib_sce_system_service/lib_sce_system_service.h"
#include "base/arch.h"
#include "guest_abi.h"

#include "runtime/vprx/sys_params.h"

int PS4ABI sceSystemServiceReportAbnormalTermination(void* /*param*/) {
  return 0;  // SCE_OK; telemetry no-op
}

int PS4ABI sceSystemServiceParamGetInt(i32 param_id, i32* value) {
  return runtime::sysparam::ParamGetInt(param_id, value);
}
