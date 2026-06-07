#include "libSceSystemService.h"

int PS4ABI sceSystemServiceReportAbnormalTermination(void * /*param*/) {
  return 0;  // SCE_OK; telemetry no-op
}
