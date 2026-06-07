#include "libSceSystemService.h"

static const runtime::funcInfo functions[] = {
    {0xdecf1c1e20812811, (void *)&sceSystemServiceReportAbnormalTermination},  // 3s8cHiCBKBE
};

MODULE_INIT(libSceSystemService);

// Anchor referenced from vprx.cpp so the linker keeps this archive member and
// the MODULE_INIT static initializer above runs.
extern "C" int vprx_anchor_libSceSystemService = 1;
