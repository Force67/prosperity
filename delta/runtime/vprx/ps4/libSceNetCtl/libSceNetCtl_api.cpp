#include "libSceNetCtl.h"

static const runtime::funcInfo functions[] = {
    {0x824CB4FA868D3389, (void *)&sceNetCtlInit},               // gky0+oaNM4k
    {0x678C3008588110B4, (void *)&sceNetCtlTerm},               // Z4wwCFiBELQ
    {0xB813E5AF495BBA22, (void *)&sceNetCtlGetState},           // uBPlr0lbuiI
    {0xA1BBB17538B0905F, (void *)&sceNetCtlGetInfo},            // obuxdTiwkF8
    {0x509F99ED0FB8724D, (void *)&sceNetCtlRegisterCallback},   // UJ+Z7Q+4ck0
    {0x46A9B63A764C0B3D, (void *)&sceNetCtlUnregisterCallback}, // Rqm2OnZMCz0
    {0x890C378903E1BD44, (void *)&sceNetCtlCheckCallback},      // iQw3iQPhvUQ
};

MODULE_INIT(libSceNetCtl);

// Anchor referenced from vprx.cpp so the linker keeps this archive member and
// the MODULE_INIT static initializer above runs.
extern "C" int vprx_anchor_libSceNetCtl = 1;
