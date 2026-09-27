#include "runtime/vprx/ps4/lib_sce_net_ctl/lib_sce_net_ctl.h"

static const runtime::FuncInfo functions[] = {
    {0x824CB4FA868D3389, (void*)&sceNetCtlInit},                // gky0+oaNM4k
    {0x678C3008588110B4, (void*)&sceNetCtlTerm},                // Z4wwCFiBELQ
    {0xB813E5AF495BBA22, (void*)&sceNetCtlGetState},            // uBPlr0lbuiI
    {0xA1BBB17538B0905F, (void*)&sceNetCtlGetInfo},             // obuxdTiwkF8
    {0x509F99ED0FB8724D, (void*)&sceNetCtlRegisterCallback},    // UJ+Z7Q+4ck0
    {0x46A9B63A764C0B3D, (void*)&sceNetCtlUnregisterCallback},  // Rqm2OnZMCz0
    {0x890C378903E1BD44, (void*)&sceNetCtlCheckCallback},       // iQw3iQPhvUQ
    // libSceNetCtlForNpToolkit, aliased onto this table in VprxGet.
    {0xC08B0ACBE4DF78BB,
     (void*)&sceNetCtlRegisterCallbackForNpToolkit},  // wIsKy+TfeLs
    {0xDA852A291E687467,
     (void*)&sceNetCtlUnregisterCallbackForNpToolkit},  // 2oUqKR5odGc
    {0xBB9A2AB6520FF85C,
     (void*)&sceNetCtlCheckCallbackForNpToolkit},  // u5oqtlIP+Fw
};

MODULE_INIT(libSceNetCtl);

// Anchor referenced from vprx.cc so the linker keeps this archive member and
// the MODULE_INIT static initializer above runs.
extern "C" int g_vprx_anchor_lib_sce_net_ctl = 1;
