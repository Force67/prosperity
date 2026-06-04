#include "libSceUserService.h"

static const runtime::funcInfo functions[] = {
    {0x8F760CBB531534DA, (void *)&sceUserServiceInitialize},
    {0x6B3FF447A7AF899D, (void *)&sceUserServiceInitialize2},
    {0x6F01634BE6D7F660, (void *)&sceUserServiceTerminate},
    {0xC87D7B43A356B558, (void *)&sceUserServiceGetEvent},
    {0x7CF87298A36F2BF0, (void *)&sceUserServiceGetLoginUserIdList},
    {0x09D5A9D281D61ABD, (void *)&sceUserServiceGetInitialUser},
    {0x78D6F9DCB4099883, (void *)&sceUserServiceGetForegroundUser},
    {0xD71C5C3221AED9FA, (void *)&sceUserServiceGetUserName},
};

MODULE_INIT(libSceUserService);

// Anchor referenced from vprx.cpp so the linker keeps this archive member and
// the MODULE_INIT static initializer above runs.
extern "C" int vprx_anchor_libSceUserService = 1;
