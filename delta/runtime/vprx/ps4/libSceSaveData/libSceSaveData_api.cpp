#include "libSceSaveData.h"

static const runtime::funcInfo functions[] = {
    {0x664661B2408F5C5C, (void *)&sceSaveDataInitialize},       // ZkZhskCPXFw
    {0x9753660DE0E93465, (void *)&sceSaveDataInitialize2},      // l1NmDeDpNGU
    {0x4F2C2B14A0A82C66, (void *)&sceSaveDataInitialize3},      // TywrFKCoLGY
    {0xC8A0F2F12E722C0D, (void *)&sceSaveDataTerminate},        // yKDy8S5yLA0
    {0xDF61D0010770336A, (void *)&sceSaveDataMount},            // 32HQAQdwM2o
    {0xD33E393C81FE48D2, (void *)&sceSaveDataMount2},           // 0z45PIH+SNI
    {0xC73D18322E817CD9, (void *)&sceSaveDataMount5},           // xz0YMi6BfNk
    {0x04C47817F51E9371, (void *)&sceSaveDataUmount},           // BMR4F-Uek3E
    {0x57069DC0104127CD, (void *)&sceSaveDataUmountWithBackup}, // VwadwBBBJ80
    {0xEB9547D1069ACFAB, (void *)&sceSaveDataGetMountInfo},     // 65VH0Qaaz6s
    {0x7722219D7ABFD123, (void *)&sceSaveDataDirNameSearch},    // dyIhnXq-0SM
    {0x5E0BD2B88767325C, (void *)&sceSaveDataGetParam},         // XgvSuIdnMlw
    {0xF39CEE97FFDE197B, (void *)&sceSaveDataSetParam},         // 85zul--eGXs
    {0x73CF18CB9E0CC74C, (void *)&sceSaveDataSaveIcon},         // c88Yy54Mx0w
};

MODULE_INIT(libSceSaveData);

// Anchor referenced from vprx.cpp so the linker keeps this archive member.
extern "C" int vprx_anchor_libSceSaveData = 1;
