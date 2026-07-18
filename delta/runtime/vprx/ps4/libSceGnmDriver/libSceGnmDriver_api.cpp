/*
 * PS4Delta : PS4 emulation and research project
 *
 * NID -> handler table for the HLE libSceGnmDriver submit entry points. NIDs
 * verified against the decrypted libSceGnmDriver.sprx export table.
 */

#include "libSceGnmDriver.h"

static const runtime::funcInfo functions[] = {
    {0xCF0634615F754D32, (void *)&sceGnmSubmitCommandBuffers},                    // zwY0YV91TTI
    {0x8D1708F157204F3E, (void *)&sceGnmSubmitCommandBuffersForWorkload},         // jRcI8VcgTz4
    {0xC5BC4D6AD6B0A217, (void *)&sceGnmSubmitAndFlipCommandBuffers},             // xbxNatawohc
    {0x19AEABEC7E98D112, (void *)&sceGnmSubmitAndFlipCommandBuffersForWorkload},  // Ga6r7H6Y0RI
    {0xCAF67BDEE414AAB9, (void *)&sceGnmSubmitDone},                              // yvZ73uQUqrk
    {0x6F4F0082D3E51CF8, (void *)&sceGnmAreSubmitsAllowed},                       // b08AgtPlHPg
    {0x6D7E486D1BC40979, (void *)&sceGnmDingDong},                               // bX5IbRvECXk
    {0x6F25E5AAEA5DF1C1, (void *)&sceGnmDingDongForWorkload},                     // byXlqupd8cE
    {0x881B7739ED342AF7, (void *)&sceGnmFlushGarlic},                            // iBt3Oe00Kvc
    {0xD6A5CB1C8A5138F1, (void *)&sceGnmInsertWaitFlipDone},                      // 1qXLHIpROPE
};

MODULE_INIT(libSceGnmDriver);

// Linker anchor (see vprx.cpp): keeps this archive member + its MODULE_INIT.
extern "C" int vprx_anchor_libSceGnmDriver = 1;
