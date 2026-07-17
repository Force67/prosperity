/*
 * PS4Delta : PS4 emulation and research project
 *
 * NID -> handler table for the HLE libSceSaveDataDialog. NIDs decoded from the
 * libSceSaveDataDialog export symbols (base64 obfuscated names).
 */

#include "libSceSaveDataDialog.h"

static const runtime::funcInfo functions[] = {
    {0xB3D7B7F98A519F3C, (void *)&sceSaveDataDialogInitialize},      // s9e3+YpRnzw
    {0x62E1F6140EDACEA4, (void *)&sceSaveDataDialogTerminate},       // YuH2FA7azqQ
    {0xE2D3E1B0FE85A432, (void *)&sceSaveDataDialogOpen},            // 4tPhsP6FpDI
    {0x7C7E3A2DA83CF176, (void *)&sceSaveDataDialogClose},           // fH46Lag88XY
    {0x1112B392C6AE0090, (void *)&sceSaveDataDialogGetStatus},       // ERKzksauAJA
    {0x28ADC1760D5158AD, (void *)&sceSaveDataDialogUpdateStatus},    // KK3Bdg1RWK0
    {0xC84889FEAAABE828, (void *)&sceSaveDataDialogGetResult},       // yEiJ-qqr6Cg
    {0x7A7EE03559E1F3BF, (void *)&sceSaveDataDialogIsReadyToDisplay}, // en7gNVnh878
    {0x85ACB509F4E62F20, (void *)&sceSaveDataDialogProgressBarSetValue}, // hay1CfTmLyA
    {0x57FB847852804495, (void *)&sceSaveDataDialogProgressBarInc}, // V-uEeFKARJU
};

MODULE_INIT(libSceSaveDataDialog);

// Linker anchor (see vprx.cpp).
extern "C" int vprx_anchor_libSceSaveDataDialog = 1;
