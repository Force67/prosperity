/*
 * PS4Delta : PS4 emulation and research project
 *
 * NID -> handler table for the HLE libSceMsgDialog. NIDs verified against the
 * decrypted libSceMsgDialog.sprx export table.
 */

#include "libSceMsgDialog.h"

static const runtime::funcInfo functions[] = {
    {0x943AB1698D546C4A, (void *)&sceMsgDialogInitialize},        // lDqxaY1UbEo
    {0x78FC3F92A6667A5A, (void *)&sceMsgDialogTerminate},         // ePw-kqZmelo
    {0x6F4E878740CF11A1, (void *)&sceMsgDialogOpen},              // b06Hh0DPEaE
    {0x1D3ADC0CA9452AE3, (void *)&sceMsgDialogClose},             // HTrcDKlFKuM
    {0xE9F202DD72ADDA4D, (void *)&sceMsgDialogUpdateStatus},      // 6fIC3XKt2k0
    {0x096556EFC41CDDF2, (void *)&sceMsgDialogGetStatus},         // CWVW78Qc3fI
    {0x2EBF28BC71FD97A0, (void *)&sceMsgDialogGetResult},         // Lr8ovHH9l6A
    {0xC13A5F825926BF7E, (void *)&sceMsgDialogProgressBarSetValue}, // wTpfglkmv34
    {0x19CE64D6A70AE1FB, (void *)&sceMsgDialogProgressBarInc},    // Gc5k1qcK4fs
    {0xE87FFBD4E76BA573, (void *)&sceMsgDialogProgressBarSetMsg}, // 6H-71OdrpXM
};

MODULE_INIT(libSceMsgDialog);

// Linker anchor (see vprx.cpp).
extern "C" int vprx_anchor_libSceMsgDialog = 1;
