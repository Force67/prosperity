/*
 * NID -> handler table for the HLE libSceAudioIn.
 */

#include "libSceAudioIn.h"

static const runtime::funcInfo functions[] = {
    {0x4B1429AE08EDB4A1, (void *)&sceAudioInInit},
    {0xE4D13C4A373B542F, (void *)&sceAudioInOpen},
    {0x2E8CC4394F3E6A73, (void *)&sceAudioInInput},
    {0x261E966C786723AF, (void *)&sceAudioInClose},
    {0x3434E0D1DA1B0958, (void *)&sceAudioInGetStatus},
    {0x61E0523550042DEE, (void *)&sceAudioInSetConnections},
    {0x6FFD824F9072C45E, (void *)&sceAudioInGetHandleStatus},
    {0xE6A4557F13A66E7A, (void *)&sceAudioInDeviceHqOpen},
    {0x5685FD227BB0C138, (void *)&sceAudioInDeviceOpen},
    {0x4F2DC0D1CC2C4D4E, (void *)&sceAudioInDeviceClose},
    {0x83EA18C796B03100, (void *)&sceAudioInDeviceRead},
    {0xA5757C6E61BC6F4A, (void *)&sceAudioInDeviceState},
};

MODULE_INIT(libSceAudioIn);

extern "C" int vprx_anchor_libSceAudioIn = 1;
