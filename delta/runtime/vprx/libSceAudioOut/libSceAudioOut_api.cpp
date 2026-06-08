/*
 * PS4Delta : PS4 emulation and research project
 *
 * NID -> handler table for the HLE libSceAudioOut. NIDs computed from the symbol
 * names with the PS4 NID hash (verified against known libSceGnmDriver NIDs).
 */

#include "libSceAudioOut.h"

static const runtime::funcInfo functions[] = {
    {0x25F10F5D5C6116A0, (void *)&sceAudioOutInit},
    {0x7A436FB13DB6AEC6, (void *)&sceAudioOutOpen},
    {0x40E42D6DE0EAB13E, (void *)&sceAudioOutOutput},
    {0xB35FFFB84F66045C, (void *)&sceAudioOutClose},
    {0x6FEB8057CF489711, (void *)&sceAudioOutSetVolume},
    {0xC373DD6924D2C061, (void *)&sceAudioOutOutputs},
    {0x1AB43DB3822B35A4, (void *)&sceAudioOutGetPortState},
    {0x3ED96DB37DBAA5DB, (void *)&sceAudioOutGetLastOutputTime},
    {0xBFFBC02CA72EF31D, (void *)&sceAudioOutSetVolumeDc},
    {0x9F5E8A768C67BE5D, (void *)&sceAudioOutInitIpmiGetSession},
};

MODULE_INIT(libSceAudioOut);

// Linker anchor (see vprx.cpp): keeps this archive member + its MODULE_INIT.
extern "C" int vprx_anchor_libSceAudioOut = 1;
