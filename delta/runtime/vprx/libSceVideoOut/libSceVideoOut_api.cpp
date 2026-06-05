/*
 * PS4Delta : PS4 emulation and research project
 *
 * NID -> handler table for the HLE libSceVideoOut. NIDs verified against the
 * decrypted libSceVideoOut.sprx export table.
 */

#include "libSceVideoOut.h"

static const runtime::funcInfo functions[] = {
    {0x529DFA3D393AF3B1, (void *)&sceVideoOutOpen},                   // Up36PTk687E
    {0xBAAB951F8FC3BBBF, (void *)&sceVideoOutClose},                  // uquVH4-Du78
    {0xEA43E78F9D53EB66, (void *)&sceVideoOutGetResolutionStatus},    // 6kPnj51T62Y
    {0x8BAFEC47DD56B7FE, (void *)&sceVideoOutSetBufferAttribute},     // i6-sR91Wt-4
    {0xC37058FAD0048906, (void *)&sceVideoOutRegisterBuffers},        // w3BY+tAEiQY
    {0x379283B642238C9E, (void *)&sceVideoOutUnregisterBuffers},      // N5KDtkIjjJ4
    {0x0818AEE26084D430, (void *)&sceVideoOutSetFlipRate},            // CBiu4mCE1DA
    {0x1D7CE32BDC88DF49, (void *)&sceVideoOutAddFlipEvent},           // HXzjK9yI30k
    {0xFCECE7D05D401518, (void *)&sceVideoOutDeleteFlipEvent},        // -Ozn0F1AFRg
    {0x5EBBBDDB01C94668, (void *)&sceVideoOutAddVblankEvent},         // Xru92wHJRmg
    {0x32DE101C793190E7, (void *)&sceVideoOutGetEventCount},          // Mt4QHHkxkOc
    {0x536249B52A8D2992, (void *)&sceVideoOutGetEventId},             // U2JJtSqNKZI
    {0xAD651370A7645334, (void *)&sceVideoOutGetEventData},           // rWUTcKdkUzQ
    {0x538E8DC0E889A72B, (void *)&sceVideoOutSubmitFlip},             // U46NwOiJpys
    {0x8FCC65FBDD80D2AE, (void *)&sceVideoOutSubmitFlipEop},          // j8xl+92A0q4
    {0x49B537770A7CD254, (void *)&sceVideoOutGetFlipStatus},          // SbU3dwp80lQ
    {0xCE05E27C74FD12B6, (void *)&sceVideoOutIsFlipPending},          // zgXifHT9ErY
    {0xD456412B2F0778D5, (void *)&sceVideoOutGetVblankStatus},        // 1FZBKy8HeNU
    {0x8FA45A01495A2EFD, (void *)&sceVideoOutWaitVblank},             // j6RaAUlaLv0
    {0x39C4326D07A31C46, (void *)&sceVideoOutGetBufferLabelAddress},  // OcQybQejHEY
    {0x313C71ACE09E4A28, (void *)&sceVideoOutSetWindowModeMargins},   // MTxxrOCeSig
    {0x0D886159B2527918, (void *)&sceVideoOutColorSettingsSetGamma_}, // DYhhWbJSeRg
    {0xA63903B20C658BA7, (void *)&sceVideoOutModeSetAny_},            // pjkDsgxli6c
};

MODULE_INIT(libSceVideoOut);

// Anchor referenced by vprx_init() so the linker keeps this TU (and thus the
// MODULE_INIT static initializer) instead of dropping the unreferenced archive
// member. Without this the HLE table never registers.
extern "C" int vprx_anchor_libSceVideoOut = 1;
