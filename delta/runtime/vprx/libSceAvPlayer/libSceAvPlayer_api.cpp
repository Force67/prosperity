#include "libSceAvPlayer.h"

static const runtime::funcInfo functions[] = {
    {0x692EBA448D201A0A, (void *)&sceAvPlayerInit},
    {0xA3D79646448BF8CE, (void *)&sceAvPlayerInitEx},
    {0x1C3D58295536EBF3, (void *)&sceAvPlayerPostInit},
    {0x28C7046BEAC7B08A, (void *)&sceAvPlayerAddSource},
    {0xC7CBAFB8538F6615, (void *)&sceAvPlayerAddSourceEx},
    {0x113E06AFF52ED3BB, (void *)&sceAvPlayerStart},
    {0x642D7BC37BC1E4BA, (void *)&sceAvPlayerStop},
    {0x3642700F32A6225C, (void *)&sceAvPlayerClose},
    {0x51B42861AC0EB1F6, (void *)&sceAvPlayerIsActive},
    {0xA37F915A71D58928, (void *)&sceAvPlayerGetVideoData},
    {0x25D92C42EF2935D4, (void *)&sceAvPlayerGetVideoDataEx},
    {0x5A7A7539572B6609, (void *)&sceAvPlayerGetAudioData},
    {0xC3033DF608C57F56, (void *)&sceAvPlayerCurrentTime},
    {0x395B61B34C467E1A, (void *)&sceAvPlayerSetLooping},
    {0x85D4F247309741E4, (void *)&sceAvPlayerStreamCount},
    {0x77C15C6F37C0750C, (void *)&sceAvPlayerGetStreamInfo},
};

MODULE_INIT(libSceAvPlayer);

// Anchor referenced from vprx.cpp so the linker keeps this archive member and
// the MODULE_INIT static initializer above runs.
extern "C" int vprx_anchor_libSceAvPlayer = 1;
