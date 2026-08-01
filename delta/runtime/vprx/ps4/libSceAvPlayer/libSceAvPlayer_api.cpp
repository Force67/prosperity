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
    // Playback-control entry points; see sceAvPlayerControlOk.
    {0x38324ADAC9FDC380, (void *)&sceAvPlayerControlOk}, // ODJK2sn9w4A
    {0x04E54A033466B934, (void *)&sceAvPlayerControlOk}, // BOVKAzRmuTQ
    {0xF72E6FF9F18DE169, (void *)&sceAvPlayerControlOk}, // 9y5v+fGN4Wk
    {0xC399A80013709D16, (void *)&sceAvPlayerControlOk}, // w5moABNwnRY
    {0x5C2F7033EC542F3F, (void *)&sceAvPlayerControlOk}, // XC9wM+xULz8
    {0x6AFF19FBEF78AECD, (void *)&sceAvPlayerControlOk}, // av8Z++94rs0
    {0x93FABEC4EC5D7371, (void *)&sceAvPlayerControlOk}, // k-q+xOxdc3E
};

MODULE_INIT(libSceAvPlayer);

// Anchor referenced from vprx.cpp so the linker keeps this archive member and
// the MODULE_INIT static initializer above runs.
extern "C" int vprx_anchor_libSceAvPlayer = 1;
