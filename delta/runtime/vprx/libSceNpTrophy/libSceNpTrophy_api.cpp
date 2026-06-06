#include "libSceNpTrophy.h"

static const runtime::funcInfo functions[] = {
    {0x5DB9236E86D99426, (void *)&sceNpTrophyCreateContext},
    {0xABB53AB440107FB7, (void *)&sceNpTrophyCreateHandle},
    {0x1355ABC1DD3B2EBF, (void *)&sceNpTrophyDestroyContext},
    {0x18D705E2889D6346, (void *)&sceNpTrophyDestroyHandle},
    {0x6939C7B3B5BFF549, (void *)&sceNpTrophyAbortHandle},
    {0x4C9080C6DA3D4845, (void *)&sceNpTrophyRegisterContext},
    {0xDBCC6645415AA3AF, (void *)&sceNpTrophyUnlockTrophy},
    {0x2C7B9298EDD22DDF, (void *)&sceNpTrophyGetTrophyUnlockState},
    {0x6183F77F65B4F688, (void *)&sceNpTrophyGetGameInfo},
    {0xAAA515183810066D, (void *)&sceNpTrophyGetTrophyInfo},
    {0xC1353019FB292A27, (void *)&sceNpTrophyGetGroupInfo},
    {0x1CBC33D5F448C9C0, (void *)&sceNpTrophyGetGameIcon},
    {0xC38B8C3E612B0F82, (void *)&sceNpTrophyGetGroupIcon},
    {0x7812FE97A1C6F719, (void *)&sceNpTrophyGetTrophyIcon},
    {0x72A1A460037F811C, (void *)&sceNpTrophyCaptureScreenshot},
    {0x77D8E974FCF97FFF, (void *)&sceNpTrophyShowTrophyList},
};

MODULE_INIT(libSceNpTrophy);

// Anchor referenced from vprx.cpp so the linker keeps this archive member and the
// MODULE_INIT static initializer above runs.
extern "C" int vprx_anchor_libSceNpTrophy = 1;
