// PS5 Videodec2 HLE: the native player still owns demuxing, audio and timing.
#include "runtime/vprx/vprx.h"
#include "runtime/media/videodec.h"
#ifdef DELTA_HAVE_AVCODEC
namespace {
const runtime::funcInfo functions[] = {
    {0x4670E26DC1823CACull, (void*)&runtime::video::QueryCompute}, // QueryComputeMemoryInfo
    {0x783F97D929B152DEull, (void*)&runtime::video::AllocateQueue}, // AllocateComputeQueue
    {0x52FB40DC50221786ull, (void*)&runtime::video::ReleaseQueue}, // ReleaseComputeQueue
    {0xAAA302C2550B47E1ull, (void*)&runtime::video::QueryMemory}, // QueryDecoderMemoryInfo
    {0x08D351A1161DF172ull, (void*)&runtime::video::Create}, // CreateDecoder
    {0x8F0226C5744648A0ull, (void*)&runtime::video::Delete}, // DeleteDecoder
    {0xF39D85E7EABAFA23ull, (void*)&runtime::video::Decode}, // Decode
    {0x975857C2C70BB826ull, (void*)&runtime::video::Flush}, // Flush
    {0xC095E2906E9014DFull, (void*)&runtime::video::Reset}, // Reset
    {0x36D5D16B7751CD4Dull, (void*)&runtime::video::Picture}, // GetPictureInfo
    {0x923ACB6DCCA1122Cull, (void*)&runtime::video::Picture}, // GetAvcPictureInfo
};
MODULE_INIT_PS5(libSceVideodec2);
}
#endif
extern "C" int vprx_anchor_ps5_libSceVideodec2 = 1;
