// PS4 Videodec2 HLE. The LLE .sprx drives a VDEC engine we don't emulate
// (Uncharted 2's logo movies fail in VDECCORE); the API and structures match
// PS5's, so both share the CPU H.264 decoder.
#include "runtime/media/videodec.h"
#include "runtime/vprx/vprx.h"
#ifdef DELTA_HAVE_AVCODEC
namespace {
static const runtime::vprx::ExportEntry kExports[] = {
    {0x4670E26DC1823CACull,
     (void*)&runtime::video::QueryCompute},  // QueryComputeMemoryInfo
    {0x783F97D929B152DEull,
     (void*)&runtime::video::AllocateQueue},  // AllocateComputeQueue
    {0x52FB40DC50221786ull,
     (void*)&runtime::video::ReleaseQueue},  // ReleaseComputeQueue
    {0xAAA302C2550B47E1ull,
     (void*)&runtime::video::QueryMemory},  // QueryDecoderMemoryInfo
    {0x08D351A1161DF172ull, (void*)&runtime::video::Create},   // CreateDecoder
    {0x8F0226C5744648A0ull, (void*)&runtime::video::Delete},   // DeleteDecoder
    {0xF39D85E7EABAFA23ull, (void*)&runtime::video::Decode},   // Decode
    {0x975857C2C70BB826ull, (void*)&runtime::video::Flush},    // Flush
    {0xC095E2906E9014DFull, (void*)&runtime::video::Reset},    // Reset
    {0x36D5D16B7751CD4Dull, (void*)&runtime::video::Picture},  // GetPictureInfo
    {0x923ACB6DCCA1122Cull,
     (void*)&runtime::video::Picture},  // GetAvcPictureInfo
};
MODULE_INIT(libSceVideodec2);
}  // namespace
#endif
extern "C" int g_vprx_anchor_lib_sce_videodec2 = 1;
