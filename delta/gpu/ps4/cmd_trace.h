#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include <cstdint>
#include "base/arch.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/render/command.h"

namespace gpu::ps4 {

enum class RecompStatus {
  kOk,
  kSkipped,
  kBadAddress,
  kRejected,
  kBadAttrs,
  kAttrBindings,
  kAttrOffset,
};
const char* RecompStatusName(RecompStatus status);

void NoteRegisterWrites(u32 first_reg,
                        const u32* values,
                        u32 count,
                        u32 packet_base);

void MaybeArmRootWriteWatch(const Regs& regs, u64 ps_addr, u32 frame);
using WriteWatchFn = void (*)(uintptr_t, size_t, unsigned, bool, bool);
void SetWriteWatch(WriteWatchFn watch);

void TraceRegisterSources(const Regs& regs, u64 ps_addr, u32 frame);

void TraceFirstTexturedPs(const Regs& regs, u64 ps_addr);

void TraceDrawOpcode(u32 op, u32 prim_type, u32 auto_count);
void TraceIndexBuffer(u64 index_base,
                      u32 index_count,
                      u32 index_type,
                      bool accepted);
void TraceIndirectArgs(u32 op,
                       u64 args_addr,
                       u64 indirect_base,
                       u32 offset,
                       bool mapped,
                       const u32 args[3]);
void TraceIndexOffsetArgs(u32 max_size,
                          u32 index_offset,
                          u32 index_count,
                          u64 index_base,
                          u64 resolved_base,
                          u32 index_type,
                          bool accepted);

void TraceMrtSlotZeroGap(const render::DrawInfo& d,
                         u32 target_mask,
                         u64 vs_addr,
                         u64 ps_addr);
void TraceNoMrtBound(const Regs& regs,
                     const render::DrawInfo& d,
                     u32 target_mask,
                     u64 vs_addr,
                     u64 ps_addr);

void TraceDepthState(const Regs& regs, u32 z_info);

void TraceDepthBaseWatch(const Regs& regs);

bool ShouldTraceTextureTracking(u32 frame, u64 ps_addr);

void TraceTextureFormat(const gcn::TImage& tex, const render::DrawInfo& d);

void TracePsInputCntl(const Regs& regs,
                      u64 ps_addr,
                      u32 ps_input_ena,
                      const u32* ps_in_cntl);

void TraceShaderCacheMiss(u64 vs_addr,
                          u64 ps_addr,
                          u64 vs_hash,
                          u64 ps_hash,
                          u64 fetch_hash,
                          u32 ps_input_ena,
                          u32 tex_3d_mask,
                          u32 tex_1d_mask);

void TraceRawBuffer(const char* stage,
                    u64 vs_addr,
                    const gcn::ShaderBuffer& buffer,
                    const gcn::VBuffer& resolved,
                    u64 bytes,
                    bool accepted,
                    const render::DrawInfo& d);

void TraceBlitDraw(const Regs& regs,
                   const render::DrawInfo& d,
                   u64 vs_addr,
                   u64 ps_addr);

void TraceDrawList(const Regs& regs,
                   const render::DrawInfo& d,
                   u64 vs_addr,
                   u64 ps_addr,
                   u64 fetch_addr,
                   RecompStatus status);

void TraceColorMasks(const Regs& regs,
                     const render::DrawInfo& d,
                     u64 ps_addr,
                     RecompStatus status);

void TraceSpriteDraw(const render::DrawInfo& d);

void TraceVertexAttrs(const render::DrawInfo& d);

void TraceWorldGeometry(const Regs& regs, const render::DrawInfo& d);

void TraceDrawRegisters(const Regs& regs, u32 op, const u32* body, u32 count);

void TraceComputeShader(const Regs& regs,
                        u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 user_sgpr,
                        u32 tgid_enable,
                        u32 lds_dwords);

void TraceDroppedDispatch(u64 cs_addr,
                          const u32 groups[3],
                          const u32 threads[3],
                          const char* reason);

bool ShouldTraceCsResources(u64 cs_addr);
bool CsWatchCovers(u64 base, u64 size);

void TraceCsUnresolved(u64 cs_addr, const gcn::CsResource& res);
void TraceCsCode(u64 cs_addr);
void TraceCsZeroFill(u64 cs_addr,
                     u32 binding,
                     const gcn::TImage& image,
                     const u32* descriptor);
void TraceCsUnsupportedImage(u64 cs_addr,
                             u32 binding,
                             const gcn::TImage& image,
                             const u32* descriptor);
void TraceCsWindowedBuffer(u64 cs_addr,
                           u32 binding,
                           u64 declared_size,
                           u64 mapped_size);
void TraceCsResource(u64 cs_addr,
                     const gcn::CsResource& res,
                     u64 base,
                     u64 size,
                     u64 guest_size,
                     bool image_staging,
                     const gcn::TImage& image,
                     u32 elem_bytes,
                     u32 stage_elem_bytes);
void TraceCsInvalidRange(u64 cs_addr, u32 binding, u64 base, u64 guest_size);
void TraceCsDispatch(u64 cs_addr, bool executed, u32 num_resources);

void NotePacket(u32 op);
void NotePacketCost(u32 op, u64 ns);
bool WantPacketCost();
void DumpOpcodeHistogram();

bool CeTraceOn();

void TraceConstRam(const char* what,
                   u32 ce_offset,
                   u32 num_dwords,
                   u64 addr,
                   const char* verdict,
                   u32 first_dword);

void NoteCcbPacket(u32 op);
void TraceCcbSubmit(u32 size_bytes, u32 words);
void TraceCcbHistogram(u32 words);

void TraceWaitRegMem(bool timed_out);

void TraceDmaData(u32 control,
                  u32 command,
                  u32 src_sel,
                  u32 dst_sel,
                  u64 src,
                  u64 dst,
                  u32 bytes,
                  bool copied);

void TraceAddrWatch(const char* packet,
                    u64 dst,
                    u64 bytes,
                    u32 first_dword,
                    u32 max_lines);

void TraceLabelWrite(const char* packet, u64 addr, u32 data_sel, u64 value);
void TraceEosLabel(u64 addr, u32 value);
void TraceDataWrite(u64 addr, u32 dwords, u32 first_dword);

void TraceUnhandledOpcode(u32 op, u32 count);

void TraceIndirectBuffer(u32 position, u32 depth, u32 words, bool followed);

void TraceCounter(const char* packet, u64 value);
void TraceWaitOnCeCounter(u32 wanted, u64 ce_counter);

void TraceDesync(u32 position, u32 words, u32 type, u32 hdr, bool force);

bool ShouldDumpDcb(u32 size_bytes);
void TraceDcbPacket(u32 position, u32 op, u32 count);
void TraceSubmit(const void* dcb,
                 u32 size_bytes,
                 u32 words,
                 u64 submit_number,
                 u64 draws_so_far);
void TraceDcbWalkResult(const u32* dcb, u32 words, u32 words_walked);
void TraceDcbStat(u32 words);
void MaybeDumpOpcodeHistogram(u32 words_walked, u32 words);

}  // namespace gpu::ps4
