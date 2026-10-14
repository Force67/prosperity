#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps5/agc_regs.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/render/command.h"

namespace gpu::ps5 {

void NoteRegisterWrite(const char* source, u32 reg, u32 value);

u64 TraceRegImagePrefixDwords();

void TraceRegImage(u32 base, u64 image, const u32* body, u32 count);

enum class RegBlockOutcome {
  kApplied,
  kShortPacket,
  kBadAddress,
  kUnreadable,
  kNoPairs,
};
void NoteRegBlock(u32 base,
                  RegBlockOutcome outcome,
                  u64 address,
                  u32 num_pairs);

void TraceRegBlock(u32 base, u64 address, u32 num_pairs, u32 mode);

void TraceRegBlockAnchor(u32 pair,
                         u64 rt_base,
                         u32 info,
                         u32 width,
                         u32 height);

void TraceColorBaseWrite(u32 reg,
                         u32 value,
                         u32 pair,
                         u32 num_pairs,
                         u64 address,
                         u32 offset_dword);

void TraceShaderBind(u64 address, u32 gs_pgm_lo, u32 ps_pgm_lo);

void NoteDrawSeen();
void NoteDrawIssued(const render::DrawInfo& d);
void NoteDrawDropped(const render::DrawInfo& d,
                     u64 vs_addr,
                     u64 ps_addr,
                     size_t shader_attrs);
u64 DrawsSeen();

void NoteDrawDetail();
void TraceNggState(const Regs& regs, u64 es_addr, u64 gs_addr);

void TraceUserData(const u32* gs_user_data, const u32* es_user_data);
void TraceVsUserData(u64 vs_addr,
                     const u32* gs_user_data,
                     const u32* es_user_data,
                     bool chose_gs);
void TraceShaderScan(const u32* found_reg, const u64* found, u32 count);
void TraceUserDataPointers(const u32* vs_user_data, const u32* ps_user_data);

void TraceIndexBuffer(u32 op, const render::DrawInfo& d, u64 index_size);

void TraceRtProbe(const Regs& regs, const render::DrawInfo& d);
void TraceNoRenderTarget(const Regs& regs, u32 target_mask);
void TraceRenderTarget(const Regs& regs,
                       const render::DrawInfo& d,
                       u64 vs_addr,
                       u64 ps_addr);

void TraceRecompile(u64 vs_addr, u64 ps_addr, u32 ps_input_ena);
void TraceRecompileDone(bool ok);
void TraceRecompileFailed(u64 vs_addr, u64 ps_addr);

void TraceAttrPlan(size_t attrs,
                   u32 vs_user_sgprs,
                   size_t resolved,
                   const u32* vs_user_data);
void TraceAttrReplay(u32 index, u32 use_pc, bool found, bool valid);
void TraceAttr(u32 index,
               const gcn::ShaderAttr& attr,
               const rdna::VBuffer& vb,
               u32 fetch_soffset,
               const char* how);

void TraceRawBufBinding(bool vertex_stage,
                        u32 binding,
                        u32 use_pc,
                        u32 srsrc_sgpr,
                        bool replayed,
                        u64 base,
                        u64 bytes);
void TraceCbufBinding(bool vertex_stage,
                      u32 binding,
                      u32 use_pc,
                      u64 base,
                      u32 num_dwords);

void TraceRejectedTexture(u32 binding, const gcn::TImage& tex);
void TraceDrawTextures(const render::DrawInfo& d);

void TraceVertexDump(const render::DrawInfo& d,
                     const u32* vs_user_data,
                     u64 vs_addr,
                     u64 ps_addr);

void TraceBeginFrame();
void TraceDrawSubmit(const render::DrawInfo& d);
void TraceDrawDone();

void TraceShaderListing(u64 address);

void NoteDispatch(u64 cs_addr, const u32 threads[3], u32 rsrc2);
void TraceComputeShader(const Regs& regs,
                        u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 rsrc2);

void TraceCsUnsupported(u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 user_sgpr);
void TraceCsUnresolved(u64 cs_addr, const gcn::CsResource& res, u32 ud_dwords);
void TraceCsUnsupportedImage(u64 cs_addr,
                             u32 binding,
                             const gcn::TImage& image,
                             const u32* descriptor);
void TraceCsNoLinearStaging(u64 cs_addr, u32 binding, const gcn::TImage& image);
void TraceCsInvalidRange(u64 cs_addr,
                         const gcn::CsResource& res,
                         u64 base,
                         u64 guest_size);
void TraceCsTooManyResources(u64 cs_addr, u32 max_resources);
void TraceCsDispatchFailed(u64 cs_addr, u32 num_resources);

void TraceCsResource(u64 cs_addr,
                     const gcn::CsResource& res,
                     u64 base,
                     u64 size,
                     u64 guest_size,
                     bool zero_fill);
void TraceCsDispatch(u64 cs_addr, bool executed, u32 num_resources);

void NoteOpcode(u32 op);

void NoteSkippedOpcode(u32 op, const char* why);

void NoteUnhandledOpcode(u32 op,
                         u32 hdr,
                         u32 position,
                         u32 words,
                         const u32* body,
                         u32 count,
                         u32 prev_op);

void TraceOpcodeBody(u32 op, const u32* body, u32 count);

void TraceColorBaseSource(u32 op, u32 color0_base);

void TraceDcbPacket(u32 position, u32 op, const u32* body, u32 count);

void TraceIndirectBuffer(u64 address, u32 words, bool followed);
void TraceOcclusionQuery(u64 address, u64 value);
void TraceResync(u32 position, u32 words, u32 hdr, u32 op, u32 count);

void TraceType0ShaderRegs(u32 first_reg, const u32* values, u32 count);

void TraceDmaData(u32 control, u64 src, u64 dst, u32 bytes, bool copied);

bool TraceSubmit(const void* dcb, u32 size_bytes, u32 words, u64 submission);

void TraceOpcodeCensus(const void* dcb, u32 words, u64 submission);
void TraceWalkDone();

void TraceRegShadowScan();

}  // namespace gpu::ps5
