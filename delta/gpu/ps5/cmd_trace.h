#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The DELTA_AGC_* / DELTA_GPU_* instrumentation of the PS5 command stream: one
 * function per knob, each taking the state it reports on and each a no-op
 * unless its knob is set. What a probe remembers between calls (line caps,
 * histograms, the draw counters nothing else reads) lives behind these
 * functions rather than in the code being observed.
 *
 * Much of this is archaeology: the AGC binds state through register shadow
 * images and (offset, value) blocks whose encoding had to be recovered from the
 * command stream itself, and the probes that recovered it are kept because the
 * next title programs its state a way the last one did not.
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps5/agc_regs.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/rhi/command.h"

namespace gpu::ps5 {

// --- register writes -------------------------------------------------------

// Every register the command stream writes, whatever packet carried it.
// `source` names that packet. Feeds DELTA_AGC_CLIPTRACE (the clip convention
// lives in one register, and a title whose writes never reach us silently gets
// the reset value), DELTA_AGC_UDTRACE (the PS user-data SGPRs above 15, which a
// shader samples a descriptor out of whether or not anything programmed them)
// and DELTA_AGC_SHCENSUS (which SH offsets this title writes at all: the ones
// it does not write are the ones we read as zero).
void NoteRegisterWrite(const char* source, u32 reg, u32 value);

// --- register images -------------------------------------------------------

// The image dumps read a fixed prefix of the image rather than only the ranges
// the packet names, so the caller's readability check has to cover it. Zero
// when they are off.
u64 TraceRegImagePrefixDwords();

// DELTA_AGC_TRACE: what a LOAD_*_REG image holds. A register shadow the title
// never wrote reads as all zeros, which is indistinguishable downstream from a
// title that programs no state; the ranges, both candidate readings of the
// image (offset-indexed and cursor-packed) and any shader pointer in it are how
// that was settled. `body` is the packet body, `image` its resolved address.
void TraceRegImage(u32 base, u64 image, const u32* body, u32 count);

// --- state blocks ----------------------------------------------------------

// DELTA_AGC_REGSTAT: why an indirect register load carried nothing. Every one
// that bails is a whole draw's state (render target, shaders) left at whatever
// the shadow held, which downstream looks like "the title drew nothing" rather
// than "we dropped its state".
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

// DELTA_AGC_REGSTAT>=2: a whole state block, entry by entry (>=3 trades the
// full dump for a non-zero digest of ten times as many blocks, so a steady-state
// frame's state stream fits in one run). DELTA_AGC_REGSTAT_FROM=<draw> skips
// the init blocks and starts at the draw the frame is really made of.
void TraceRegBlock(u32 base, u64 address, u32 num_pairs, u32 mode);

// DELTA_AGC_REGSTAT: the entry a colour-target bind was anchored on.
void TraceRegBlockAnchor(u32 pair,
                         u64 rt_base,
                         u32 info,
                         u32 width,
                         u32 height);

// DELTA_AGC_CBTRACE=<draw>: every write to the colour-target base/format with
// the packet's own bookkeeping, so a zero write can be traced to a mis-parse.
void TraceColorBaseWrite(u32 reg,
                         u32 value,
                         u32 pair,
                         u32 num_pairs,
                         u64 address,
                         u32 offset_dword);

// DELTA_AGC_TRACE: the first few state blocks that leave a non-zero shader
// program address behind -- the real pipeline binds.
void TraceShaderBind(u64 address, u32 gs_pgm_lo, u32 ps_pgm_lo);

// --- draw accounting -------------------------------------------------------

// DELTA_GPU_DRAWCENSUS reports these every 15s. A draw dropped before the
// renderer is indistinguishable from one the title never issued, so both ends
// are counted; `DrawsSeen` is also what the draw-indexed trace gates read.
void NoteDrawSeen();
void NoteDrawIssued(const rhi::DrawInfo& d);
void NoteDrawDropped(const rhi::DrawInfo& d,
                     u64 vs_addr,
                     u64 ps_addr,
                     size_t shader_attrs);
u64 DrawsSeen();

// --- one draw, in detail ---------------------------------------------------

// Every probe below reports on the draw named by the last NoteDrawDetail, which
// the draw path calls once, when a draw reaches shader resolution. The walk
// holds the register lock, so there is only ever one such draw in flight.
void NoteDrawDetail();

// The two shader-address recovery dumps: the user-data windows themselves, and
// (for a pipeline whose program addresses are not in the registers we read) the
// register-file scan, the pointer at SH 0x113 and the tables user data names.
void TraceUserData(const u32* gs_user_data, const u32* es_user_data);
void TraceShaderScan(const Regs& regs, const u32* found_reg, const u64* found,
                     u32 count);
void TraceUserDataPointers(const u32* vs_user_data, const u32* ps_user_data);

// The index buffer a draw packet resolved to, and its first few indices.
void TraceIndexBuffer(u32 op, const rhi::DrawInfo& d, u64 index_size);

// DELTA_AGC_RTPROBE: one line per draw naming the colour state it ran with. A
// draw with no target and a stale last write points at a bind we never
// executed; one whose last write zeroed the base points at a packet we execute
// and should not.
void TraceRtProbe(const Regs& regs, const rhi::DrawInfo& d);
// Why a colour draw ended up with no target: the state that rejected CB_COLOR0.
void TraceNoRenderTarget(const Regs& regs, u32 target_mask);
// The resolved target of a draw, next to the registers it came from.
void TraceRenderTarget(const Regs& regs,
                       const rhi::DrawInfo& d,
                       u64 vs_addr,
                       u64 ps_addr);

// The recompile of one VS/PS pair. A pair the translator declines drops its
// draw, which looks exactly like the title never issuing it.
void TraceRecompile(u64 vs_addr, u64 ps_addr, u32 ps_input_ena);
void TraceRecompileDone(bool ok);
void TraceRecompileFailed(u64 vs_addr, u64 ps_addr);

// How each vertex attribute was resolved: the plan, the descriptor replay's
// verdict on it, and the V# it ended up with.
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

void TraceCbufBinding(bool vertex_stage,
                      u32 binding,
                      u32 use_pc,
                      u64 base,
                      u32 num_dwords);

// DELTA_GPU_TEXVALID: a rejected T# has its base erased, after which it is
// indistinguishable from a descriptor that was never written. Report the two
// apart before that happens.
void TraceRejectedTexture(u32 binding, const gcn::TImage& tex);
void TraceDrawTextures(const rhi::DrawInfo& d);

// DELTA_AGC_VDUMP*: the raw vertex bytes of a resolved draw as float32 and
// uint32, its transform and constant buffers, and optionally the decoded
// program (VDUMPPROG / VDUMPPS) and a host-side projection of its first
// vertices (VDUMPPROJ) -- which say whether positions are garbage (wrong
// format), screen-space (missing projection) or clip-space (a later problem).
// VDUMPFROM/VDUMPRT/VDUMPIC pick the draw: indices shift between runs, a target
// address and an index count do not.
void TraceVertexDump(const rhi::DrawInfo& d,
                     const u32* vs_user_data,
                     u64 vs_addr,
                     u64 ps_addr);

// The state a draw is handed to the renderer with, and the frame boundary it
// opened.
void TraceBeginFrame();
void TraceDrawSubmit(const rhi::DrawInfo& d);
void TraceDrawDone();

// DELTA_AGC_DUMPSH=<hexaddr>: decode and print one shader by address, once.
// Shader dumps are otherwise tied to recompile time or to a draw index, neither
// of which is reachable for a steady-state pass without a full trace.
void TraceShaderListing(u64 address);

// --- compute ---------------------------------------------------------------

// DELTA_GPU_CSDUMP: how many dispatches name a plausible program, and (per
// distinct shader) its user data, the tables that user data points at, and an
// encoding census saying which instruction families a compute backend must
// cover before the dispatch can run.
void NoteDispatch(u64 cs_addr, const u32 threads[3], u32 rsrc2);
void TraceComputeShader(const Regs& regs,
                        u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 rsrc2);

// A dispatch dropped here is indistinguishable from one the title never issued
// (its writes just never appear), so every skip is reported. Rate-limited
// across all reasons: a title that dispatches every frame would flood the log.
void TraceCsUnsupported(u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 user_sgpr);
void TraceCsUnresolved(u64 cs_addr, const gcn::CsResource& res, u32 ud_dwords);
void TraceCsUnsupportedImage(u64 cs_addr,
                             u32 binding,
                             const gcn::TImage& image);
void TraceCsNoLinearStaging(u64 cs_addr,
                            u32 binding,
                            const gcn::TImage& image);
void TraceCsInvalidRange(u64 cs_addr,
                         const gcn::CsResource& res,
                         u64 base,
                         u64 guest_size);
void TraceCsTooManyResources(u64 cs_addr, u32 max_resources);
void TraceCsDispatchFailed(u64 cs_addr, u32 num_resources);

// DELTA_GPU_CSRES: the guest range every resource of a dispatch resolved to.
void TraceCsResource(u64 cs_addr,
                     const gcn::CsResource& res,
                     u64 base,
                     u64 size,
                     u64 guest_size,
                     bool zero_fill);
void TraceCsDispatch(u64 cs_addr, bool executed, u32 num_resources);

// --- the packet stream -----------------------------------------------------

// Every type-3 packet, for the opcode histogram.
void NoteOpcode(u32 op);

// DELTA_AGC_OPDUMP=<hex op>: the first few bodies of ONE opcode, wherever it
// occurs. The submit-level dump only covers early init submits, so a packet
// that appears only in steady-state frames is otherwise invisible.
void TraceOpcodeBody(u32 op, const u32* body, u32 count);

// DELTA_AGC_TRACE: which packet first makes CB_COLOR0_BASE non-zero, and every
// later change -- that is, who actually binds the render target.
void TraceColorBaseSource(u32 op, u32 color0_base);

// One packet of a dumped submission, with the GPU buffer an indirect register
// packet names expanded.
void TraceDcbPacket(u32 position, u32 op, const u32* body, u32 count);

// DELTA_AGC_WALKSTAT: a packet whose count runs past the buffer means we
// mis-parsed something earlier; the walker resyncs a dword at a time and every
// packet in between is lost.
void TraceIndirectBuffer(u64 address, u32 words, bool followed);
void TraceResync(u32 position, u32 words, u32 hdr, u32 op, u32 count);

// DELTA_AGC_TRACE: type-0 writes into SH space. The AGC driver programs shader
// state this way, which is why no SET_SH_REG carries it.
void TraceType0ShaderRegs(u32 first_reg, const u32* values, u32 count);

// DELTA_GPU_DMATRACE: CP DMA packets and whether the copy was performed.
void TraceDmaData(u32 control, u64 src, u64 dst, u32 bytes, bool copied);

// --- submissions -----------------------------------------------------------

// DELTA_AGC_TRACE: the first few submissions that carry packets are dumped in
// full -- the raw prefix here, then every packet through TraceDcbPacket.
// Returns whether this submission is one of them.
bool TraceSubmit(const void* dcb, u32 size_bytes, u32 words, u64 submission);

// The cumulative opcode census, periodically (so draw opcodes that appear only
// in later, undumped submits are still visible) and after a dumped walk.
void TraceOpcodeCensus(const void* dcb, u32 words, u64 submission);
void TraceWalkDone();

// DELTA_AGC_TRACE: one scan of the 2 MiB SceAgcRegShadow, deep into a run. If
// setShader wrote its state to a different shadow than the one LOAD_SH_REG
// reads, it is in there.
void TraceRegShadowScan();

}  // namespace gpu::ps5
