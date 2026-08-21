#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The DELTA_GPU_* instrumentation of the PS4 command stream: one function per
 * knob, each taking the state it reports on and each a no-op unless its knob is
 * set. What a probe remembers between calls (line caps, histograms, the
 * register-write bookkeeping nothing else reads) lives behind these functions
 * rather than in the code being observed.
 *
 * The frame debugger (DELTA_GPU_CAPTURE, ../DEBUGGER.md) is the richer tool.
 * These answer questions that span a whole run rather than one frame.
 */

#include <cstdint>
#include "base/arch.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/rhi/command.h"

namespace gpu::ps4 {

// Why a draw did or did not reach the recompiled-shader path. A draw that
// renders nothing looks the same either way from the outside.
enum class RecompStatus {
  kOk,
  kDisabled,      // DELTA_GPU_RECOMP=0
  kSkipped,       // DELTA_GPU_SKIPSH names this VS or PS
  kBadAddress,    // VS/PS program address outside the guest window
  kRejected,      // the translator declined the shader pair
  kBadAttrs,      // an attribute's V# did not resolve
  kAttrBindings,  // more distinct vertex buffers than bindings
  kAttrOffset,    // an attribute lies outside its binding's record
};
const char* RecompStatusName(RecompStatus status);

// --- register file ---------------------------------------------------------

// `packet_base` is the packet's register base, zero for a type-0 write. Keeps
// the CB_SHADER_MASK / CB_TARGET_MASK write counts (a zero mask reads the same
// whether it was cleared or never written) and the per-register source pointer
// DELTA_GPU_REGSRC reports.
void NoteRegisterWrites(u32 first_reg,
                        const u32* values,
                        u32 count,
                        u32 packet_base);

// --- draws -----------------------------------------------------------------

// DELTA_GPU_ROOT_WPROT_*: watch the guest memory a shader resource root lives
// in. Armed once, on the first draw whose PS matches the address or code hash.
void MaybeArmRootWriteWatch(const Regs& regs, u64 ps_addr, u32 frame);
// Installed by the composition root (SetWriteWatchCallback); the watch above is
// its only user.
using WriteWatchFn = void (*)(uintptr_t, size_t, unsigned, bool, bool);
void SetWriteWatch(WriteWatchFn watch);

// DELTA_GPU_REGSRC_PS: the user-data SGPRs of one pixel shader with the packet
// word each was written from.
void TraceRegisterSources(const Regs& regs, u64 ps_addr, u32 frame);

// DELTA_GPU_TRACE: the first pixel shader that samples a texture, with its
// user data and what TrackTextures made of it.
void TraceFirstTexturedPs(const Regs& regs, u64 ps_addr);

// DELTA_GPU_DRAWPKT: which opcode each draw arrives as, and how each index
// packet was decoded. A count that never leaves the packet shows up downstream
// as "too few vertices", which points at the wrong layer.
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

// DELTA_GPU_NOMRT: draws whose colour targets do not add up. A write mask
// enabling a target the CB registers never name renders into nothing, and a gap
// at slot 0 drops the draw downstream; both show up only as a black target.
void TraceMrtSlotZeroGap(const rhi::DrawInfo& d,
                         u32 target_mask,
                         u64 vs_addr,
                         u64 ps_addr);
void TraceNoMrtBound(const Regs& regs,
                     const rhi::DrawInfo& d,
                     u32 target_mask,
                     u64 vs_addr,
                     u64 ps_addr);

// DELTA_GPU_DBTRACE: the depth/stencil registers of every depth-testing draw.
// `z_info` is the value the draw path acted on, which DELTA_GPU_NODEPTH forces
// to zero.
void TraceDepthState(const Regs& regs, u32 z_info);

// DELTA_GPU_DBWATCH: does an address ever appear in a DB base register? A
// surface read as depth that nothing writes is either CPU-filled or a plane the
// depth path never registers; only the registers tell those apart.
void TraceDepthBaseWatch(const Regs& regs);

// DELTA_GPU_TEXTRACK_FRAME: whether TrackTextures should run in tracing mode
// for this draw.
bool ShouldTraceTextureTracking(u32 frame, u64 ps_addr);

// DELTA_GPU_TEXFMT: the format/tiling of sampled textures, to pin a scrambled
// draw to an encoding we mishandle.
void TraceTextureFormat(const gcn::TImage& tex, const rhi::DrawInfo& d);

// DELTA_GPU_PSINCNTL: the VS parameter export each PS input slot reads.
void TracePsInputCntl(const Regs& regs,
                      u64 ps_addr,
                      u32 ps_input_ena,
                      const u32* ps_in_cntl);

// DELTA_GPU_SHRELOC: attribute a recompile-cache miss to the key field that
// changed since this VS/PS content pair was last compiled. The distinct-value
// tallies say whether a churning field is bounded (the cache converges) or
// unbounded (it misses forever).
void TraceShaderCacheMiss(u64 vs_addr,
                          u64 ps_addr,
                          u64 vs_hash,
                          u64 ps_hash,
                          u64 fetch_hash,
                          u32 ps_input_ena,
                          u32 tex_3d_mask,
                          u32 tex_1d_mask);

// DELTA_GPU_RAWBUF: the V# behind each buffer a recompiled stage reads by hand.
void TraceRawBuffer(const char* stage,
                    u64 vs_addr,
                    const gcn::ShaderBuffer& buffer,
                    const gcn::VBuffer& resolved,
                    u64 bytes,
                    bool accepted,
                    const rhi::DrawInfo& d);

// DELTA_GPU_BLITDUMP: the full state of the first few draws into a
// scanout-sized target, with the PS listing and its resolved constants.
void TraceBlitDraw(const Regs& regs,
                   const rhi::DrawInfo& d,
                   u64 vs_addr,
                   u64 ps_addr);

// DELTA_GPU_DRAWLIST: one line per draw before any gating, so draws dropped for
// unresolved vertex data or shaders are visible.
void TraceDrawList(const Regs& regs,
                   const rhi::DrawInfo& d,
                   u64 vs_addr,
                   u64 ps_addr,
                   u64 fetch_addr,
                   RecompStatus status);

// DELTA_GPU_MASKTRACE: the channels a draw writes are (CB_TARGET_MASK &
// CB_SHADER_MASK) nibble n. We read the first to decide whether a target is
// bound and never read the second, so this puts both next to CB_COLOR_CONTROL
// and the recompiler's own export mask.
void TraceColorMasks(const Regs& regs,
                     const rhi::DrawInfo& d,
                     u64 ps_addr,
                     RecompStatus status);

// DELTA_GPU_SPRITEDUMP: transform, vertices and textures of the first few
// textured draws: degenerate MVP vs UV=0 vs blend.
void TraceSpriteDraw(const rhi::DrawInfo& d);

// DELTA_GPU_VATTRDUMP: the raw first-vertex bytes per attribute of small
// multi-attribute draws, to separate "the colours in guest memory are zero"
// from "the fetch path zeroes them".
void TraceVertexAttrs(const rhi::DrawInfo& d);

// DELTA_GPU_GEOMDUMP: high-index (world geometry) draws, with their vertices
// projected through the resolved transform to say whether the geometry is on
// screen at all.
void TraceWorldGeometry(const Regs& regs, const rhi::DrawInfo& d);

// DELTA_GPU_TRACE: the register state behind a draw, plus a one-time probe of
// the shader binaries, their user data and the descriptor tables they point at.
void TraceDrawRegisters(const Regs& regs,
                        u32 op,
                        const u32* body,
                        u32 count);

// --- compute ---------------------------------------------------------------

// DELTA_GPU_CSDUMP: the listing and user data of each distinct compute shader.
void TraceComputeShader(const Regs& regs,
                        u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 user_sgpr,
                        u32 tgid_enable,
                        u32 lds_dwords);

// DELTA_GPU_CSDROPS: dispatches refused before they could reach a capture,
// where one the guest issued and we dropped looks like one never issued.
void TraceDroppedDispatch(u64 cs_addr,
                          const u32 groups[3],
                          const u32 threads[3],
                          const char* reason);

// DELTA_GPU_CSRES=1 traces the first dispatch of each distinct shader; =<addr>
// every dispatch of that one shader, since a streaming copy runs thousands of
// times with a different destination and the first says nothing about the rest.
bool ShouldTraceCsResources(u64 cs_addr);
// DELTA_GPU_CSWATCH=<addr>: also report any resource whose range covers it.
bool CsWatchCovers(u64 base, u64 size);

void TraceCsUnresolved(u64 cs_addr, const gcn::CsResource& res);
// DELTA_GPU_EUDFAIL: raw code of an unresolvable CS, once, so its descriptor
// chain can be decoded offline.
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
void TraceCsInvalidRange(u64 cs_addr,
                         u32 binding,
                         u64 base,
                         u64 guest_size);
void TraceCsDispatch(u64 cs_addr, bool executed, u32 num_resources);

// --- the packet stream -----------------------------------------------------

// Every type-3 packet, for the opcode histogram and the per-opcode time
// DELTA_GPU_DCBSTAT reports.
void NotePacket(u32 op);
void NotePacketCost(u32 op, u64 ns);
// The timing costs two clock reads per packet, so the walk asks first.
bool WantPacketCost();
void DumpOpcodeHistogram();

// DELTA_GPU_CETRACE: every constant-engine RAM packet and whether it was
// applied. A descriptor table the CE publishes and we drop reads back as zeros,
// which looks exactly like a resolver bug.
// Whether DELTA_GPU_CETRACE is on, for callers that want to print the raw
// packet next to the decoded line.
bool CeTraceOn();

void TraceConstRam(const char* what,
                   u32 ce_offset,
                   u32 num_dwords,
                   u64 addr,
                   const char* verdict,
                   u32 first_dword);

// DELTA_GPU_CCBHIST: what the constant command buffers are made of. The
// histogram is reported after the walk it is labelled with, not before.
void NoteCcbPacket(u32 op);
void TraceCcbSubmit(u32 size_bytes, u32 words);
void TraceCcbHistogram(u32 words);

// DELTA_GPU_WAITTRACE: how many WAIT_REG_MEM polls were satisfied and how many
// timed out. A timeout means the word polled is one we write but had not yet.
void TraceWaitRegMem(bool timed_out);

// DELTA_GPU_DMATRACE: CP DMA packets by class. Shader prefetches (src == dst)
// outnumber real transfers by thousands to one, so they are counted rather than
// printed.
void TraceDmaData(u32 control,
                  u32 command,
                  u32 src_sel,
                  u32 dst_sel,
                  u64 src,
                  u64 dst,
                  u32 bytes,
                  bool copied);

// DELTA_GPU_ADDRWATCH=<addr>: name the packet that writes it. `max_lines` is
// this packet class's own budget, not a share of a common one.
void TraceAddrWatch(const char* packet,
                    u64 dst,
                    u64 bytes,
                    u32 first_dword,
                    u32 max_lines);

// DELTA_GPU_EOPTRACE: every completion label this command processor writes.
void TraceLabelWrite(const char* packet,
                     u64 addr,
                     u32 data_sel,
                     u64 value);
// EOS carries no DATA_SEL, so it reports one field fewer.
void TraceEosLabel(u64 addr, u32 value);
void TraceDataWrite(u64 addr, u32 dwords, u32 first_dword);

// DELTA_GPU_OPTRACE: PM4 opcodes nothing handles. Register state set through a
// packet we ignore reads back zero, exactly as if the guest never set it.
void TraceUnhandledOpcode(u32 op, u32 count);

// DELTA_GPU_IBTRACE: chained indirect buffers, followed or skipped.
void TraceIndirectBuffer(u32 position,
                         u32 depth,
                         u32 words,
                         bool followed);

// DELTA_GPU_COUNTERTRACE: the CE/DE counter packets.
void TraceCounter(const char* packet, u64 value);
void TraceWaitOnCeCounter(u32 wanted, u64 ce_counter);

// A type-1 header, which is a genuine desync. Reported under DELTA_GPU_DESYNC,
// or unconditionally while a submission is being dumped packet by packet.
void TraceDesync(u32 position,
                 u32 words,
                 u32 type,
                 u32 hdr,
                 bool force);

// --- submissions -----------------------------------------------------------

// Whether this submission should be dumped packet by packet: the first large
// (real rendering) command buffer under DELTA_GPU_TRACE.
bool ShouldDumpDcb(u32 size_bytes);
void TraceDcbPacket(u32 position, u32 op, u32 count);
void TraceSubmit(const void* dcb,
                 u32 size_bytes,
                 u32 words,
                 u64 submit_number,
                 u64 draws_so_far);
// After a dumped walk: how far it got, plus a brute scan for draw opcodes the
// walker may have desynced past.
void TraceDcbWalkResult(const u32* dcb,
                        u32 words,
                        u32 words_walked);
// DELTA_GPU_DCBSTAT: what the command stream is made of and which handler owns
// the time. DELTA_GPU_OPHIST: the cumulative histogram, once, deep into a run.
void TraceDcbStat(u32 words);
void MaybeDumpOpcodeHistogram(u32 words_walked, u32 words);

}  // namespace gpu::ps4
