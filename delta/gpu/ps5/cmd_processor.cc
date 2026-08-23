/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The AGC packet stream: the walk, the state it carries and the fence labels it
 * writes. See cmd_processor.h.
 */

#include "gpu/ps5/cmd_processor.h"
#include "base/arch.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>

#include <base/logging.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/agc_regs.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/draw_state.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/reg_state.h"
#include "gpu/rhi/command.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_perf.h"

namespace {
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
// Report every fence label written into the window after this address. A
// consumer stuck one count short of its target is the readable half of a
// missing submit, and the label is the only place that count exists.
DELTA_OPTION(u64, kLabelWatch, "DELTA_AGC_LABELWATCH", 0);
// How wide that window is. Narrow it to one object and the throttle below
// stops aliasing: proving a slot is NEVER written needs every write in the
// window, not a sample of them.
DELTA_OPTION(u64, kLabelWatchSize, "DELTA_AGC_LABELWATCH_SIZE", 0x4000000);
}  // namespace

// Whether a target address is a buffer the title registered for display; see
// EndFrame.
extern "C" bool prosperity_ps5_is_display_buffer(u64 addr);

namespace gpu::ps5 {
namespace {

// The register file is the state of one GPU: two submit threads walking it
// concurrently would interleave one draw's registers with another's.
std::mutex g_mutex;
// Persistent across submits: AGC programs a register once and relies on it
// holding for every later submission.
Regs g_regs;
u64 g_total_submits = 0;
bool g_renderer_started = false;
bool g_frame_active = false;

// Latched by the IT_* packets that precede a draw and consumed by it.
struct IndexState {
  u32 type = 0;  // VGT_INDEX_TYPE[1:0]: 0 = 16-bit, 1 = 32-bit, 2 = 8-bit
  u64 base = 0;  // IT_INDEX_BASE: DRAW_INDEX_OFFSET_2 has only an offset
  u32 max = 0;   // IT_INDEX_BUFFER_SIZE
  u32 num_instances = 1;  // IT_NUM_INSTANCES, for the following draw(s)
};
IndexState g_index;

// IT_SET_BASE(base_index=1) publishes where the indirect argument buffers live;
// an indirect draw or dispatch then names an offset into one. The header's
// shader-type bit picks which of the two it set.
u64 g_draw_indirect_base = 0;
u64 g_dispatch_indirect_base = 0;

// The colour target of the last draw actually issued; EndFrame presents it when
// it is a registered display buffer.
u64 g_last_draw_rt = 0;

// A cycle in the IB chain would recurse until the stack overflowed. Real
// submissions are flat or a couple of levels deep.
constexpr u32 kMaxIbDepth = 8;

// --- completion labels -----------------------------------------------------

// Labels live in guest memory the title allocated; anything plausibly mapped
// and non-null will do.
bool IsLabelAddress(u64 address) {
  return IsGuestAddress(address);
}

// EOP/RELEASE_MEM DATA_SEL 3 (GPU clock) and 4 (system clock) tell the GPU to
// write its current 64-bit clock counter into the label, NOT the packet's
// immediate data (which is 0 for these). Our submit is synchronous, so any
// advancing non-zero value reads as "already complete".
u64 GpuClockTimestamp() {
  using namespace std::chrono;
  return static_cast<u64>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count());
}

// Our submit is synchronous: every draw in the buffer is finished by the time
// the walk passes these packets, so the fence the GPU would signal is complete
// the instant we process it. Writing it immediately is what lets the guest's
// CPU-side polls make progress.
void WriteLabel(u64 address, u64 value, bool is_64bit) {
  if (!IsLabelAddress(address))
    return;
  if (is_64bit)
    *reinterpret_cast<volatile u64*>(address) = value;
  else
    *reinterpret_cast<volatile u32*>(address) = static_cast<u32>(value);
}

// INT_SEL asks the CP to raise an end-of-pipe interrupt once the write lands.
// libSceAgcDriver turns that into an event on the equeue sceAgcAddEqEvent
// registered, and a consumer that never polls its label -- the video decoder
// parks in sceKernelWaitEqueue -- makes no progress without it.
extern "C" void prosperity_gpu_end_of_pipe();
extern "C" void prosperity_gpu_end_of_pipe_ctx(u64 context_id);

// The label write shared by EOP and RELEASE_MEM, which encode DATA_SEL the same
// way: 1 = 32-bit immediate, 2 = 64-bit immediate, 3/4 = a clock counter.
void WriteEventLabel(u64 address,
                     u32 data_sel,
                     u64 value,
                     u32 int_sel = 0,
                     u64 context_id = 0) {
  if (kLabelWatch && address >= kLabelWatch &&
      address < kLabelWatch + kLabelWatchSize) {
    static int n = 0;
    // A window narrow enough to be one object is reported in full: sampling it
    // cannot show which write was the last one.
    if (kLabelWatchSize <= 0x10000 || n++ < 4096 || n % 256 == 0)
      BASE_LOGI("agclabel", "{:#x} sel={} int={} value={:#x} ctx={:#x}", address,
                data_sel, int_sel, value, context_id);
  }
  // A fence packet whose write we skip is a waiter that never wakes, so say so
  // rather than passing over it: sel 0 asks for no write at all, but an address
  // we refuse is our own gap.
  if (data_sel && !IsLabelAddress(address)) {
    static int n = 0;
    if (n++ < 16)
      BASE_LOGW("agc", "fence label {:#x} (sel {}) is not writable, skipped",
                address, data_sel);
  }
  // 5 stores GDS data, not the packet's own: the immediate words are a GDS
  // offset and size. A timestamp there puts a huge number where a fence value
  // belongs and the waiter's `label == expected` never comes true, and the
  // immediate is not the value either, so write nothing and say so.
  if (data_sel == 1)
    WriteLabel(address, value, false);
  else if (data_sel == 2)
    WriteLabel(address, value, true);
  else if (data_sel == 3 || data_sel == 4)
    WriteLabel(address, GpuClockTimestamp(), true);
  else if (data_sel == 5) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGW("agc", "fence label {:#x} asks for GDS data, unimplemented",
                address);
  }
  if (int_sel)
    prosperity_gpu_end_of_pipe_ctx(context_id ? context_id : value);
}

// --- packet handlers -------------------------------------------------------

// CP DMA. body: ctrl, srcLo/Hi, dstLo/Hi, command(byteCount). Skyrim never
// issues SET_CONTEXT_REG: it builds its context state (so CB_COLOR -- the
// render target) into a shadow image with CP DMA and then restores it with
// LOAD_CONTEXT_REG. Without the copy the shadow reads zero, every colour draw
// runs with no target bound and the frame is black. ctrl: SRC_SEL[30:29],
// DST_SEL[21:20]; sel 0/3 = memory address, 2 = immediate (a fill).
void HandleDmaData(rhi::Renderer& renderer, const u32* body, u32 count) {
  if (count < 6)
    return;
  const u32 control = body[0];
  const u32 src_sel = (control >> 29) & 0x3;
  const u32 dst_sel = (control >> 20) & 0x3;
  const u64 src = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
  const u64 dst = (static_cast<u64>(body[4] & 0xFFFF) << 32) | body[3];
  const u32 bytes = body[5] & 0x1FFFFF;
  const bool src_is_memory = src_sel == 0 || src_sel == 3;
  const bool dst_is_memory = dst_sel == 0 || dst_sel == 3;
  // The sel bits report "memory" even for GDS/register targets, which are not
  // mapped in our address space and would segfault; every real guest
  // allocation sits far above 16 MiB.
  const auto addressable = [](u64 address) {
    return address >= 0x1000000ull && address < 0x20000000000ull;
  };
  bool copied = false;
  if (!kNoCopy && src_is_memory && dst_is_memory && bytes &&
      bytes <= 0x1000000u && src != dst && addressable(src) &&
      addressable(src + bytes) && addressable(dst) &&
      addressable(dst + bytes)) {
    std::memcpy(reinterpret_cast<void*>(dst),
                reinterpret_cast<const void*>(src), bytes);
    copied = true;
  }
  // src_sel 2 = the packet's own dword, repeated: a fill. That is how a title
  // clears a surface -- there is no clear packet -- so apply it to guest memory
  // and let the renderer clear any target it covers.
  if (!kNoCopy && src_sel == 2 && dst_is_memory && bytes &&
      bytes <= 0x8000000u && addressable(dst) && addressable(dst + bytes)) {
    const u32 fill = body[1];
    u32* words = reinterpret_cast<u32*>(dst);
    for (u32 k = 0; k < bytes / 4; k++)
      words[k] = fill;
    rhi::NoteMemoryFill(renderer, dst, bytes, fill);
  }
  TraceDmaData(control, src, dst, bytes, copied);
}

// body: control, dstLo, dstHi, data...
void HandleWriteData(const u32* body, u32 count) {
  if (count < 4)
    return;
  // DST_SEL (control[11:8]) picks the destination space. 0 is the
  // memory-mapped REGISTER file, where dstLo is a register offset and not an
  // address at all; treating it as one sent the write to a pointer far below
  // the guest and silently dropped it. Titles stream state through these, so
  // the registers they carry were simply missing.
  if (((body[0] >> 8) & 0xF) == 0) {
    WriteDataRegs(g_regs, body, count);
    return;
  }
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u32 dwords = count - 3;
  if (IsLabelAddress(address) &&
      IsLabelAddress(address + static_cast<u64>(dwords) * 4))
    std::memcpy(reinterpret_cast<void*>(address), &body[3],
                static_cast<size_t>(dwords) * 4);
}

// body: eventCtrl, addrLo, addrHi+sel, dataLo, dataHi
void HandleEventWriteEop(const u32* body, u32 count) {
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u64 value = static_cast<u64>(body[3]) |
                    (static_cast<u64>(count >= 5 ? body[4] : 0) << 32);
  // INT_SEL is two bits here; RELEASE_MEM widens it to three, EOP does not.
  WriteEventLabel(address, (body[2] >> 29) & 0x7, value,
                  (body[2] >> 24) & 0x3);
}

// body: eventCtrl, selBits, addrLo, addrHi, dataLo, dataHi
void HandleReleaseMem(const u32* body, u32 count) {
  if (count < 5)
    return;
  const u64 address =
      (static_cast<u64>(body[3] & 0xFFFF) << 32) | (body[2] & ~0x3u);
  const u64 value = static_cast<u64>(body[4]) |
                    (static_cast<u64>(count >= 6 ? body[5] : 0) << 32);
  // DW6 is INT_CTXID: the id the title tags its submit with and matches the
  // interrupt against.
  WriteEventLabel(address, (body[1] >> 29) & 0x7, value,
                  (body[1] >> 24) & 0x7, count >= 7 ? body[6] : 0);
}

// body: eventCtrl, addrLo, addrHi+cmd, data
void HandleEventWriteEos(const u32* body, u32 count) {
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  WriteLabel(address, body[3], false);
}

// IT_COPY_DATA: one small copy between memory, a register and the GPU's own
// clocks. gfx10 widens both selectors by a bit from the header's top, so they
// are read as (field << 1) | bit30 and the useful values double: 2/4/5 name
// memory, 3/6/7 a register, 10/11 an immediate in the packet, and src 9 / 18
// are the GPU and system reference clocks. A title reads the clock through
// this packet and polls the result, so dropping it is a wait that never ends.
void HandleCopyData(const u32* body, u32 count) {
  if (count < 5)
    return;
  const u32 control = body[0];
  const u32 src_sel = ((control & 0xF) << 1) | ((control >> 30) & 0x1);
  const u32 dst_sel = ((control >> 8) & 0xF) << 1;
  const u32 bytes = ((control >> 16) & 0x1) ? 8u : 4u;
  const u64 src = body[1] | (static_cast<u64>(body[2]) << 32);
  const u64 dst = body[3] | (static_cast<u64>(body[4]) << 32);
  const auto is_memory = [](u32 sel) {
    return sel == 2 || sel == 4 || sel == 5;
  };
  if (!is_memory(dst_sel) || !dst || (dst & (bytes - 1)) ||
      !IsLabelAddress(dst) || !gpu::IsReadableRange(dst, bytes))
    return;
  u64 value = 0;
  if (src_sel == 9 || src_sel == 18) {         // GPU clock / system clock
    value = GpuClockTimestamp();
  } else if (src_sel == 10 || src_sel == 11) { // immediate, in the packet
    value = src;
  } else if (is_memory(src_sel)) {
    if (!IsLabelAddress(src) || !gpu::IsReadableRange(src, bytes))
      return;
    std::memcpy(&value, reinterpret_cast<const void*>(src), bytes);
  } else {
    return;                                    // a register we do not model
  }
  if (bytes == 8)
    *reinterpret_cast<volatile u64*>(dst) = value;
  else
    *reinterpret_cast<volatile u32*>(dst) = static_cast<u32>(value);
}

// IT_EVENT_WRITE carrying ZPASS_DONE (event type 0x39, index 1) is an occlusion
// query: the packet names a result buffer and the hardware writes one
// begin/end pair per depth block into it, with bit 63 set once a value is
// ready. We have no host query, and a title that waits for that bit waits
// forever -- Astro Bot's boot stops on the first frame it draws, its main
// thread parked on a semaphore nothing posts. Publish an always-visible count
// instead: nothing culls, which is wrong but visible, and the ready bit is
// what the wait is actually looking for.
void HandleEventWrite(const u32* body, u32 count) {
  if (count < 3)
    return;
  const u32 event_type = body[0] & 0x3F;
  const u32 event_index = (body[0] >> 8) & 0x7;
  if (event_type != 0x39 || event_index != 1)
    return;
  const u64 address = body[1] | (static_cast<u64>(body[2]) << 32);
  constexpr u32 kBlocks = 16;              // one begin/end pair per DB
  constexpr u64 kBytes = kBlocks * 2 * sizeof(u64);
  if (!address || (address & 7) || !IsLabelAddress(address) ||
      !gpu::IsReadableRange(address, kBytes))
    return;
  constexpr u64 kReady = 1ull << 63;
  // The packet address selects the column of a one-pair-per-depth-block
  // record: begin at the base, end at +8, 16 bytes per DB. The title computes
  // end - begin, so a fixed pair of values reads as zero visible and it culls
  // everything (fixed end values dropped the world map entirely; contiguous
  // qwords hung the boot). The counter advances one sample per packet, so
  // every end exceeds its begin.
  static u64 samples = 0;
  const u64 value = kReady | (samples & (kReady - 1));
  auto* results = reinterpret_cast<volatile u64*>(address);
  for (u32 db = 0; db < kBlocks; db++)
    results[db * 2] = value;
  samples++;
  TraceOcclusionQuery(address, value);
}

// IT_DISPATCH_INDIRECT: the workgroup counts live in memory rather than in the
// packet. Our submits run synchronously, so whatever wrote them has already
// run and the counts are readable now.
void HandleDispatchIndirect(rhi::Renderer& renderer,
                            const u32* body,
                            u32 count) {
  u64 args = 0;
  if (count >= 3)  // pointer form: addrLo, addrHi, mode
    args = body[0] | (static_cast<u64>(body[1]) << 32);
  else if (count >= 1)  // offset form: an offset into the SET_BASE buffer
    args = g_dispatch_indirect_base + body[0];
  if (!args || !IsGuestAddress(args) || !gpu::IsReadableRange(args, 12))
    return;
  const u32* a = reinterpret_cast<const u32*>(args);
  const u32 groups[3] = {a[0], a[1], a[2]};
  if (!groups[0] || !groups[1] || !groups[2])
    return;
  DispatchCompute(renderer, g_regs, groups, 3);
}

// IT_DRAW_INDIRECT / IT_DRAW_INDEX_INDIRECT: same as the direct forms with the
// counts read from the argument buffer. Rebuilding the direct packet keeps one
// draw path rather than a second one that would drift from it.
void HandleDrawIndirect(rhi::Renderer& renderer,
                        u32 op,
                        const u32* body,
                        u32 count,
                        void (*issue)(rhi::Renderer&, u32, const u32*, u32)) {
  if (count < 1)
    return;
  const bool indexed = op == 0x25;
  const u64 args = g_draw_indirect_base + body[0];
  const u32 want = indexed ? 20u : 16u;
  if (!g_draw_indirect_base || !IsGuestAddress(args) ||
      !gpu::IsReadableRange(args, want))
    return;
  const u32* a = reinterpret_cast<const u32*>(args);
  const u32 initiator = count >= 4 ? body[3] : 0;
  const u32 saved_instances = g_index.num_instances;
  if (a[1])
    g_index.num_instances = a[1];
  if (!indexed) {
    const u32 auto_body[2] = {a[0], initiator};
    issue(renderer, IT_DRAW_INDEX_AUTO, auto_body, 2);
  } else {
    const u32 stride = g_index.type == 1 ? 4u : g_index.type == 2 ? 1u : 2u;
    const u64 base = g_index.base + static_cast<u64>(a[2]) * stride;
    const u32 idx_body[5] = {g_index.max ? g_index.max : a[0],
                             static_cast<u32>(base),
                             static_cast<u32>(base >> 32), a[0], initiator};
    issue(renderer, IT_DRAW_INDEX_2, idx_body, 5);
  }
  g_index.num_instances = saved_instances;
}

void HandleDrawPacket(rhi::Renderer& renderer,
                      u32 op,
                      const u32* body,
                      u32 count) {
  NoteDrawSeen();
  if (!renderer.available())
    return;
  DrawPacket packet;
  packet.op = op;
  packet.body = body;
  packet.count = count;
  packet.index_type = g_index.type;
  packet.index_base = g_index.base;
  packet.index_max = g_index.max;
  packet.num_instances = g_index.num_instances;

  rhi::DrawInfo d;
  if (!BuildDrawInfo(g_regs, packet, d))
    return;
  if (!g_frame_active) {
    TraceBeginFrame();
    rhi::BeginFrame(renderer);
    g_frame_active = true;
  }
  TraceDrawSubmit(d);
  NoteDrawIssued(d);
  g_last_draw_rt = d.rt_base;
  rhi::Draw(renderer, d);
  TraceDrawDone();
}

bool IsDraw(u32 op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDEX_MULTI_AUTO;
}

// --- the walk --------------------------------------------------------------

// Walk one AGC stream, following INDIRECT_BUFFER, latching registers, decoding
// draws and writing completion labels. `depth` guards a malformed
// self-reference.
void Walk(rhi::Renderer& renderer,
          const u32* p,
          u32 words,
          bool dump,
          u32 depth) {
  if (!p || depth > kMaxIbDepth)
    return;
  u32 i = 0;
  // The last few packet starts, so a desync can name the packet whose size was
  // wrong instead of the data it eventually ran into. A type-0 header consumes
  // its run silently, so a walk that is already off can travel a long way
  // before anything complains.
  u32 trail_pos[32] = {}, trail_hdr[32] = {};
  u32 trail_n = 0;
  while (i < words) {
    const u32 hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    trail_pos[trail_n & 31] = i;
    trail_hdr[trail_n & 31] = hdr;
    trail_n++;
    if (type == Pm4Type::kType2 || hdr == 0) {
      i += 1;  // filler / alignment
      continue;
    }
    if (type == Pm4Type::kType0) {
      // Type-0 writes a run of consecutive registers directly. The walker used
      // to SKIP these -- but the AGC driver programs shader PGM_LO/HI (and
      // other SH state) via type-0, which is why no SET_SH_REG carried them.
      // A type-0 header is only two zero bits, so any data dword looks like
      // one: a walk that has already lost alignment travels thousands of
      // dwords through "type-0 runs" without a word of complaint, which is how
      // a desync stays invisible until it lands on data that happens to be
      // type-1. Bound it to the register file it claims to write.
      const u32 count = Pm4Count(hdr);
      const u32 first = Pm4Type0Reg(hdr);
      if (first + count > kRegFileSize || i + 1 + count > words) {
        i += 1;
        continue;
      }
      SetRegRun(g_regs, first, &p[i + 1], count);
      i += 1 + count;
      continue;
    }
    if (type != Pm4Type::kType3) {
      // A type-1 header is a desync, and abandoning the buffer costs the fence
      // the submission still had to write, which parks whatever waits on it
      // forever. Skip the dword and resync on the next header, exactly as a
      // type-3 packet whose count overruns the buffer already does.
      static int n = 0;
      if (n++ < 4) {
        // The dwords either side of the desync, so the packet that mis-sized
        // itself can be identified rather than guessed at.
        base::String line;
        base::FormatTo(line, "type-{} header {:#x} at dword {}/{} of {:p}, "
                             "resyncing; packets in:",
                       static_cast<u32>(type), hdr, i, words, (const void*)p);
        const u32 seen = trail_n < 32 ? trail_n : 32;
        for (u32 k = 0; k < seen; k++) {
          const u32 s = (trail_n - seen + k) & 31;
          base::FormatTo(line, " [{}]{:08x}", trail_pos[s], trail_hdr[s]);
        }
        // Plus the stream just before the oldest of those, which is where the
        // walk was still aligned.
        const u32 oldest = trail_pos[(trail_n - seen) & 31];
        const u32 from = oldest > 24 ? oldest - 24 : 0;
        base::FormatTo(line, " | dwords from {}:", from);
        for (u32 k = from; k < words && k < oldest + 4; k++)
          base::FormatTo(line, " {:08x}", p[k]);
        BASE_LOGW("agc", "{}", line.c_str());
      }
      i += 1;
      continue;
    }

    const u32 op = Pm4Opcode(hdr);
    const u32 count = Pm4Count(hdr);  // body dword count
    const u32* body = &p[i + 1];
    // 0xffff1000: IT_NOP with the count field saturated is the canonical
    // one-dword pad, and a command buffer is padded with it. Reading it as a
    // 16385-dword packet skipped 64 KiB of real commands -- for Astro Bot's
    // world map that was the whole tail of the buffer, RELEASE_MEM included,
    // so its DrawThread waited on a fence nothing was left to write.
    if (op == IT_NOP && ((hdr >> 16) & 0x3FFF) == 0x3FFF) {
      i += 1;
      continue;
    }
    // Desync recovery: a data dword misread as a huge-count packet would
    // abandon the rest of the buffer -- and with it the shader bind that
    // follows. Instead of bailing, skip one dword and resync on the next
    // header.
    if (i + 1 + count > words) {
      TraceResync(i, words, hdr, op, count);
      i += 1;
      continue;
    }
    NoteOpcode(op);
    TraceOpcodeBody(op, body, count);
    TraceColorBaseSource(op, g_regs[mmCB_COLOR0_BASE]);
    if (dump)
      TraceDcbPacket(i, op, body, count);
    switch (op) {
      case IT_INDIRECT_BUFFER:  // baseLo, baseHi, sizeDwords(+flags)
      case IT_INDIRECT_BUFFER_CNST: {  // the AGC constant/Cue chain, which
                                       // carries the pipeline shader setup
        if (count < 3)
          break;
        const u64 ib = (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
        const u32 ib_words = body[2] & 0xFFFFF;
        // Bounds-guard: a sane size, in the guest map, actually readable. The
        // GPU aperture is deliberately NOT required -- the video decoder builds
        // its command buffers in its own allocation (0x6_0000_0000 for Astro
        // Bot, well under the aperture floor), and skipping those left its
        // completion fence one submit short forever, so the decode thread spun
        // holding the lock the whole player waits on.
        const bool follow =
            ib_words && ib_words <= 0x40000 && IsGuestAddress(ib) &&
            gpu::IsReadableRange(ib, static_cast<u64>(ib_words) * sizeof(u32));
        TraceIndirectBuffer(ib, ib_words, follow);
        if (follow) {
          Walk(renderer, reinterpret_cast<const u32*>(ib), ib_words, dump,
               depth + 1);
        } else {
          static int n = 0;
          if (n++ < 16)
            BASE_LOGW("agc", "IB {:#x}+{:#x} dwords not followed", (u64)ib,
                      ib_words);
        }
        break;
      }
      case IT_SET_CONTEXT_REG:
        SetRegs(g_regs, kContextRegBase, body, count);
        break;
      case IT_SET_SH_REG:
        SetRegs(g_regs, kShRegBase, body, count);
        break;
      case IT_SET_UCONFIG_REG:
        SetRegs(g_regs, kUConfigRegBase, body, count);
        break;
      case IT_SET_CONFIG_REG:
        SetRegs(g_regs, kConfigRegBase, body, count);
        break;
      // LOAD_*_REG: registers loaded from a shadow image in GPU memory.
      case 0x61:  // LOAD_CONTEXT_REG
        LoadRegImage(g_regs, kContextRegBase, body, count);
        break;
      case 0x5f:  // LOAD_SH_REG
        LoadRegImage(g_regs, kShRegBase, body, count);
        break;
      case 0x5e:  // LOAD_UCONFIG_REG
        LoadRegImage(g_regs, kUConfigRegBase, body, count);
        break;
      // SET_*_REG_INDIRECT: a state block of (offset, value) entries, which is
      // how the per-draw render target and shaders arrive.
      case 0x9f:
        LoadRegBlock(g_regs, kContextRegBase, body, count);
        break;
      case 0x64:
        LoadRegBlock(g_regs, kUConfigRegBase, body, count);
        break;
      case 0x63:
        LoadRegBlock(g_regs, kShRegBase, body, count);
        break;
      // SET_UCONFIG_REG_INDEX: the index lives in the offset dword's top bits
      // and SetRegs already masks the selector off, so the register write is
      // the same one. Dropping these lost the index type and primitive type a
      // title programs through this form.
      case 0x7a:
        SetRegs(g_regs, kUConfigRegBase, body, count);
        break;
      // WAIT_REG_MEM_64: wait until a 64-bit value in memory satisfies a
      // comparison. Our submit is synchronous, so the condition is met by the
      // time we walk the packet. It is NOT a register write: reading it as one
      // stored the poll address over the shader user-data registers, which is
      // why the draws that followed sampled from nothing.
      case 0x93:
        break;
      case IT_DMA_DATA:
        HandleDmaData(renderer, body, count);
        break;
      case IT_INDEX_TYPE:
        if (count >= 1)
          g_index.type = body[0] & 0x3;
        break;
      case IT_INDEX_BASE:  // baseLo, baseHi
        if (count >= 2)
          g_index.base =
              (static_cast<u64>(body[1] & 0xFFFF) << 32) | (body[0] & ~1u);
        break;
      case IT_INDEX_BUFFER_SIZE:
        if (count >= 1)
          g_index.max = body[0];
        break;
      case IT_NUM_INSTANCES:
        g_index.num_instances = (count >= 1 && body[0]) ? body[0] : 1;
        break;
      case IT_DISPATCH_DIRECT:
        DispatchCompute(renderer, g_regs, body, count);
        break;
      case 0x16:  // DISPATCH_INDIRECT
        HandleDispatchIndirect(renderer, body, count);
        break;
      case 0x24:  // DRAW_INDIRECT
      case 0x25:  // DRAW_INDEX_INDIRECT
        HandleDrawIndirect(renderer, op, body, count, HandleDrawPacket);
        break;
      case 0x11: {  // SET_BASE
        if (count < 3 || (body[0] & 0xF) != 1)
          break;
        const u64 base = (body[1] & ~0x7ull) |
                         (static_cast<u64>(body[2] & 0xFFFF) << 32);
        if ((hdr >> 1) & 0x3)
          g_dispatch_indirect_base = base;
        else
          g_draw_indirect_base = base;
        break;
      }
      case IT_COND_EXEC: {  // skip the following dwords when the flag is zero
        if (count < 4)
          break;
        const u64 flag = (static_cast<u64>(body[1] & 0xFFFF) << 32) |
                         (body[0] & 0xFFFFFFFCu);
        const u32 skip = body[3] & 0x3FFF;
        if (IsGuestAddress(flag) && gpu::IsReadableRange(flag, 4) &&
            *reinterpret_cast<const volatile u32*>(flag) == 0)
          i += skip;  // the guarded block did not run on the real CP either
        break;
      }
      case IT_WRITE_DATA:
        HandleWriteData(body, count);
        break;
      case IT_COPY_DATA:
        HandleCopyData(body, count);
        break;
      case IT_EVENT_WRITE:
        HandleEventWrite(body, count);
        break;
      case IT_EVENT_WRITE_EOP:
        HandleEventWriteEop(body, count);
        break;
      case IT_RELEASE_MEM:
        HandleReleaseMem(body, count);
        break;
      case IT_EVENT_WRITE_EOS:
        HandleEventWriteEos(body, count);
        break;
      default:
        if (IsDraw(op))
          HandleDrawPacket(renderer, op, body, count);
        break;
    }
    i += 1 + count;
  }
}

// The renderer comes up on the first submission rather than at startup: a title
// that never submits never needs a device.
void StartRendererOnce(rhi::Renderer& renderer) {
  if (g_renderer_started)
    return;
  g_renderer_started = true;
  rhi::Init(renderer);
  // The descriptor replay reads SRT tables out of guest memory a previous
  // dispatch may still own, and it sits below the renderer, so it cannot ask
  // for the flush itself.
  gcn::g_flush_guest_range = [](u64 address, u64 bytes) {
    rhi::FlushCsWritesRange(rhi::DefaultRenderer(), address, bytes);
  };
}

}  // namespace

void SubmitDcb(const void* dcb, u32 size_bytes) {
  if (!dcb || size_bytes < 4)
    return;
  const u64 t_enter = vk::NowNs();
  std::lock_guard<std::mutex> lock(g_mutex);
  const u64 t_held = vk::NowNs();
  rhi::g_ns_dcb_lock += t_held - t_enter;
  rhi::Renderer& renderer = rhi::DefaultRenderer();
  StartRendererOnce(renderer);
  TraceRegShadowScan();

  const u32 words = size_bytes / 4;
  const u64 submission = ++g_total_submits;
  const bool dump = TraceSubmit(dcb, size_bytes, words, submission);
  Walk(renderer, static_cast<const u32*>(dcb), words, dump, 0);
  TraceOpcodeCensus(dcb, words, submission);
  if (dump)
    TraceWalkDone();
  rhi::g_ns_dcb += vk::NowNs() - t_held;
  rhi::g_dcb_n++;
}

void SubmitCcb(const void* ccb, u32 size_bytes) {
  if (!ccb || size_bytes < 4)
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  Walk(rhi::DefaultRenderer(), static_cast<const u32*>(ccb), size_bytes / 4,
       false, 0);
}

void EndFrame(u64 scanout_base) {
  std::lock_guard<std::mutex> lock(g_mutex);
  // New frame -> shader code may have been rewritten; let the cached programs
  // revalidate each address once next frame instead of once per draw. Before
  // the early return, or a frame that ends with nothing active never advances
  // it and the cache stops revalidating at all.
  rdna::NextProgramGeneration();
  gpu::NextMemoryGeneration();
  rhi::Renderer& renderer = rhi::DefaultRenderer();
  if (!g_frame_active || !renderer.available())
    return;
  // The trigger that ends a frame is often the NEXT frame's state submit, and
  // the flip it reads still names the buffer before this one. When the frame
  // composited straight into a registered display buffer, that buffer is what
  // the title just finished, so present it instead -- otherwise every frame
  // presents its neighbour, which the title has already cleared.
  if (g_last_draw_rt && g_last_draw_rt != scanout_base &&
      prosperity_ps5_is_display_buffer(g_last_draw_rt))
    scanout_base = g_last_draw_rt;
  rhi::EndFrame(renderer, scanout_base);
  g_frame_active = false;
}

}  // namespace gpu::ps5

// LLE submit bridge: the kernel /dev/gc AGC ioctls (gc_dev.cpp) forward the DCB
// here, mirroring prosperity_gc_submit on the PS4 path.
extern "C" void prosperity_agc_submit(u64 dcb_base, u32 size_bytes) {
  gpu::ps5::SubmitDcb(reinterpret_cast<const void*>(dcb_base), size_bytes);
}

// PS5 flip bridge: the shared dce/VideoOut flip path calls this when the active
// process is PS5, so the frame the AGC submit rendered is read back and
// presented through rhi::EndFrame (mirrors prosperity_gc_flip on the PS4 path).
extern "C" void prosperity_agc_flip(u64 scanout_base) {
  gpu::ps5::EndFrame(scanout_base);
}

// GPU aperture bridge: the kernel tells us where the title mapped its direct
// memory, so a packet naming a pool outside the assumed band is still followed.
extern "C" void prosperity_gpu_note_aperture(u64 base, u64 size) {
  gpu::ps5::NoteGpuPool(base, size);
}

extern "C" int prosperity_gpu_is_aperture(u64 address) {
  return gpu::ps5::IsGpuAddress(address) ? 1 : 0;
}
