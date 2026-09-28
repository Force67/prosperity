/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PM4 packet stream: the DE and CE walks and the state they carry. See
 * cmd_processor.h.
 */

#include "gpu/ps4/cmd_processor.h"
#include "base/arch.h"
#include "base/logging.h"
#include "gpu/ps4/render_queue.h"
#include "gpu/write_tracker.h"

#include <cstring>

#include "host_memory/host_memory.h"
#include "options/options.h"

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/set.h"
#include "base/memory/move.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/compute_dispatch.h"
#include "gpu/ps4/draw_state.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/ps4/pm4.h"
#include "gpu/render/renderer.h"

namespace {
DELTA_OPTION(bool, kCeOn, "DELTA_GPU_CE", true);
// Log every memory WAIT_REG_MEM from this many seconds into the run: where in
// which buffer, on what word, and what the word held when the walk got there.
DELTA_OPTION(u32, kWaitTraceFrom, "DELTA_GPU_WAITTRACE", 0);
// Record draws on a renderer thread of their own (see render_queue.h).
DELTA_OPTION(bool, kRenderThread, "DELTA_GPU_RENDER_THREAD", true);
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
DELTA_OPTION(bool, kFenceFlush, "DELTA_GPU_CS_FENCE_FLUSH", true);
DELTA_OPTION(bool, kPrefetchShaders, "DELTA_GPU_PREFETCH_SHADERS", true);
}  // namespace

namespace gpu::render {
// Declared in render/command.h; the frame-time overlay reports them.
u64 g_ns_dcb = 0;
u32 g_submit_queue = 0;
u64 g_ns_dcb_lock = 0;
u32 g_dcb_n = 0;
}  // namespace gpu::render

namespace gpu::ps4 {
namespace {

// The register file is the state of one GPU: two submit threads walking it
// concurrently would interleave one draw's registers with another's.
base::Mutex g_mutex;
// Persistent across submits: Gnm programs a register once and relies on it
// holding for every later submission.
Regs g_regs;
base::Atomic<u64> g_total_submits{0};
base::Atomic<u64> g_total_draws{0};
bool g_renderer_started = false;
bool g_frame_active = false;
u32 g_presented_frames = 0;

// Latched by the IT_* packets that precede a draw and consumed by it.
struct IndexState {
  u32 type = 0;           // VGT_DMA_INDEX_TYPE[1:0]: 0 = 16-bit, 1 = 32-bit
  u64 base = 0;           // IT_INDEX_BASE (DRAW_INDEX_2 carries its own)
  u64 indirect_base = 0;  // IT_SET_BASE(1): where indirect args live
  u32 num_instances = 1;  // IT_NUM_INSTANCES, for the following draw(s)
};
IndexState g_index;

// CE/DE synchronization counters (24-bit). The DE's WAIT_ON_CE_COUNTER runs
// the CE until it is ahead; the CE's WAIT_ON_DE_COUNTER_DIFF pauses it until
// the DE catches up.
u64 g_ce_counter = 0;
u64 g_de_counter = 0;

// A cycle in the IB chain would recurse until the stack overflowed. Real
// submissions are flat or a couple of levels deep.
constexpr u32 kMaxIbDepth = 8;

// --- completion labels -----------------------------------------------------

// The guest words this command processor writes: the EOP / EOS / RELEASE_MEM /
// WRITE_DATA fence labels a title's CPU threads and other rings poll to order
// themselves after this one. A WAIT_REG_MEM on anything else is waiting for a
// producer we do not run, and spinning on that buys nothing.
// Titles that allocate labels per frame (Uncharted 2) keep producing new
// addresses, so the set forgets the oldest generation rather than filling up:
// a full set turned every later wait on a label of ours into a skipped one.
class FenceLabels {
 public:
  void Note(u64 address) {
    if (!address)
      return;
    base::LockGuard<base::Mutex> lock(mutex_);
    if (current_.size() >= kGeneration) {
      previous_ = base::move(current_);
      current_ = {};
    }
    current_.insert(address & ~3ull);
  }
  bool Contains(u64 address) const {
    base::LockGuard<base::Mutex> lock(mutex_);
    const u64 key = address & ~3ull;
    return current_.count(key) != 0 || previous_.count(key) != 0;
  }

 private:
  static constexpr size_t kGeneration = 65536;
  mutable base::Mutex mutex_;
  base::HashSet<u64> current_, previous_;
};
FenceLabels g_fence_labels;

// Labels live in guest memory the game allocated (Garlic/Onion), in low guest
// heaps, or in the GnmDriver area. Accept any plausibly-mapped, non-low
// address; reject null/garbage.
bool IsLabelAddress(u64 address) {
  return address >= 0x10000ull && address < kGuestEnd;
}

// A GPU write to unmapped memory faults the GPU, not the CPU: drop it rather
// than let the emulator segfault, and report the first few so a packet parsed
// from the wrong bytes shows up.
bool GpuWriteMapped(const char* packet, u64 address, u64 bytes) {
  if (host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(address),
                                       bytes))
    return true;
  static base::Atomic<u32> reported{0};
  if (reported.fetch_add(1) < 16)
    BASE_LOGW("gpu", "{} writes unmapped {:#x}+{:#x}, dropped", packet,
              (unsigned long long)address, (unsigned long long)bytes);
  return false;
}

// Our submit is synchronous: every draw in the buffer is finished by the time
// the walk passes these packets, so the fence the GPU would signal is complete
// the instant we process it. Writing it immediately is what lets the guest's
// CPU-side polls (the flip-done / submit-done labels Gnm spins on between
// frames) make progress. Without it the title stalls once the few in-flight
// display buffers drain.
void NoteGpuWriteNear(const char* packet, u64 address, u64 value);

void WriteLabel(u64 address, u64 value, bool is_64bit) {
  NoteGpuWriteNear("label", address, value);

  g_fence_labels.Note(address);
  if (!IsLabelAddress(address) ||
      !GpuWriteMapped("label", address, is_64bit ? 8 : 4))
    return;
  if (is_64bit)
    *reinterpret_cast<volatile u64*>(address) = value;
  else
    *reinterpret_cast<volatile u32*>(address) = static_cast<u32>(value);
}

// EOP/RELEASE_MEM DATA_SEL 3 (GPU clock) and 4 (system clock) tell the GPU to
// write its current 64-bit clock counter into the label, NOT the packet's
// immediate data (which is 0 for these). A title polling such a label for
// "non-zero == the GPU reached this point" needs a real, monotonically
// increasing, non-zero value; our submit is synchronous, so any advancing clock
// reads as "already complete". Without this Doom64's per-frame submit-done wait
// (a spin with a hard 2s timeout) burns the full 2s every frame -> ~0.5 fps.
u64 GpuClockTimestamp() {
  return static_cast<u64>((base::TickClock::NowNs()));
}

// INT_SEL asks the CP to raise an end-of-pipe interrupt once the write lands
// (1 = on write confirm, 2/3 = with the data). libSceGnmDriver turns that into
// the graphics-core equeue event a title's fence bookkeeping runs off, so the
// label write alone is only half the packet. GTA:SA's async-compute ring is
// RELEASE_MEM with INT_SEL=3 throughout.
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe();

// The label write shared by EOP and RELEASE_MEM, which encode DATA_SEL the same
// way: 1 = 32-bit immediate, 2 = 64-bit immediate, 3/4 = a clock counter.
void WriteEventLabel(const char* packet,
                     u64 address,
                     u32 data_sel,
                     u64 value,
                     u32 int_sel = 0) {
  // The guest CPU reads a dispatch's output once it sees a fence behind it:
  // those results go to guest memory before the fence does.
  if (data_sel && kFenceFlush && render::CsTracksGuestWrites())
    render::FlushCsWrites(render::DefaultRenderer());
  if (data_sel == 1)
    WriteLabel(address, value, false);
  else if (data_sel == 2)
    WriteLabel(address, value, true);
  else if (data_sel >= 3)
    WriteLabel(address, GpuClockTimestamp(), true);
  TraceLabelWrite(packet, address, data_sel, value);
  if (int_sel)
    // NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
    prosperity_gpu_end_of_pipe();
}

// --- constant engine RAM ---------------------------------------------------

// Liverpool's Constant Engine RAM: 48 KiB of on-chip scratch the CE fills and
// dumps to guest memory as the shaders' constant buffers. Every access is
// bounds checked here so a malformed packet can never write outside it.
class ConstRam {
 public:
  bool Fits(u32 offset, u32 dwords) const {
    return (u64)offset + (u64)dwords * 4 <= sizeof(data_);
  }
  void Write(u32 offset, const void* src, u32 dwords) {
    std::memcpy(data_ + offset, src, (size_t)dwords * 4);
  }
  void Read(u32 offset, void* dst, u32 dwords) const {
    std::memcpy(dst, data_ + offset, (size_t)dwords * 4);
  }
  u32 DwordAt(u32 offset) const {
    u32 value;
    std::memcpy(&value, data_ + offset, sizeof(value));
    return value;
  }

 private:
  u8 data_[48 * 1024] = {};
};
ConstRam g_const_ram;

// --- register writes -------------------------------------------------------

void SetRegs(u32 base, const u32* body, u32 count) {
  if (!count)
    return;
  // Indexed SET packets use bits 28..31 for the index; only the low 16 bits are
  // the register offset. Treating the whole word as an offset drops Neo
  // register writes such as 0x40000258.
  const u32 first = Pm4SetRegAddress(base, body[0]);
  const u32 values = count - 1;
  for (u32 i = 0; i < values; i++)
    if (first + i < kRegFileSize)
      g_regs[first + i] = body[1 + i];
  NoteRegisterWrites(first, &body[1], values, base);
}

// --- packet handlers -------------------------------------------------------

// The guest's only way to make this ring wait for another engine: poll a word
// (or a register) until it satisfies a comparison. P.T. issues 86,611 of them a
// run and every one was ignored, so with the async compute rings walked on
// their own thread a draw could run before the dispatch that produced what it
// samples.
//
// body: [0] function/space, [1] addr lo or reg offset, [2] addr hi,
// [3] reference, [4] mask, [5] poll interval.
bool WaitRegMemPasses(const u32* body, u32 polled) {
  const u32 function = body[0] & 0x7;
  const u32 a = polled & body[4], b = body[3] & body[4];
  switch (function) {
    case 1:
      return a < b;
    case 2:
      return a <= b;
    case 3:
      return a == b;
    case 4:
      return a != b;
    case 5:
      return a >= b;
    case 6:
      return a > b;
    default:
      return true;  // 0 = always, 7 = reserved
  }
}

// The word a WAIT_REG_MEM polls, or null when it waits on nothing we model.
const volatile u32* WaitRegMemTarget(const u32* body) {
  if ((body[0] >> 4) & 1) {
    const u64 address =
        ((static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
    // Only where we are the producer. A poll on a word nothing of ours writes
    // can never be satisfied, and waiting out its timeout is pure loss.
    if (IsGuestAddress(address) && g_fence_labels.Contains(address) &&
        host_memory::IsMemoryRangeMapped(reinterpret_cast<const void*>(address),
                                         4))
      return reinterpret_cast<const volatile u32*>(address);
    return nullptr;
  }
  if ((body[1] & 0xFFFF) < kRegFileSize)
    return &g_regs[body[1] & 0xFFFF];
  return nullptr;
}

// DELTA_GPU_WAITWATCH: a poll on a word nothing of ours writes, still
// unsatisfied, hands the word to this hook (a write watch) so the run names
// whoever does write it. The first few distinct words only.
void (*g_unknown_wait_hook)(u64 address) = nullptr;
// Pages of the words handed to the hook: a write of ours landing next to one
// (a label decoded at the wrong address) is the other half of the question.
base::Atomic<u64> g_watched_pages[4] = {~0ull, ~0ull, ~0ull, ~0ull};

void NoteGpuWriteNear(const char* packet, u64 address, u64 value) {
  if (!g_unknown_wait_hook)
    return;
  for (auto& page : g_watched_pages)
    if (page.load(base::memory_order_relaxed) == (address & ~0xFFFull)) {
      BASE_LOGI("waitwatch", "{} writes {:#x} = {:#x} on a watched page",
                packet, (unsigned long long)address,
                (unsigned long long)value);
      return;
    }
}

void NoteUnknownWait(const u32* body) {
  if (!g_unknown_wait_hook || !((body[0] >> 4) & 1))
    return;
  const u64 address =
      ((static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
  if (!IsGuestAddress(address) ||
      !host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(address), 4))
    return;
  const u32 value = *reinterpret_cast<const volatile u32*>(address);
  if (WaitRegMemPasses(body, value))
    return;
  // One new watch every few seconds: the first unknown words of a run are
  // rarely the ones behind a later stall.
  static u32 watched = 0;
  static base::TimeTicks last;
  const base::TimeTicks now = base::TimeTicks::Now();
  if (watched >= 20 || (watched && now - last < base::Seconds(5)))
    return;
  g_watched_pages[watched % 4].store(address & ~0xFFFull);
  watched++;
  last = now;
  BASE_LOGI("waitwatch",
            "WAIT_REG_MEM on unknown word {:#x} (fn={} ref={:#x} mask={:#x} "
            "now {:#x}); watching its writer",
            (unsigned long long)address, body[0] & 7, body[3], body[4], value);
  g_unknown_wait_hook(address);
}

void HandleWaitRegMem(const u32* body, u32 count) {
  if (count < 5)
    return;
  const volatile u32* polled = WaitRegMemTarget(body);
  const auto passes = [&](u32 value) { return WaitRegMemPasses(body, value); };
  if (!polled) {
    NoteUnknownWait(body);
    return;
  }

  // Bounded, because a producer can still be one we dropped and an unbounded
  // poll would hang the title outright. Yield rather than spin: the thread that
  // will satisfy this needs the core.
  const auto deadline = base::TimeTicks::Now() + base::Microseconds(200);
  bool timed_out = false;
  while (!passes(*polled)) {
    if (base::TimeTicks::Now() >= deadline) {
      timed_out = true;
      break;
    }
    base::YieldCurrentThread();
  }
  TraceWaitRegMem(timed_out);
}

// CP DMA. body: ctrl, srcLo/Hi, dstLo/Hi, command(byteCount). Doom64 uploads
// its level texture atlases this way, so without performing the copy the T#
// addresses stay zero and the 3D world samples blank textures. ctrl word:
// SRC_SEL[30:29], DST_SEL[21:20]; sel 0/3 = memory address, 2 = immediate data
// (a fill, not a copy); only true mem->mem is copied.
void HandleDmaData(render::Renderer& renderer, const u32* body, u32 count) {
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
  // Only copy between REAL guest memory: the sel bits report "memory" even for
  // GDS/register targets (e.g. dst=0x3022c), which are not mapped in our
  // address space and segfault. Every real guest allocation sits far above
  // 16 MiB, so that floor excludes the on-chip GDS/low targets while keeping
  // texture/buffer uploads.
  const auto addressable = [](u64 address) {
    return address >= 0x1000000ull && address < kGuestEnd;
  };
  bool copied = false;
  if (!kNoCopy && src_is_memory && dst_is_memory && bytes &&
      bytes <= 0x1000000u && src != dst && addressable(src) &&
      addressable(src + bytes) && addressable(dst) &&
      addressable(dst + bytes) && GpuWriteMapped("DMA_DATA", dst, bytes) &&
      host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(src), bytes)) {
    // src may be CS-written; land pending writes first, and those under dst
    // before the copy lands over them. Copy even if the flush fails: a
    // possibly-stale source beats silently dropping the copy. Unless compute
    // ranges track guest writes (DELTA_GPU_CS_TRACK), every range is flushed:
    // GTA:SA issues ~300 of these a frame, which keeps lazy writeback short,
    // and some reader of long-dirty ranges depends on that.
    if (render::CsTracksGuestWrites()) {
      render::FlushCsWritesRange(renderer, src, bytes, "dma");
      render::FlushCsWritesRange(renderer, dst, bytes, "dma");
    } else {
      render::FlushCsWrites(renderer);
    }
    NoteGpuWriteNear("DMA_DATA", dst, bytes);
    std::memcpy(reinterpret_cast<void*>(dst),
                reinterpret_cast<const void*>(src), bytes);
    copied = true;
  }
  // src_sel 2 = the packet's own dword, repeated: a fill. GNM clears surfaces
  // and their CMASK/HTILE metadata this way, so apply it to guest memory and
  // let the renderer clear any target it covers.
  if (!kNoCopy && src_sel == 2 && dst_is_memory && bytes &&
      bytes <= 0x8000000u && addressable(dst) && addressable(dst + bytes) &&
      GpuWriteMapped("DMA_DATA fill", dst, bytes)) {
    render::ApplyMemoryFill(renderer, dst, bytes, body[1]);
    copied = true;
  }
  TraceDmaData(control, body[5] & ~0x1fffffu, src_sel, dst_sel, src, dst, bytes,
               copied);
}

// Gnm writes its 32/64-bit submit/flip fence labels with WRITE_DATA. The
// control field's dst_sel encoding varies (the flip-label packet built by
// sceGnmInsertFlip uses control=5, not the [11:8]=memory form), so don't gate
// on dst_sel: a memory write resolves to a real guest label address, while a
// register write yields a tiny offset IsLabelAddress rejects.
// body: control, dstLo, dstHi, data...
void HandleWriteData(const u32* body, u32 count) {
  if (count >= 3)
    TraceAddrWatch("WRITE_DATA",
                   (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1],
                   (count - 3) * 4u, count >= 4 ? body[3] : 0,
                   /*max_lines=*/12);
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u32 dwords = count - 3;
  NoteGpuWriteNear("WRITE_DATA", address, body[3]);
  if (IsLabelAddress(address) && IsLabelAddress(address + (u64)dwords * 4) &&
      GpuWriteMapped("WRITE_DATA", address, (u64)dwords * 4)) {
    std::memcpy(reinterpret_cast<void*>(address), &body[3], (size_t)dwords * 4);
    g_fence_labels.Note(address);
  }
  TraceDataWrite(address, dwords, dwords ? body[3] : 0);
}

// body: eventCtrl, addrLo, addrHi+sel, dataLo, dataHi
void HandleEventWriteEop(const u32* body, u32 count) {
  if (count >= 3)
    TraceAddrWatch("EVENT_WRITE_EOP",
                   (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1], 8,
                   count >= 4 ? body[3] : 0, /*max_lines=*/8);
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u64 value = static_cast<u64>(body[3]) |
                    (static_cast<u64>(count >= 5 ? body[4] : 0) << 32);
  WriteEventLabel("EOP", address, (body[2] >> 29) & 0x7, value,
                  (body[2] >> 24) & 0x3);
}

// body: eventCtrl, selBits, addrLo, addrHi, dataLo, dataHi
void HandleReleaseMem(const u32* body, u32 count) {
  if (count >= 4)
    TraceAddrWatch("RELEASE_MEM",
                   (static_cast<u64>(body[3] & 0xFFFF) << 32) | body[2], 8,
                   count >= 5 ? body[4] : 0, /*max_lines=*/8);
  if (count < 5)
    return;
  const u64 address =
      (static_cast<u64>(body[3] & 0xFFFF) << 32) | (body[2] & ~0x3u);
  const u64 value = static_cast<u64>(body[4]) |
                    (static_cast<u64>(count >= 6 ? body[5] : 0) << 32);
  WriteEventLabel("RELEASE_MEM", address, (body[1] >> 29) & 0x7, value,
                  (body[1] >> 24) & 0x7);
}

// EVENT_WRITE with an address is an occlusion/statistics query: the CP samples
// a counter and writes it as a 64 bit value whose TOP BIT is the done flag.
// body: eventCtrl (EVENT_TYPE[5:0], EVENT_INDEX[11:8]), addrLo, addrHi.
// GTA:SA polls exactly that bit (16 query slots ANDed against
// 0x8000000000000000) before it will render anything.
// The value is a monotonically increasing counter, not a constant: the slots
// come in begin/end pairs eight bytes apart and the caller reads end - begin as
// "samples that passed"; writing the same number to both would answer "nothing
// was visible" and cull the frame just as thoroughly as never writing at all.
// Rendering everything is the honest answer for a backend that does not track
// occlusion.
void HandleEventWrite(const u32* body, u32 count) {
  if (count < 3)
    return;  // no address: a plain pipeline event (cache flush and friends)
  // Only EVENT_INDEX 1 (ZPASS_DONE), 2 (SAMPLE_PIPELINESTAT) and 3
  // (SAMPLE_STREAMOUTSTATS) carry a destination. The rest reuse the same
  // opcode for events that write nothing, and taking their DW2/DW3 as an
  // address stores through whatever those words happen to hold.
  const u32 event_index = (body[0] >> 8) & 0xF;
  if (event_index < 1 || event_index > 3)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x7u);
  TraceAddrWatch("EVENT_WRITE", address, 8, body[0], /*max_lines=*/8);
  if (!IsLabelAddress(address) || !IsLabelAddress(address + 8))
    return;
  // ZPASS_DONE does not write ONE counter: the CP fans it out to one qword per
  // render backend, at a 16 byte stride, and the caller sums them. Liverpool
  // has 8 RBs, so a begin packet at `base` fills base + i*16 and the matching
  // end packet at base+8 fills base + i*16 + 8, which is exactly the pair
  // stride GTA:SA's poll walks. Writing a single qword leaves fourteen of its
  // sixteen slots at zero and the poll never completes.
  constexpr u32 kRenderBackends = 8;
  const u64 span = static_cast<u64>(kRenderBackends) * 16;
  if (!IsGuestRange(address, span) ||
      !host_memory::IsMemoryRangeMapped(reinterpret_cast<const void*>(address),
                                        span))
    return;
  static base::Atomic<u64> samples{0};
  const u64 value = (1ull << 63) | (samples.fetch_add(1) + 1);
  for (u32 rb = 0; rb < kRenderBackends; rb++)
    WriteLabel(address + rb * 16, value, true);
  g_fence_labels.Note(address);
  TraceLabelWrite("EVENT_WRITE", address, 2, value);
}

// body: eventCtrl, addrLo, addrHi+cmd, data
void HandleEventWriteEos(const u32* body, u32 count) {
  if (count >= 3)
    TraceAddrWatch("EVENT_WRITE_EOS",
                   (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1], 8,
                   count >= 4 ? body[3] : 0, /*max_lines=*/8);
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  WriteLabel(address, body[3], false);
  TraceEosLabel(address, body[3]);
}

void HandleDrawPacket(render::Renderer& renderer,
                      u32 op,
                      const u32* body,
                      u32 count) {
  g_total_draws.fetch_add(1);
  // Ahead of the renderer gate: the watch these can arm is a kernel one, and a
  // run whose device failed to come up is when register state is worth having.
  const u32 frame = g_presented_frames + 1;
  const u64 ps_addr = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  MaybeArmRootWriteWatch(g_regs, ps_addr, frame);
  TraceRegisterSources(g_regs, ps_addr, frame);
  TraceFirstTexturedPs(g_regs, ps_addr);

  if (renderer.available()) {
    DrawPacket packet;
    packet.op = op;
    packet.body = body;
    packet.count = count;
    packet.index_type = g_index.type;
    packet.index_base = g_index.base;
    packet.indirect_base = g_index.indirect_base;
    packet.num_instances = g_index.num_instances;
    packet.frame = frame;

    RenderQueue& queue = GuestRenderQueue();
    if (queue.running()) {
      render::DrawInfo& d = queue.NextDraw();
      const bool renderable = BuildDrawInfo(renderer, g_regs, packet, d);
      if (!g_frame_active) {
        queue.PushBeginFrame();
        g_frame_active = true;
      }
      if (renderable)
        queue.PushDraw();
    } else {
      thread_local render::DrawInfo d;
      d.Reset();
      const bool renderable = BuildDrawInfo(renderer, g_regs, packet, d);
      // On the first draw of a frame, and for every draw the renderer sees
      // including the ones dropped below, so a frame whose draws all decline
      // still ends (and presents) like any other.
      if (!g_frame_active) {
        render::BeginFrame(renderer);
        g_frame_active = true;
      }
      if (renderable)
        render::Draw(renderer, d);
    }
  }
  TraceDrawRegisters(g_regs, op, body, count);
}

// A packet that writes guest memory the title waits on (fence labels, query
// results) runs behind the draws queued ahead of it: the title may reuse what
// those draws read as soon as it sees the write.
// `label` is the address the packet writes, noted here so a WAIT_REG_MEM on
// it knows it is ours before the write has run.
void NoteLateLabel(u64 address, const char* packet);

void InOrder(void (*handle)(const u32*, u32),
             const u32* body,
             u32 count,
             u64 label) {
  NoteLateLabel(label, handle == HandleWriteData ? "WRITE_DATA" : "label");
  RenderQueue& queue = GuestRenderQueue();
  if (!queue.running()) {
    handle(body, count);
    return;
  }
  if (label && IsLabelAddress(label)) {
    g_fence_labels.Note(label);
    // Wide enough for EVENT_WRITE's eight 16-byte slots.
    queue.NotePendingWrite(label, 128);
  }
  queue.PushCall([handle, words = base::Vector<u32>(body, body + count)] {
    handle(words.data(), static_cast<u32>(words.size()));
  });
}

// The 48-bit address a packet carries as lo, hi dwords at body[at].
u64 PacketAddress(const u32* body, u32 at) {
  return ((static_cast<u64>(body[at + 1] & 0xFFFF) << 32) | body[at]) & ~3ull;
}

// DMA runs behind the queued draws like any other GPU write; the walk only
// waits for it when it reads what the copy writes.
void QueueDmaData(render::Renderer& renderer, const u32* body, u32 count) {
  RenderQueue& queue = GuestRenderQueue();
  if (!queue.running() || count < 6) {
    HandleDmaData(renderer, body, count);
    return;
  }
  queue.NotePendingWrite(PacketAddress(body, 3), body[5] & 0x1FFFFF);
  queue.PushCall([&renderer, words = base::Vector<u32>(body, body + count)] {
    HandleDmaData(renderer, words.data(), static_cast<u32>(words.size()));
  });
}

bool IsDraw(u32 op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDIRECT ||
         op == IT_DRAW_INDEX_INDIRECT || op == IT_DRAW_INDEX_MULTI_AUTO;
}

// --- the walks -------------------------------------------------------------

// Resolve an in-stream IT_INDIRECT_BUFFER body into a mapped host pointer and
// dword count. Mirrors the kernel's gc_insert_indirect_buffer checks: a
// non-zero ib_size whose GPU address sits in the guest range and is actually
// mapped. Any failure returns false so the caller skips the chain instead of
// dereferencing garbage.
bool ResolveIndirectBuffer(const u32* body,
                           u32 count,
                           const u32*& out,
                           u32& out_dwords) {
  out_dwords = 0;
  if (count < 3)
    return false;
  const u64 address = (static_cast<u64>(body[1] & 0xFF) << 32) | body[0];
  const u32 dwords = body[2] & 0xFFFFF;
  const u64 bytes = static_cast<u64>(dwords) * 4;
  if (!IsGuestRange(address, bytes))
    return false;
  const void* p = reinterpret_cast<const void*>(address);
  if (!host_memory::IsMemoryRangeMapped(p, bytes))
    return false;
  // A queued DMA or WRITE_DATA may still be writing the chained commands.
  if (GuestRenderQueue().PendingWriteOverlaps(address, bytes))
    OwnRenderer("pending-write");
  out = static_cast<const u32*>(p);
  out_dwords = dwords;
  return true;
}

u32 WalkDcb(render::Renderer& renderer,
            const u32* p,
            u32 words,
            u32 depth,
            bool dump);

// DELTA_GPU_WAITTRACE: the words recent waits gave up on, so the packet that
// writes one later can be named with its ring and how late it came.
struct FailedWait {
  u64 address = 0;
  base::TimeTicks when;
};
FailedWait g_failed_waits[2048];
u32 g_failed_next = 0;
// The ring the walk on this thread executes ("gfx", or "acb" for a compute
// queue's commands).
thread_local const char* t_walk_ring = "gfx";

struct LabelWrite {
  base::TimeTicks when;
  const char* ring;
  const char* packet;
  u64 value;
};
base::Mutex g_label_writes_mutex;
base::HashMap<u64, LabelWrite> g_label_writes;

void NoteLateLabel(u64 address, const char* packet) {
  if (!kWaitTraceFrom || !address)
    return;
  {
    base::LockGuard<base::Mutex> lock(g_label_writes_mutex);
    if (g_label_writes.size() > (1u << 20))
      g_label_writes.clear();
    g_label_writes[address & ~3ull] = {base::TimeTicks::Now(), t_walk_ring,
                                       packet, 0};
  }
  for (FailedWait& f : g_failed_waits)
    if (f.address == (address & ~3ull)) {
      BASE_LOGI("waittrace", "late: {} on {} writes {:#x} {} ms after the wait",
                packet, t_walk_ring, (unsigned long long)address,
                (base::TimeTicks::Now() - f.when).InMilliseconds());
      f.address = 0;
      return;
    }
}

void TraceWaitPacket(const u32* p,
                     u32 i,
                     u32 words,
                     u32 depth,
                     const u32* body,
                     u32 count) {
  static const base::TimeTicks start = base::TimeTicks::Now();
  if (!kWaitTraceFrom || count < 5 || !((body[0] >> 4) & 1) ||
      base::TimeTicks::Now() - start < base::Seconds(kWaitTraceFrom))
    return;
  static u32 lines = 0;
  if (lines++ > 3000)
    return;
  const u64 address =
      ((static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
  const bool mapped =
      IsGuestAddress(address) &&
      host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(address), 4);
  const u32 value =
      mapped ? *reinterpret_cast<const volatile u32*>(address) : 0;
  char last[96] = "never written by us";
  if (!WaitRegMemPasses(body, value)) {
    g_failed_waits[g_failed_next++ % 2048] = {address, base::TimeTicks::Now()};
    base::LockGuard<base::Mutex> lock(g_label_writes_mutex);
    auto it = g_label_writes.find(address);
    if (it != g_label_writes.end())
      std::snprintf(last, sizeof(last), "last %s on %s %lld ms ago",
                    it->second.packet, it->second.ring,
                    (long long)(base::TimeTicks::Now() - it->second.when)
                        .InMilliseconds());
  }
  BASE_LOGI("waittrace",
            "frame {} {} dcb {:p}+{:#x}/{:#x} d{} wait {:#x} fn={} want {:#x} "
            "have {:#x} {} {} ({})",
            g_presented_frames, t_walk_ring, (const void*)p, i * 4, words * 4,
            depth,
            (unsigned long long)address, body[0] & 7, body[3], value,
            WaitRegMemPasses(body, value) ? "pass" : "FAIL",
            g_fence_labels.Contains(address) ? "ours" : "foreign", last);
}

// INDIRECT_BUFFER's CHAIN bit (size dword bit 20): continue in the target
// and never come back.
bool IbChains(const u32* body, u32 count) {
  const bool chains = count >= 3 && ((body[2] >> 20) & 1);
  if (chains) {
    static base::Atomic<u64> n{0};
    if (n.fetch_add(1) == 0)
      BASE_LOGI("gpu", "first chained indirect buffer");
  }
  return chains;
}

// Walk one CCB (CE stream), recursing into any chained indirect buffers at
// `depth`. The CE runs ahead of the draw engine: it fills its on-chip RAM and
// dumps it to the guest memory the DE's draws then read as constant buffers.
// The CE stream of the current submit, flattened to its packets in order
// (chained const buffers followed), and how far the CE has run. The CE and the
// DE run side by side, ordered only by their counters: the CE dumps each draw's
// tables into a ring the DE reads, and waits (WAIT_ON_DE_COUNTER_DIFF) before
// lapping what the DE has not consumed. Running all of it before the DE
// overwrote tables still to be read.
struct CeStream {
  base::Vector<const u32*> packets;
  size_t pos = 0;
};
CeStream g_ce;

void FlattenCcb(const u32* p, u32 words, u32 depth) {
  u32 i = 0;
  while (i < words) {
    const u32 hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    if (type != Pm4Type::kType3) {
      if (type == Pm4Type::kType2 || hdr == 0)
        i += 1;
      else if (type == Pm4Type::kType0)
        i += 1 + Pm4Count(hdr);
      else
        break;  // type-1 desync
      continue;
    }
    const u32 op = Pm4Opcode(hdr), count = Pm4Count(hdr);
    if (i + 1 + count > words)
      break;
    const u32* chain = nullptr;
    u32 chain_dwords = 0;
    if (op == IT_INDIRECT_BUFFER_CNST) {
      if (depth < kMaxIbDepth &&
          ResolveIndirectBuffer(&p[i + 1], count, chain, chain_dwords))
        FlattenCcb(chain, chain_dwords, depth + 1);
      if (IbChains(&p[i + 1], count))
        return;
    } else {
      g_ce.packets.push_back(&p[i]);
    }
    i += 1 + count;
  }
}

void ExecCePacket(render::Renderer& renderer, const u32* packet) {
  const u32 op = Pm4Opcode(packet[0]), count = Pm4Count(packet[0]);
  const u32* body = packet + 1;
  NoteCcbPacket(op);
  switch (op) {
    case IT_WRITE_CONST_RAM: {  // body[0] = byte offset, body[1..] = data
      if (!kCeOn)
        break;
      const u32 offset = body[0] & 0xFFFF;
      const u32 dwords = count > 1 ? count - 1 : 0;
      const bool fits = g_const_ram.Fits(offset, dwords);
      if (fits)
        g_const_ram.Write(offset, &body[1], dwords);
      TraceConstRam("write", offset, dwords, 0, fits ? "ok" : "off+n>ceram",
                    dwords ? body[1] : 0);
      break;
    }
    case IT_LOAD_CONST_RAM: {  // addrLo, addrHi, num_dwords, byte offset
      if (!kCeOn || count < 4)
        break;
      // The raw body, once: an address that does not move across a run of
      // chunked loads is a field-order mistake, not a title loading the same
      // bytes forty-eight times, and the decoded trace below cannot show the
      // difference.
      {
        static int shown = 0;
        if (CeTraceOn() && shown < 6) {
          shown++;
          BASE_LOGI(
              "ce", "load raw count={} body={:08x} {:08x} {:08x} {:08x}",
              count, body[0], body[1], body[2], count > 3 ? body[3] : 0u);
        }
      }
      const u64 address =
          (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
      const u32 dwords = body[2] & 0x7FFF, offset = body[3] & 0xFFFF;
      const bool in_guest = IsGuestRange(address, (u64)dwords * 4);
      const bool fits = g_const_ram.Fits(offset, dwords);
      if (in_guest && fits)
        g_const_ram.Write(offset, reinterpret_cast<const void*>(address),
                          dwords);
      TraceConstRam("load", offset, dwords, address,
                    !in_guest ? "addr-not-guest"
                    : !fits   ? "off+n>ceram"
                              : "ok",
                    in_guest ? *reinterpret_cast<const u32*>(address) : 0);
      break;
    }
    case IT_DUMP_CONST_RAM:
    case IT_DUMP_CONST_RAM_OFFSET: {  // offset, num_dwords, addrLo, addrHi
      if (!kCeOn || count < 4)
        break;
      const u32 offset = body[0] & 0xFFFF, dwords = body[1] & 0x7FFF;
      const u64 address =
          (static_cast<u64>(body[3] & 0xFFFF) << 32) | body[2];
      const bool in_guest = IsGuestRange(address, (u64)dwords * 4);
      const bool fits = g_const_ram.Fits(offset, dwords);
      if (in_guest && fits)
        g_const_ram.Read(offset, reinterpret_cast<void*>(address), dwords);
      TraceConstRam(op == IT_DUMP_CONST_RAM ? "dump" : "dump.off", offset,
                    dwords, address,
                    !in_guest ? "addr-not-guest"
                    : !fits   ? "off+n>ceram"
                              : "ok",
                    fits ? g_const_ram.DwordAt(offset) : 0);
      break;
    }
    case IT_INCREMENT_CE_COUNTER:  // the DE later waits on this value
      g_ce_counter = (g_ce_counter + 1) & 0xFFFFFF;
      break;
    case IT_INDIRECT_BUFFER: {
      const u32* chain = nullptr;
      u32 chain_dwords = 0;
      if (ResolveIndirectBuffer(body, count, chain, chain_dwords))
        WalkDcb(renderer, chain, chain_dwords, 1, false);
      break;
    }
    default:
      break;
  }
}

// Run the CE until `done` holds or it reaches a WAIT_ON_DE_COUNTER_DIFF the DE
// has not satisfied. `force` runs everything left, waits included: the DE has
// finished and will not move the counter again.
template <class Done>
void RunCe(render::Renderer& renderer, Done done, bool force) {
  while (g_ce.pos < g_ce.packets.size() && !done()) {
    const u32* packet = g_ce.packets[g_ce.pos];
    if (!force && Pm4Opcode(packet[0]) == IT_WAIT_ON_DE_COUNTER_DIFF &&
        Pm4Count(packet[0]) >= 1) {
      const u32 ahead = (g_ce_counter - g_de_counter) & 0xFFFFFF;
      if (ahead >= (packet[1] & 0xFFFFFF))
        return;
    }
    ExecCePacket(renderer, packet);
    g_ce.pos++;
  }
}

void FinishCe(render::Renderer& renderer) {
  RunCe(renderer, [] { return false; }, /*force=*/true);
  g_ce.packets.clear();
  g_ce.pos = 0;
}

// Walk one DCB (DE stream), issuing draws and dispatches and recursing into any
// chained IT_INDIRECT_BUFFER at `depth`. Returns the walk position (dwords
// consumed) so the top-level caller can report how far it got.
u32 WalkDcb(render::Renderer& renderer,
            const u32* p,
            u32 words,
            u32 depth,
            bool dump) {
  const bool time_packets = WantPacketCost();
  u32 i = 0;
  while (i < words) {
    const u32 hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    if (type == Pm4Type::kType2 || hdr == 0) {
      // Type-2 NOPs and the zero-dword alignment padding Gnm sprinkles between
      // packets; real packets resume after it.
      // NOT the all-ones filler an async compute ring is initialised with:
      // 0xFFFFFFFF is a well-formed type-3 header (opcode 0xff, count 16384),
      // so it falls through to the packet path, where the truncation guard
      // stops the walk and every dispatch past the untouched ring is dropped.
      // Skipping it here would change which dispatches run, so that is a fix
      // to make and measure on its own.
      i += 1;
      continue;
    }
    if (type == Pm4Type::kType0) {
      // Type-0 writes a run of consecutive registers (base in hdr[15:0], count
      // in hdr[29:16]+1) directly into the register file. Treating this as a
      // desync and stopping dropped every later draw (the room floor) in any
      // command buffer that used type-0.
      const u32 count = Pm4Count(hdr);
      const u32 base = Pm4Type0Reg(hdr);  // absolute register offset
      const u32 available = base::Min(count, words - i - 1);
      for (u32 k = 0; k < available; k++)
        if (base + k < kRegFileSize)
          g_regs[base + k] = p[i + 1 + k];
      NoteRegisterWrites(base, &p[i + 1], available, 0);
      i += 1 + count;
      continue;
    }
    if (type != Pm4Type::kType3) {
      TraceDesync(i, words, static_cast<u32>(type), hdr, /*force=*/dump);
      break;  // a type-1 header is a genuine desync
    }

    const u32 op = Pm4Opcode(hdr);
    const u32 count = Pm4Count(hdr);  // body dword count
    const u32* body = &p[i + 1];
    NotePacket(op);
    const auto op_start =
        time_packets ? base::TimeTicks::Now() : base::TimeTicks{};
    if (dump)
      TraceDcbPacket(i, op, count);
    if (i + 1 + count > words)
      break;  // truncated / desync
    switch (op) {
      case IT_DISPATCH_DIRECT:
        DispatchCompute(renderer, g_regs, body, count);
        break;
      case IT_SET_CONTEXT_REG:
        SetRegs(kContextRegBase, body, count);
        break;
      case IT_SET_SH_REG:
      case IT_SET_SH_REG_INDEX:
        SetRegs(kShRegBase, body, count);
        break;
      case IT_SET_UCONFIG_REG:
        SetRegs(kUConfigRegBase, body, count);
        break;
      case IT_SET_CONFIG_REG:
        SetRegs(kConfigRegBase, body, count);
        break;
      case IT_INDEX_TYPE:
        if (count >= 1)
          g_index.type = body[0] & 0x3;
        break;
      case IT_INDEX_BASE:  // index buffer base (byte address) lo/hi
        if (count >= 2)
          g_index.base = (static_cast<u64>(body[1] & 0xFF) << 32) | body[0];
        break;
      case IT_SET_BASE:
        // base_index 1 = DRAW_INDIRECT_BASE: where the indirect draws read
        // their argument structs from. body: baseIndex, addrLo, addrHi.
        if (count >= 3 && (body[0] & 0xF) == 1)
          g_index.indirect_base =
              (static_cast<u64>(body[2] & 0xFF) << 32) | (body[1] & ~0x3u);
        break;
      case IT_NUM_INSTANCES:
        g_index.num_instances = (count >= 1 && body[0]) ? body[0] : 1;
        break;
      case IT_WAIT_REG_MEM:
        do {
          // A memory poll waits on a label, and the renderer thread reaches it
          // only after running every command that could write one ahead of it;
          // a register poll reads the walk's own register file.
          // What the walk reads after the wait may be what it waits for, and
          // the producer may be the title's CPU as well as a queued label: wait
          // here, but only drain the queue when the word is not there yet.
          // A dispatch may have stored the word, and its results stay on the
          // GPU until something reads them.
          if (count >= 5 && ((body[0] >> 4) & 1)) {
            const u64 address =
                ((static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
            if (IsGuestAddress(address))
              FlushForWalkRead(renderer, address, 4, "wait-reg-mem");
          }
          if (count >= 5 && GuestRenderQueue().running()) {
            const volatile u32* polled = WaitRegMemTarget(body);
            if (polled && GuestRenderQueue().PendingWriteOverlaps(
                              reinterpret_cast<u64>(polled), 4))
              OwnRenderer("wait-reg-mem");
            if (!polled) {
              NoteUnknownWait(body);
              break;
            }
            if (WaitRegMemPasses(body, *polled))
              break;
            OwnRenderer("wait-reg-mem");
          }
          HandleWaitRegMem(body, count);
        } while (false);
        TraceWaitPacket(p, i, words, depth, body, count);
        break;
      case IT_DMA_DATA:
        QueueDmaData(renderer, body, count);
        break;
      case IT_WRITE_DATA:
        if (count >= 4)
          GuestRenderQueue().NotePendingWrite(PacketAddress(body, 1),
                                              (count - 3) * 4ull);
        InOrder(HandleWriteData, body, count,
                count >= 3 ? PacketAddress(body, 1) : 0);
        break;
      case IT_EVENT_WRITE_EOP:
        InOrder(HandleEventWriteEop, body, count,
                count >= 3 ? PacketAddress(body, 1) : 0);
        break;
      case IT_RELEASE_MEM:
        InOrder(HandleReleaseMem, body, count,
                count >= 4 ? PacketAddress(body, 2) : 0);
        break;
      case IT_EVENT_WRITE:
        InOrder(HandleEventWrite, body, count,
                count >= 3 ? PacketAddress(body, 1) & ~7ull : 0);
        break;
      case IT_EVENT_WRITE_EOS:
        InOrder(HandleEventWriteEos, body, count,
                count >= 3 ? PacketAddress(body, 1) : 0);
        break;
      case IT_INDIRECT_BUFFER:
      case IT_INDIRECT_BUFFER_CNST: {  // chained buffer (nested CMDBUF)
        const u32* chain = nullptr;
        u32 chain_dwords = 0;
        const bool followed =
            depth < kMaxIbDepth &&
            ResolveIndirectBuffer(body, count, chain, chain_dwords);
        TraceIndirectBuffer(i, depth, chain_dwords, followed);
        if (followed) {
          if (op == IT_INDIRECT_BUFFER_CNST)
            FlattenCcb(chain, chain_dwords, depth + 1);
          else
            WalkDcb(renderer, chain, chain_dwords, depth + 1, dump);
        }
        // CHAIN: a jump, not a call. What follows the packet in this buffer
        // is whatever an earlier frame left there; Uncharted 2 reuses its
        // command memory, and walking on ran its stale waits and dispatches.
        if (IbChains(body, count))
          return words;
        break;
      }
      case IT_INCREMENT_CE_COUNTER:
        g_ce_counter = (g_ce_counter + 1) & 0xFFFFFF;
        TraceCounter("IT_INCREMENT_CE_COUNTER", g_ce_counter);
        break;
      case IT_INCREMENT_DE_COUNTER:
        g_de_counter = (g_de_counter + 1) & 0xFFFFFF;
        TraceCounter("IT_INCREMENT_DE_COUNTER", g_de_counter);
        break;
      case IT_WAIT_ON_CE_COUNTER:
        // The DE waits until the CE is ahead of it.
        RunCe(
            renderer,
            [] {
              const u32 ahead = (g_ce_counter - g_de_counter) & 0xFFFFFF;
              return ahead != 0 && ahead < 0x800000;
            },
            /*force=*/false);
        TraceWaitOnCeCounter(count >= 1 ? (body[0] & 0xFFFFFF) : 0,
                             g_ce_counter);
        break;
      default:
        if (IsDraw(op))
          HandleDrawPacket(renderer, op, body, count);
        else
          TraceUnhandledOpcode(op, count);
        break;
    }
    if (time_packets)
      NotePacketCost(
          op, ((base::TimeTicks::Now() - op_start).InMicroseconds() * 1000));
    i += 1 + count;
  }
  return i;
}

// Look ahead through a command buffer for the shaders its draws will need, so
// they compile on worker threads before the walk reaches them. Only register
// writes are followed; a draw whose state this misreads just compiles in line
// as it always did.
void PrefetchWalk(Regs& regs, const u32* p, u32 words, u32 depth) {
  u32 i = 0;
  while (i < words) {
    const u32 hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    if (type == Pm4Type::kType2 || hdr == 0) {
      i += 1;
      continue;
    }
    const u32 count = Pm4Count(hdr);
    if (type == Pm4Type::kType0) {
      const u32 base = Pm4Type0Reg(hdr);
      for (u32 k = 0; k < count && i + 1 + k < words; k++)
        if (base + k < kRegFileSize)
          regs[base + k] = p[i + 1 + k];
      i += 1 + count;
      continue;
    }
    if (type != Pm4Type::kType3 || i + 1 + count > words)
      return;
    const u32 op = Pm4Opcode(hdr);
    const u32* body = &p[i + 1];
    u32 reg_base = 0;
    switch (op) {
      case IT_SET_CONTEXT_REG:
        reg_base = kContextRegBase;
        break;
      case IT_SET_SH_REG:
      case IT_SET_SH_REG_INDEX:
        reg_base = kShRegBase;
        break;
      case IT_SET_UCONFIG_REG:
        reg_base = kUConfigRegBase;
        break;
      case IT_SET_CONFIG_REG:
        reg_base = kConfigRegBase;
        break;
      case IT_INDIRECT_BUFFER: {
        const u32* chain = nullptr;
        u32 chain_dwords = 0;
        if (depth < kMaxIbDepth &&
            ResolveIndirectBuffer(body, count, chain, chain_dwords))
          PrefetchWalk(regs, chain, chain_dwords, depth + 1);
        if (IbChains(body, count))
          return;
        break;
      }
      case IT_DISPATCH_DIRECT:
        PrefetchComputeDispatch(regs);
        break;
      case IT_WAIT_REG_MEM:
        // Past a wait that has not passed, the title may not have written
        // the commands yet (Uncharted 2 fills a submitted frame behind
        // them): what follows is not worth reading ahead, or safe to.
        if (count >= 5 && ((body[0] >> 4) & 1)) {
          const u64 address =
              ((static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
          if (!IsGuestAddress(address) ||
              !host_memory::IsMemoryRangeMapped(
                  reinterpret_cast<const void*>(address), 4) ||
              !WaitRegMemPasses(
                  body, *reinterpret_cast<const volatile u32*>(address)))
            return;
        }
        break;
      default:
        if (IsDraw(op))
          PrefetchDrawShaders(regs);
        break;
    }
    if (reg_base && count) {
      const u32 first = Pm4SetRegAddress(reg_base, body[0]);
      for (u32 k = 1; k < count; k++)
        if (first + k - 1 < kRegFileSize)
          regs[first + k - 1] = body[k];
    }
    i += 1 + count;
  }
}

// Wall time of one command-buffer walk, including the wait for the lock: a
// second submit thread blocked behind the first is time the guest is stalled on
// us either way.
struct ScopedWalkTimer {
  base::TimeTicks start = base::TimeTicks::Now();
  ~ScopedWalkTimer() {
    render::g_ns_dcb +=
        ((base::TimeTicks::Now() - start).InMicroseconds() * 1000);
    render::g_dcb_n++;
  }
};

// The renderer comes up on the first submission rather than at startup: a title
// that never submits never needs a device.
void StartRendererOnce(render::Renderer& renderer) {
  if (g_renderer_started)
    return;
  g_renderer_started = true;
  render::Init(renderer);
  // PS4 memory is mapped once per address (no aliases), so the CPU writes the
  // tracker sees are all of them: the draw path may keep guest copies across
  // submissions.
  GuestWriteTracker().Enable();
  // The resource replay reads descriptor tables out of guest memory a compute
  // dispatch may still own; it is below the renderer, so it cannot ask itself.
  gcn::g_flush_guest_range = [](u64 address, u64 bytes) {
    FlushForWalkRead(render::DefaultRenderer(), address, bytes, "desc");
  };
  if (kRenderThread && renderer.available())
    GuestRenderQueue().Start(renderer);
}

}  // namespace

void SetWriteWatchCallback(WriteWatchCallback callback) {
  SetWriteWatch(callback);
}

void SetPs4NeoMode(bool enabled) {
  base::LockGuard<base::Mutex> lock(g_mutex);
  gcn::SetDefaultIsaMode(enabled ? gcn::IsaMode::kNeo : gcn::IsaMode::kBase);
}

void EndFrame(u64 scanout_base) {
  base::LockGuard<base::Mutex> lock(g_mutex);
  // New frame -> shader code may have been rewritten; let CachedProgram
  // revalidate each address once next frame instead of once per draw.
  gcn::NextProgramCacheGeneration();
  gpu::NextMemoryGeneration();
  render::Renderer& renderer = render::DefaultRenderer();
  if (!g_frame_active || !renderer.available())
    return;
  if (GuestRenderQueue().running())
    GuestRenderQueue().PushEndFrame(scanout_base);
  else
    render::EndFrame(renderer, scanout_base);
  g_frame_active = false;
  g_presented_frames++;
}

void SubmitCcb(const void* ccb, u32 size_bytes) {
  if (!ccb || size_bytes < 4)
    return;
  base::LockGuard<base::Mutex> lock(g_mutex);
  const u32 words = size_bytes / 4;
  TraceCcbSubmit(size_bytes, words);
  FlattenCcb(static_cast<const u32*>(ccb), words, 0);
  TraceCcbHistogram(words);
}

void SubmitDcb(const void* dcb, u32 size_bytes) {
  if (!dcb || size_bytes < 4)
    return;
  ScopedWalkTimer timer;
  // Time the wait for the lock apart from the walk: they mean opposite things,
  // one says "make the walk faster", the other "stop serialising the threads".
  const auto lock_start = base::TimeTicks::Now();
  base::LockGuard<base::Mutex> lock(g_mutex);
  render::g_ns_dcb_lock +=
      ((base::TimeTicks::Now() - lock_start).InMicroseconds() * 1000);

  render::Renderer& renderer = render::DefaultRenderer();
  StartRendererOnce(renderer);

  const auto* p = static_cast<const u32*>(dcb);
  const u32 words = size_bytes / 4;
  const u64 submission = g_total_submits.fetch_add(1) + 1;
  TraceSubmit(dcb, size_bytes, words, submission, g_total_draws.load());
  const bool dump = ShouldDumpDcb(size_bytes);
  TraceDcbStat(words);

  if (kPrefetchShaders && renderer.available()) {
    static Regs prefetch_regs;
    prefetch_regs = g_regs;
    PrefetchWalk(prefetch_regs, p, words, 0);
  }
  const u32 walked = WalkDcb(renderer, p, words, 0, dump);
  FinishCe(renderer);

  if (dump)
    TraceDcbWalkResult(p, words, walked);
  MaybeDumpOpcodeHistogram(walked, words);
}

void SetUnknownWaitHook(void (*hook)(u64 address)) {
  g_unknown_wait_hook = hook;
}

void SubmitRingDcb(const void* dcb, u32 size_bytes) {
  t_walk_ring = "acb";
  SubmitDcb(dcb, size_bytes);
  t_walk_ring = "gfx";
}

}  // namespace gpu::ps4
