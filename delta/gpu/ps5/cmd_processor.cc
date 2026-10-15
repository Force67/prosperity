/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/ps5/cmd_processor.h"
#include "base/arch.h"
#include "guest/session.h"

#include <cstring>

#include "base/logging.h"
#include "base/meta/traits.h"
#include "options/options.h"

#include "base/atomic.h"
#include "base/containers/array.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/agc_regs.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/dmem_aliases.h"
#include "gpu/ps5/draw_state.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/label_publisher.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/reg_state.h"
#include "gpu/render/command.h"
#include "gpu/render/memory/guest_direct.h"
#include "gpu/render/renderer.h"
#include "gpu/write_tracker.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
DELTA_OPTION(u64, kMemWatch, "DELTA_AGC_MEMWATCH", 0);
DELTA_OPTION(u64, kLabelWatch, "DELTA_AGC_LABELWATCH", 0);
DELTA_OPTION(u64, kLabelWatchSize, "DELTA_AGC_LABELWATCH_SIZE", 0x4000000);
}  // namespace

namespace gpu::ps5 {
static void FinishGraphicsFlip(u64 scanout_base);
namespace {

base::Mutex g_mutex;

bool MemWatchHit(u64 base, u64 bytes) {
  const u64 w = kMemWatch;
  return w && base < ((w | 0xFFFF) + 1) && (w & ~0xFFFFull) < base + bytes;
}

struct RingStall {
  bool ring_walk = false;
  bool stalled = false;
  u64 ib[10] = {};
  u32 ib_dw[10] = {};
  u64 wait_addr = 0;
  u64 wait_ref = 0;
  base::TimeTicks wait_since;
  u64 visible_ns = 0;
};
u64 g_total_submits = 0;
bool g_renderer_started = false;
bool g_frame_active = false;

struct IndexState {
  u64 base = 0;
  u32 max = 0;
  u32 num_instances = 1;
};

struct QueueState {
  Regs regs;
  RingStall stall;
  IndexState index;
  u64 draw_indirect_base = 0;
  u64 indirect_args = 0;
  u64 dispatch_indirect_base = 0;
  base::Array<u32, 0x400> pushed_context{};
  bool context_pushed = false;
};
QueueState g_graphics_queue;
base::Map<u32, QueueState> g_ring_queues;
QueueState* g_queue = &g_graphics_queue;

struct GraphicsJob {
  base::Vector<u32> words;
  u64 sequence = 0;
  u64 scanout = 0;
  u64 enqueued_ns = 0;
  u32 tag = 0;
  bool flip = false;
};
base::Mutex g_graphics_lock;
base::ConditionVariable g_graphics_wake;
base::Vector<GraphicsJob> g_graphics_jobs;
size_t g_graphics_head = 0;
u64 g_graphics_sequence = 0, g_graphics_done = 0;
bool g_graphics_started = false;
void RunGraphics();
u64 EnqueueGraphics(GraphicsJob job);

constexpr u32 kMaxIbDepth = 8;

bool IsLabelAddress(u64 address) {
  return IsGuestAddress(address);
}

u64 GpuClockTimestamp() {
  return static_cast<u64>((base::TickClock::NowNs()));
}

struct LabelWrite {
  u64 address, value, old, ns;
};
LabelWrite g_label_history[4096];
base::Atomic<u32> g_label_history_n{0};

bool Compares(u32 function, u64 value, u64 ref) {
  switch (function) {
    case 1:
      return value < ref;
    case 2:
      return value <= ref;
    case 3:
      return value == ref;
    case 4:
      return value != ref;
    case 5:
      return value >= ref;
    case 6:
      return value > ref;
    default:
      return true;
  }
}

struct ParkedWait {
  u64 address, ref, mask;
  u32 function;
  bool wide, hit;
};
base::Mutex g_parked_lock;
base::Map<const void*, ParkedWait> g_parked;

void RecordLabelWrite(u64 address, u64 value, u64 old) {
  const u32 i =
      g_label_history_n.fetch_add(1) % base::ArraySize(g_label_history);
  g_label_history[i] = {address, value, old, base::TickClock::NowNs()};
  NoteLabelWritten();
  base::LockGuard<base::Mutex> lock(g_parked_lock);
  for (auto& [queue, w] : g_parked) {
    (void)queue;
    if (w.hit || w.address != address)
      continue;
    const u64 now = w.wide ? *reinterpret_cast<const volatile u64*>(address)
                           : *reinterpret_cast<const volatile u32*>(address);
    w.hit = Compares(w.function, now & w.mask, w.ref);
  }
}

void WriteLabel(u64 address, u64 value, bool is_64bit) {
  if (!IsLabelAddress(address))
    return;
  u64 old;
  if (is_64bit) {
    old = *reinterpret_cast<volatile u64*>(address);
    *reinterpret_cast<volatile u64*>(address) = value;
  } else {
    old = *reinterpret_cast<volatile u32*>(address);
    *reinterpret_cast<volatile u32*>(address) = static_cast<u32>(value);
  }
  RecordLabelWrite(address, value, old);
}

// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe();
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe_ctx(u64 context_id);
extern "C" bool prosperity_ps5_begin_eop_flip(u64 context, u64* scanout);
extern "C" void prosperity_ps5_complete_eop_flip(u64 context);

void WriteEventLabel(u64 address,
                     u32 data_sel,
                     u64 value,
                     u32 int_sel = 0,
                     u64 context_id = 0) {
  if (kLabelWatch && address >= kLabelWatch &&
      address < kLabelWatch + kLabelWatchSize) {
    static int n = 0;
    if (kLabelWatchSize <= 0x10000 || n++ < 4096 || n % 256 == 0)
      BASE_LOGI("agclabel", "{:#x} sel={} int={} value={:#x} ctx={:#x}",
                address, data_sel, int_sel, value, context_id);
  }
  if (data_sel && !IsLabelAddress(address)) {
    static int n = 0;
    if (n++ < 16)
      BASE_LOGW("agc", "fence label {:#x} (sel {}) is not writable, skipped",
                address, data_sel);
  }
  if (data_sel == 5) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGW("agc", "fence label {:#x} asks for GDS data, unimplemented",
                address);
  }
  const u64 written =
      data_sel == 3 || data_sel == 4 ? GpuClockTimestamp() : value;
  PublishLabel(
      [=] {
        if (data_sel == 1)
          WriteLabel(address, written, false);
        else if (data_sel >= 2 && data_sel <= 4)
          WriteLabel(address, written, true);
        if (int_sel)
          // NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
          prosperity_gpu_end_of_pipe_ctx(context_id ? context_id : value);
      },
      address,
      data_sel == 1                    ? 4
      : data_sel >= 2 && data_sel <= 4 ? 8
                                       : 0,
      data_sel >= 1 && data_sel <= 4 ? &written : nullptr);
}

void HandleDmaData(render::Renderer& renderer, const u32* body, u32 count) {
  if (count < 6)
    return;
  const u32 control = body[0];
  const u32 src_sel = (control >> 29) & 0x3;
  const u32 dst_sel = (control >> 20) & 0x3;
  const u64 src = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
  const u64 dst = (static_cast<u64>(body[4] & 0xFFFF) << 32) | body[3];
  const u32 bytes = body[5] & 0x3FFFFFF;
  const bool src_is_memory = src_sel == 0 || src_sel == 3;
  const bool dst_is_memory = dst_sel == 0 || dst_sel == 3;
  if (MemWatchHit(dst, bytes))
    BASE_LOGI("memwatch", "DMA_DATA control={:#x} src={:#x} dst={:#x} bytes={}",
              control, (unsigned long)src, (unsigned long)dst, bytes);
  const auto addressable = [](u64 address) {
    return address >= 0x1000000ull && address < 0x20000000000ull;
  };
  bool copied = false;
  if (!kNoCopy && bytes && (src_sel == 1 || dst_sel == 1)) {
    copied =
        render::DmaGds(renderer, src_sel, dst_sel, src, dst, bytes, body[1]);
    TraceDmaData(control, src, dst, bytes, copied);
    return;
  }
  constexpr u32 kLabelBytes = 64;
  if (!kNoCopy && src_is_memory && dst_is_memory && bytes &&
      bytes <= 0x1000000u && src != dst && addressable(src) &&
      addressable(src + bytes) && addressable(dst) &&
      addressable(dst + bytes)) {
    if (bytes <= kLabelBytes) {
      render::FlushCsWritesRange(renderer, src, bytes, "dma");
      render::FlushCsWritesRange(renderer, dst, bytes, "dma");
      const auto* from = reinterpret_cast<const u8*>(src);
      PublishLabel(
          [dst, data = base::Vector<u8>(from, from + bytes)] {
            for (size_t k = 0; k + 4 <= data.size(); k += 4) {
              u32 word, old;
              std::memcpy(&word, data.data() + k, 4);
              std::memcpy(&old, reinterpret_cast<const void*>(dst + k), 4);
              std::memcpy(reinterpret_cast<void*>(dst + k), &word, 4);
              RecordLabelWrite(dst + k, word, old);
            }
            const size_t tail = data.size() & ~size_t(3);
            std::memcpy(reinterpret_cast<void*>(dst + tail), data.data() + tail,
                        data.size() - tail);
          },
          dst, bytes, from);
      copied = true;
    } else if (render::GuestBytesSettled(src, bytes) &&
               render::GuestBytesSettled(dst, bytes)) {
      std::memcpy(reinterpret_cast<void*>(dst),
                  reinterpret_cast<const void*>(src), bytes);
      copied = true;
    } else {
      copied = render::ApplyMemoryCopy(renderer, dst, src, bytes);
    }
  }
  if (!kNoCopy && src_sel == 2 && dst_is_memory && bytes &&
      bytes <= 0x8000000u && addressable(dst) && addressable(dst + bytes)) {
    const u32 fill = body[1];
    if (bytes <= kLabelBytes) {
      u32 filled[kLabelBytes / 4];
      for (u32& word : filled)
        word = fill;
      PublishLabel(
          [dst, bytes, fill] {
            u32* words = reinterpret_cast<u32*>(dst);
            for (u32 k = 0; k < bytes / 4; k++) {
              const u32 old = words[k];
              words[k] = fill;
              RecordLabelWrite(dst + k * 4, fill, old);
            }
          },
          dst, bytes, filled);
      render::NoteMemoryFill(renderer, dst, bytes, fill);
    } else {
      render::ApplyMemoryFill(renderer, dst, bytes, fill);
    }
    copied = true;
  }
  TraceDmaData(control, src, dst, bytes, copied);
}

template <typename T>
T AtomicMemOp(u32 op, T dst, T src, T cmp) {
  using S = base::conditional_t<sizeof(T) == 8, i64, i32>;
  switch (op) {
    case 1:
    case 9:
      return dst == cmp ? src : dst;
    case 2:
    case 10:
    case 17:
      return static_cast<S>(src) < static_cast<S>(dst) ? src : dst;
    case 3:
    case 11:
    case 19:
      return static_cast<S>(src) > static_cast<S>(dst) ? src : dst;
    case 7:
      return src;
    case 8:
      return dst == cmp ? src : dst;
    case 15:
      return dst + src;
    case 16:
      return dst - src;
    case 21:
      return dst & src;
    case 22:
      return dst | src;
    case 23:
      return dst ^ src;
    case 24:
      return dst >= src ? T(0) : dst + 1;
    case 25:
      return (dst == 0 || dst > src) ? src : dst - 1;
    default:
      return dst;
  }
}

void HandleAtomicMem(const u32* body, u32 count) {
  if (count < 7)
    return;
  const u32 op = body[0] & 0x7F;
  const u64 address = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
  const u64 src = (static_cast<u64>(body[4]) << 32) | body[3];
  const u64 cmp = (static_cast<u64>(body[6]) << 32) | body[5];
  const bool wide = (op & 0x20) != 0;
  const u32 base_op = op & ~0x60u;
  if (MemWatchHit(address, wide ? 8 : 4))
    BASE_LOGI("memwatch", "ATOMIC_MEM op={} {:#x} src={:#x}", op,
              (unsigned long)address, (unsigned long)src);
  if (!IsLabelAddress(address) || (address & (wide ? 7 : 3)) ||
      !gpu::IsReadableRangeCached(address, wide ? 8 : 4)) {
    static int n = 0;
    if (n++ < 4)
      BASE_LOGW("agc", "ATOMIC_MEM op={} at {:#x}: not guest memory", op,
                address);
    return;
  }
  if (wide) {
    auto* p = reinterpret_cast<volatile u64*>(address);
    const u64 dst = *p;
    u64 r;
    if (base_op == 18)
      r = src < dst ? src : dst;
    else if (base_op == 20)
      r = src > dst ? src : dst;
    else
      r = AtomicMemOp<u64>(base_op, dst, src, cmp);
    *p = r;
  } else {
    auto* p = reinterpret_cast<volatile u32*>(address);
    const u32 dst = *p;
    const u32 s32 = static_cast<u32>(src), c32 = static_cast<u32>(cmp);
    u32 r;
    if (base_op == 18)
      r = s32 < dst ? s32 : dst;
    else if (base_op == 20)
      r = s32 > dst ? s32 : dst;
    else
      r = AtomicMemOp<u32>(base_op, dst, s32, c32);
    *p = r;
  }
}

void HandleClearState(const u32* body, u32 count) {
  if (count < 1)
    return;
  const u32 cmd = body[0] & 0x3;
  if (cmd == 1 || cmd == 3) {
    std::memcpy(g_queue->pushed_context.data(),
                g_queue->regs.At(kContextRegBase),
                g_queue->pushed_context.size() * sizeof(u32));
    g_queue->context_pushed = true;
  } else if (cmd == 2 && g_queue->context_pushed) {
    std::memcpy(&g_queue->regs[kContextRegBase], g_queue->pushed_context.data(),
                g_queue->pushed_context.size() * sizeof(u32));
    g_queue->context_pushed = false;
  }
  if (cmd == 0 || cmd == 3) {
    static int n = 0;
    if (n++ < 2)
      BASE_LOGI("agc", "CLEAR_STATE cmd={}: context defaults not modelled",
                cmd);
  }
}

bool WaitSatisfied(u32 op, const u32* body, u32 count) {
  DELTA_ZONE("ps5.wait_check");
  const bool wide = op == 0x93;
  if (count < (wide ? 8u : 6u))
    return true;
  const u32 function = body[0] & 0x7;
  const u64 address = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
  u64 ref, mask;
  if (wide) {
    ref = (static_cast<u64>(body[4]) << 32) | body[3];
    mask = (static_cast<u64>(body[6]) << 32) | body[5];
  } else {
    ref = body[3];
    mask = body[4];
  }
  if (function == 0 || !IsLabelAddress(address) ||
      !gpu::IsReadableRangeCached(address, wide ? 8 : 4))
    return true;
  if (!render::FlushCsWritesRange(render::DefaultRenderer(), address,
                                  wide ? 8 : 4, "label"))
    return false;
  u64 value = wide ? *reinterpret_cast<const volatile u64*>(address)
                   : *reinterpret_cast<const volatile u32*>(address);
  if (!OverlayHeld(address, wide ? 8 : 4, reinterpret_cast<u8*>(&value))) {
    LabelPending(address, wide ? 8 : 4);
    return false;
  }
  return Compares(function, value & mask, ref);
}

bool WrittenSinceVisible(u64 address, u64 ref, u64 mask, u32 control) {
  const u64 since = g_queue->stall.visible_ns;
  if (!since)
    return false;
  const u32 function = control & 0x7;
  const LabelWrite* before = nullptr;
  for (const LabelWrite& w : g_label_history) {
    if (w.address != address)
      continue;
    if (w.ns >= since && (Compares(function, w.value & mask, ref) ||
                          Compares(function, w.old & mask, ref)))
      return true;
    if (w.ns < since && (!before || w.ns > before->ns))
      before = &w;
  }
  return before && Compares(function, before->value & mask, ref);
}

bool CheckWait(u32 op, const u32* body, u32 count) {
  const bool wide = op == 0x93;
  if (count < (wide ? 8u : 6u))
    return true;
  const u64 address = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
  const u64 ref = wide ? (static_cast<u64>(body[4]) << 32) | body[3] : body[3];
  const u64 mask = wide ? (static_cast<u64>(body[6]) << 32) | body[5] : body[4];
  {
    base::LockGuard<base::Mutex> lock(g_parked_lock);
    ParkedWait& w = g_parked[g_queue];
    if (w.address == address && w.ref == ref && w.hit) {
      g_parked.erase(g_queue);
      return true;
    }
    w = {address, ref, mask, body[0] & 0x7, wide, false};
  }
  if (!WaitSatisfied(op, body, count) &&
      !WrittenSinceVisible(address, ref, mask, body[0]))
    return false;
  base::LockGuard<base::Mutex> lock(g_parked_lock);
  g_parked.erase(g_queue);
  return true;
}

bool WaitGraphics(u32 op, const u32* body, u32 count) {
  DELTA_ZONE("ps5.gfx_wait");
  QueueState* const queue = g_queue;
  const u32 submit_queue = render::g_submit_queue;
  const auto since = base::TimeTicks::Now();
  bool told = false;
  for (;;) {
    if (guest::Stopping())
      return false;
    const u64 seen = LabelSequence();
    if (CheckWait(op, body, count))
      break;
    if (!told && base::TimeTicks::Now() - since >= base::Seconds(5)) {
      told = true;
      const u64 address = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
      BASE_LOGW("agc", "graphics waitOnAddress {:#x} ref={:#x} fn={} pending",
                (unsigned long)address, (unsigned long)body[3], body[0] & 7);
      const u64 now_ns = base::TickClock::NowNs();
      for (const LabelWrite& w : g_label_history)
        if ((w.address & ~(1ull << 63)) == address)
          BASE_LOGW("agc", "    written {:#x} {:.3f} s ago", w.value,
                    (now_ns - w.ns) / 1e9);
      BASE_LOGW("agc", "    now {:#x}, submitted {:.3f} s ago",
                *reinterpret_cast<const volatile u32*>(address),
                (now_ns - g_queue->stall.visible_ns) / 1e9);
    }
    render::SubmitForPublishedLabels();
    g_mutex.unlock();
    WaitLabelWritten(seen, base::Microseconds(200));
    g_mutex.lock();
    g_queue = queue;
    render::g_submit_queue = submit_queue;
  }
  return true;
}

bool StallOnWait(u32 op, const u32* body, u32 count) {
  if (!g_queue->stall.ring_walk) {
    return !WaitGraphics(op, body, count);
  }
  if (CheckWait(op, body, count))
    return false;
  const bool wide = op == 0x93;
  const u64 address = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
  const u64 ref = wide ? (static_cast<u64>(body[4]) << 32) | body[3] : body[3];
  const auto now = base::TimeTicks::Now();
  if (address != g_queue->stall.wait_addr || ref != g_queue->stall.wait_ref) {
    g_queue->stall.wait_addr = address;
    g_queue->stall.wait_ref = ref;
    g_queue->stall.wait_since = now;
    return true;
  }
  if (now - g_queue->stall.wait_since >= base::Seconds(5)) {
    BASE_LOGW("agc", "queue {} waitOnAddress {:#x} ref={:#x} remains pending",
              render::g_submit_queue, (unsigned long)address,
              (unsigned long)ref);
    const u64 now_ns = base::TickClock::NowNs();
    for (const auto& [q, s] : g_ring_queues) {
      if (!s.stall.wait_addr)
        continue;
      BASE_LOGW("agc", "  queue {} last wait {:#x} ref={:#x} now={:#x}", q,
                (unsigned long)s.stall.wait_addr,
                (unsigned long)s.stall.wait_ref,
                (unsigned long)*reinterpret_cast<const volatile u32*>(
                    s.stall.wait_addr));
      for (const LabelWrite& w : g_label_history)
        if ((w.address & ~(1ull << 63)) == s.stall.wait_addr)
          BASE_LOGW(
              "agc", "    {} {:#x} {:.3f} s ago",
              w.address >> 63 ? "graphics went past a wait for" : "written",
              w.value, (now_ns - w.ns) / 1e9);
    }
    ReportHeldLabels();
    g_queue->stall.wait_since = now;
  }
  return true;
}

void HandleWriteData(const u32* body, u32 count) {
  if (count < 4)
    return;
  if (((body[0] >> 8) & 0xF) == 0) {
    WriteDataRegs(g_queue->regs, body, count);
    return;
  }
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u32 dwords = count - 3;
  if (MemWatchHit(address, static_cast<u64>(dwords) * 4))
    BASE_LOGI("memwatch", "WRITE_DATA {:#x} {} dwords: {:08x} {:08x}",
              (unsigned long)address, dwords, body[3],
              dwords > 1 ? body[4] : 0u);
  if (IsLabelAddress(address) &&
      IsLabelAddress(address + static_cast<u64>(dwords) * 4))
    PublishLabel(
        [address, words = base::Vector<u32>(body + 3, body + count)] {
          auto* dst = reinterpret_cast<u32*>(address);
          for (size_t k = 0; k < words.size(); k++) {
            const u32 old = dst[k];
            dst[k] = words[k];
            RecordLabelWrite(address + k * 4, words[k], old);
          }
        },
        address, static_cast<u64>(dwords) * 4, body + 3);
}

void HandleEventWriteEop(const u32* body, u32 count) {
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u64 value = static_cast<u64>(body[3]) |
                    (static_cast<u64>(count >= 5 ? body[4] : 0) << 32);
  WriteEventLabel(address, (body[2] >> 29) & 0x7, value, (body[2] >> 24) & 0x3);
}

void HandleReleaseMem(const u32* body, u32 count) {
  if (count < 5)
    return;
  const u64 address =
      (static_cast<u64>(body[3] & 0xFFFF) << 32) | (body[2] & ~0x3u);
  const u64 value = static_cast<u64>(body[4]) |
                    (static_cast<u64>(count >= 6 ? body[5] : 0) << 32);
  WriteEventLabel(address, (body[1] >> 29) & 0x7, value, (body[1] >> 24) & 0x7,
                  count >= 7 ? body[6] : 0);
  const u64 context = count >= 7 && body[6] ? body[6] : value;
  u64 scanout = 0;
  if (((body[1] >> 24) & 0x7) &&
      prosperity_ps5_begin_eop_flip(context, &scanout)) {
    if (scanout)
      FinishGraphicsFlip(scanout);
    PublishLabel([context] { prosperity_ps5_complete_eop_flip(context); });
    render::SubmitForPublishedLabels();
  }
}

void HandleEventWriteEos(const u32* body, u32 count) {
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  PublishLabel(
      [address, value = body[3]] { WriteLabel(address, value, false); },
      address, 4, body + 3);
}

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
      !IsLabelAddress(dst) || !gpu::IsReadableRangeCached(dst, bytes))
    return;
  u64 value = 0;
  if (src_sel == 9 || src_sel == 18) {
    value = GpuClockTimestamp();
  } else if (src_sel == 10 || src_sel == 11) {
    value = src;
  } else if (is_memory(src_sel)) {
    if (!IsLabelAddress(src) || !gpu::IsReadableRangeCached(src, bytes))
      return;
    std::memcpy(&value, reinterpret_cast<const void*>(src), bytes);
  } else {
    return;
  }
  if (bytes == 8)
    *reinterpret_cast<volatile u64*>(dst) = value;
  else
    *reinterpret_cast<volatile u32*>(dst) = static_cast<u32>(value);
}

void HandleEventWrite(const u32* body, u32 count) {
  if (count < 3)
    return;
  const u32 event_type = body[0] & 0x3F;
  const u32 event_index = (body[0] >> 8) & 0x7;
  if (event_type != 0x39 || event_index != 1)
    return;
  const u64 address = body[1] | (static_cast<u64>(body[2]) << 32);
  constexpr u32 kBlocks = 16;
  constexpr u64 kBytes = kBlocks * 2 * sizeof(u64);
  if (!address || (address & 7) || !IsLabelAddress(address) ||
      !gpu::IsReadableRangeCached(address, kBytes))
    return;
  constexpr u64 kReady = 1ull << 63;
  static u64 samples = 0;
  const u64 value = kReady | (samples & (kReady - 1));
  auto* results = reinterpret_cast<volatile u64*>(address);
  for (u32 db = 0; db < kBlocks; db++)
    results[db * 2] = value;
  samples++;
  TraceOcclusionQuery(address, value);
}

void PublishPacket(void (*handle)(const u32*, u32),
                   const u32* body,
                   u32 count,
                   u64 base = 0,
                   u64 bytes = ~0ull) {
  PublishLabel(
      [handle, words = base::Vector<u32>(body, body + count)] {
        handle(words.data(), static_cast<u32>(words.size()));
      },
      base, bytes);
}

void HandleDispatchIndirect(render::Renderer& renderer,
                            const u32* body,
                            u32 count) {
  u64 args = 0;
  if (count >= 3)
    args = body[0] | (static_cast<u64>(body[1]) << 32);
  else if (count >= 1)
    args = g_queue->dispatch_indirect_base + body[0];
  if (!args || !IsGuestAddress(args) || !gpu::IsReadableRangeCached(args, 12))
    return;
  const u32 initiator = count >= 2 ? body[count - 1] : 5;
  const bool starts_at_zero =
      (initiator & 4) ||
      (!g_queue->regs[mmCOMPUTE_START_X] && !g_queue->regs[mmCOMPUTE_START_Y] &&
       !g_queue->regs[mmCOMPUTE_START_Z]);
  if (starts_at_zero && (render::CanDispatchIndirect(args) ||
                         !render::GuestBytesSettled(args, 12))) {
    const u32 placeholder[4] = {1, 1, 1, initiator};
    DispatchCompute(renderer, g_queue->regs, placeholder, 4, args);
    return;
  }
  if (!render::FlushCsWritesRange(renderer, args, 12, "indirect-cs"))
    return;
  const u32* a = reinterpret_cast<const u32*>(args);
  const u32 groups[4] = {a[0], a[1], a[2], count >= 2 ? body[count - 1] : 5};
  if (groups[0] > (1u << 20) || groups[1] > (1u << 20) ||
      groups[2] > (1u << 20)) {
    static u32 reported = 0;
    if (reported++ < 16)
      BASE_LOGW("agc",
                "indirect dispatch cs={:#x} args={:#x} packet={:p} count={} "
                "body=[{:#x} {:#x} {:#x}] groups=[{:#x} {:#x} {:#x}]",
                g_queue->regs.ShaderAddr(mmCOMPUTE_PGM_LO), args, body, count,
                body[0], count >= 2 ? body[1] : 0, count >= 3 ? body[2] : 0,
                groups[0], groups[1], groups[2]);
  }
  if (!groups[0] || !groups[1] || !groups[2])
    return;
  DispatchCompute(renderer, g_queue->regs, groups, 4);
}

void HandleDrawIndirect(
    render::Renderer& renderer,
    u32 op,
    const u32* body,
    u32 count,
    void (*issue)(render::Renderer&, u32, const u32*, u32)) {
  if (count < 1)
    return;
  const bool indexed = op == 0x25;
  const u64 args = g_queue->draw_indirect_base + body[0];
  const u32 want = indexed ? 20u : 16u;
  if (!g_queue->draw_indirect_base || !IsGuestAddress(args) ||
      !gpu::IsReadableRangeCached(args, want))
    return;
  const bool ngg = (g_queue->regs[mmVGT_SHADER_STAGES_EN] >> 7) & 1;
  const u32 primitive = g_queue->regs[mmVGT_PRIMITIVE_TYPE];
  const bool gpu_args =
      !(args & 3) &&
      (ngg ? !indexed && primitive == 1 && render::CanDrawMeshIndirect()
           : render::CanDrawIndirect(primitive, indexed));
  const u32 initiator = count >= 4 ? body[3] : 0;
  if (gpu_args && indexed && g_queue->index.max && g_queue->index.base) {
    const u64 base = g_queue->index.base;
    const u32 idx_body[5] = {g_queue->index.max, static_cast<u32>(base),
                             static_cast<u32>(base >> 32), g_queue->index.max,
                             initiator};
    g_queue->indirect_args = args;
    issue(renderer, IT_DRAW_INDEX_2, idx_body, 5);
    g_queue->indirect_args = 0;
    return;
  }
  if (gpu_args && !indexed) {
    const u32 auto_body[2] = {3, initiator};
    g_queue->indirect_args = args;
    issue(renderer, IT_DRAW_INDEX_AUTO, auto_body, 2);
    g_queue->indirect_args = 0;
    return;
  }
  if (!render::FlushCsWritesRange(renderer, args, want, "indirect-draw"))
    return;
  const u32* a = reinterpret_cast<const u32*>(args);
  if (a[1] > (1u << 20))
    BASE_LOGW("agc",
              "indirect draw op={:#x} args={:#x} packet={:p} words=[{:#x} "
              "{:#x} {:#x} {:#x} {:#x}]",
              op, args, body, a[0], a[1], a[2], a[3], indexed ? a[4] : 0);
  if (!a[0] || !a[1])
    return;
  const u32 saved_instances = g_queue->index.num_instances;
  g_queue->index.num_instances = a[1];
  if (!indexed) {
    const u32 auto_body[2] = {a[0], initiator};
    issue(renderer, IT_DRAW_INDEX_AUTO, auto_body, 2);
  } else {
    const u32 type = g_queue->regs[mmVGT_INDEX_TYPE] & 3u;
    const u32 stride = type == 1 ? 4u : type == 2 ? 1u : 2u;
    const u64 base = g_queue->index.base + static_cast<u64>(a[2]) * stride;
    const u32 idx_body[5] = {g_queue->index.max ? g_queue->index.max : a[0],
                             static_cast<u32>(base),
                             static_cast<u32>(base >> 32), a[0], initiator};
    issue(renderer, IT_DRAW_INDEX_2, idx_body, 5);
  }
  g_queue->index.num_instances = saved_instances;
}

void HandleDrawPacket(render::Renderer& renderer,
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
  packet.index_type = g_queue->regs[mmVGT_INDEX_TYPE] & 3u;
  packet.index_base = g_queue->index.base;
  packet.index_max = g_queue->index.max;
  packet.num_instances = g_queue->index.num_instances;
  packet.indirect_args = g_queue->indirect_args;

  thread_local render::DrawInfo d;
  d.Reset();
  if (!BuildDrawInfo(g_queue->regs, packet, d))
    return;
  if (!g_frame_active) {
    TraceBeginFrame();
    render::BeginFrame(renderer);
    g_frame_active = true;
  }
  TraceDrawSubmit(d);
  NoteDrawIssued(d);
  render::Draw(renderer, d);
  TraceDrawDone();
}

bool IsDraw(u32 op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDEX_MULTI_AUTO;
}

bool OnlyDraws(const u32* p, u32 at, u32 dwords, u32 words) {
  const u32 end = at + dwords;
  if (!dwords || end > words)
    return false;
  while (at < end) {
    const u32 hdr = p[at];
    if (Pm4TypeOf(hdr) != Pm4Type::kType3)
      return false;
    const u32 op = Pm4Opcode(hdr);
    if (op == IT_NOP && ((hdr >> 16) & 0x3FFF) == 0x3FFF) {
      at++;
      continue;
    }
    if (!IsDraw(op) && op != IT_NOP)
      return false;
    at += 1 + Pm4Count(hdr);
  }
  return at == end;
}

u32 Walk(render::Renderer& renderer,
         const u32* p,
         u32 words,
         bool dump,
         u32 depth) {
  if (!p || depth > kMaxIbDepth)
    return words;
  u32 i = 0;
  u32 trail_pos[32] = {}, trail_hdr[32] = {};
  u32 trail_n = 0;
  u32 last_op = 0;
  u32 predicate_end = 0;
  while (i < words) {
    if (predicate_end && i >= predicate_end) {
      render::EndGuestPredicate();
      predicate_end = 0;
    }
    const u32 hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    trail_pos[trail_n & 31] = i;
    trail_hdr[trail_n & 31] = hdr;
    trail_n++;
    if (type == Pm4Type::kType2 || hdr == 0) {
      i += 1;
      continue;
    }
    if (type == Pm4Type::kType0) {
      const u32 count = Pm4Count(hdr);
      const u32 first = Pm4Type0Reg(hdr);
      if (first + count > kRegFileSize || i + 1 + count > words) {
        i += 1;
        continue;
      }
      SetRegRun(g_queue->regs, first, &p[i + 1], count);
      i += 1 + count;
      continue;
    }
    if (type != Pm4Type::kType3) {
      static int n = 0;
      if (n++ < 4) {
        base::String line;
        base::FormatTo(line,
                       "type-{} header {:#x} at dword {}/{} of {:p}, "
                       "resyncing; packets in:",
                       static_cast<u32>(type), hdr, i, words, (const void*)p);
        const u32 seen = trail_n < 32 ? trail_n : 32;
        for (u32 k = 0; k < seen; k++) {
          const u32 s = (trail_n - seen + k) & 31;
          base::FormatTo(line, " [{}]{:08x}", trail_pos[s], trail_hdr[s]);
        }
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
    const u32 count = Pm4Count(hdr);
    const u32* body = &p[i + 1];
    if (op == IT_NOP && ((hdr >> 16) & 0x3FFF) == 0x3FFF) {
      i += 1;
      continue;
    }
    if (i + 1 + count > words) {
      TraceResync(i, words, hdr, op, count);
      i += 1;
      continue;
    }
    NoteOpcode(op);
    TraceOpcodeBody(op, body, count);
    TraceColorBaseSource(op, g_queue->regs[mmCB_COLOR0_BASE]);
    if (dump)
      TraceDcbPacket(i, op, body, count);
    switch (op) {
      case IT_INDIRECT_BUFFER:
      case IT_INDIRECT_BUFFER_CNST: {
        if (count < 3)
          break;
        const u64 ib = (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
        const u32 ib_words = body[2] & 0xFFFFF;
        const bool follow = ib_words && ib_words <= 0x40000 &&
                            IsGuestAddress(ib) &&
                            gpu::IsReadableRangeCached(
                                ib, static_cast<u64>(ib_words) * sizeof(u32));
        TraceIndirectBuffer(ib, ib_words, follow);
        if (follow) {
          render::FlushCsWritesRange(
              renderer, ib, static_cast<u64>(ib_words) * sizeof(u32), "ib");
          u32 start = 0;
          if (g_queue->stall.ib[depth + 1] == ib) {
            start = base::Min(g_queue->stall.ib_dw[depth + 1], ib_words);
            g_queue->stall.ib[depth + 1] = 0;
          }
          const u32 done =
              Walk(renderer, reinterpret_cast<const u32*>(ib) + start,
                   ib_words - start, dump, depth + 1);
          if (g_queue->stall.stalled) {
            g_queue->stall.ib[depth + 1] = ib;
            g_queue->stall.ib_dw[depth + 1] = start + done;
            return i;
          }
        } else {
          static int n = 0;
          if (n++ < 16)
            BASE_LOGW("agc", "IB {:#x}+{:#x} dwords not followed", (u64)ib,
                      ib_words);
        }
        break;
      }
      case IT_SET_CONTEXT_REG:
        SetRegs(g_queue->regs, kContextRegBase, body, count);
        break;
      case IT_SET_SH_REG:
        SetRegs(g_queue->regs, kShRegBase, body, count);
        break;
      case IT_SET_UCONFIG_REG:
        SetRegs(g_queue->regs, kUConfigRegBase, body, count);
        break;
      case IT_SET_CONFIG_REG:
        SetRegs(g_queue->regs, kConfigRegBase, body, count);
        break;
      case 0x61:
        LoadRegImage(g_queue->regs, kContextRegBase, body, count);
        break;
      case 0x5f:
        LoadRegImage(g_queue->regs, kShRegBase, body, count);
        break;
      case 0x5e:
        LoadRegImage(g_queue->regs, kUConfigRegBase, body, count);
        break;
      case 0x9f:
        LoadRegBlock(g_queue->regs, kContextRegBase, body, count);
        break;
      case 0x64:
        LoadRegBlock(g_queue->regs, kUConfigRegBase, body, count);
        break;
      case 0x63:
        LoadRegBlock(g_queue->regs, kShRegBase, body, count);
        break;
      case 0x7a:
        SetRegs(g_queue->regs, kUConfigRegBase, body, count);
        break;
      case 0x10:
        NoteSkippedOpcode(op, "NOP padding");
        break;
      case 0x28:
        NoteSkippedOpcode(op, "CONTEXT_CONTROL, no shadow to select");
        break;
      case 0x4a:
        NoteSkippedOpcode(op, "PREAMBLE_CNTL, a bracket not state");
        break;
      case 0x42:
        NoteSkippedOpcode(op, "PFP_SYNC_ME, one engine here");
        break;
      case 0x3c:
      case 0x93:
        if (StallOnWait(op, body, count)) {
          g_queue->stall.stalled = true;
          return i;
        }
        NoteSkippedOpcode(op, "waitOnAddress, already satisfied");
        break;
      case 0x58:
        if (count >= 7 && (body[6] & 0x3)) {
          rdna::NextProgramGeneration();
          static int n = 0;
          if (n++ < 8)
            BASE_LOGI("agc", "acquireMem gcr={:#x} -> program generation bump",
                      body[6]);
        }
        break;
      case IT_DMA_DATA:
        HandleDmaData(renderer, body, count);
        break;
      case IT_INDEX_TYPE:
        if (count >= 1)
          g_queue->regs[mmVGT_INDEX_TYPE] = body[0];
        break;
      case IT_INDEX_BASE:
        if (count >= 2)
          g_queue->index.base =
              (static_cast<u64>(body[1] & 0xFFFF) << 32) | (body[0] & ~1u);
        break;
      case IT_INDEX_BUFFER_SIZE:
        if (count >= 1)
          g_queue->index.max = body[0];
        break;
      case IT_NUM_INSTANCES:
        if (count && body[0] > (1u << 20))
          BASE_LOGW("agc", "NUM_INSTANCES packet={:p} value={:#x}", body,
                    body[0]);
        g_queue->index.num_instances = (count >= 1 && body[0]) ? body[0] : 1;
        break;
      case IT_DISPATCH_DIRECT:
        DispatchCompute(renderer, g_queue->regs, body, count);
        break;
      case 0x16:
        HandleDispatchIndirect(renderer, body, count);
        break;
      case 0x24:
      case 0x25:
        HandleDrawIndirect(renderer, op, body, count, HandleDrawPacket);
        break;
      case 0x11: {
        if (count < 3 || (body[0] & 0xF) != 1)
          break;
        const u64 base =
            (body[1] & ~0x7ull) | (static_cast<u64>(body[2] & 0xFFFF) << 32);
        if ((hdr >> 1) & 0x3)
          g_queue->dispatch_indirect_base = base;
        else
          g_queue->draw_indirect_base = base;
        break;
      }
      case IT_COND_EXEC: {
        if (count < 4)
          break;
        const u64 flag = (static_cast<u64>(body[1] & 0xFFFF) << 32) |
                         (body[0] & 0xFFFFFFFCu);
        const u32 skip = body[3] & 0x3FFF;
        if (!IsGuestAddress(flag) || !gpu::IsReadableRangeCached(flag, 4))
          break;
        const u32 block = i + 1 + count;
        if (!predicate_end && !render::GuestBytesSettled(flag, 4) &&
            OnlyDraws(p, block, skip, words) &&
            render::BeginGuestPredicate(flag)) {
          predicate_end = block + skip;
          break;
        }
        render::FlushCsWritesRange(renderer, flag, 4, "cond");
        if (*reinterpret_cast<const volatile u32*>(flag) == 0)
          i += skip;
        break;
      }
      case 0x1e:
        if (count >= 3)
          PublishPacket(HandleAtomicMem, body, count,
                        (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1],
                        8);
        break;
      case 0x12:
        HandleClearState(body, count);
        break;
      case 0x8e:
        NoteSkippedOpcode(op, "0x8e, an empty packet after acquireMem");
        break;
      case IT_WRITE_DATA:
        HandleWriteData(body, count);
        break;
      case IT_COPY_DATA:
        if (count >= 5) {
          const u32 dst_sel = ((body[0] >> 8) & 0xF) << 1;
          const bool to_memory = dst_sel == 2 || dst_sel == 4 || dst_sel == 5;
          PublishPacket(HandleCopyData, body, count,
                        body[3] | (static_cast<u64>(body[4]) << 32),
                        to_memory ? 8 : 0);
        }
        break;
      case IT_EVENT_WRITE:
        if (count >= 3 && (body[0] & 0x3F) == 0x39 &&
            ((body[0] >> 8) & 0x7) == 1)
          PublishPacket(HandleEventWrite, body, count,
                        body[1] | (static_cast<u64>(body[2]) << 32), 256);
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
        else
          NoteUnhandledOpcode(op, hdr, i, words, body, count, last_op);
        break;
    }
    last_op = op;
    i += 1 + count;
  }
  if (predicate_end)
    render::EndGuestPredicate();
  return words;
}

void StartRendererOnce(render::Renderer& renderer) {
  if (g_renderer_started)
    return;
  g_renderer_started = true;
  render::Init(renderer);
  render::DeferLabelsToGpu();
  render::RecordComputeInFrame();
  InstallWriteTrackerPolicy();
  GuestWriteTracker().Enable();
  gcn::g_flush_guest_range = [](u64 address, u64 bytes) {
    render::Renderer& renderer = render::DefaultRenderer();
    if (!render::FlushCsWritesForGpu(renderer, address, bytes, "srt") ||
        !render::SpeculateGuestRead(address, bytes))
      render::FlushCsWritesRange(renderer, address, bytes, "srt");
  };
}

}  // namespace

u32 SubmitDcbRing(const void* dcb, u32 size_bytes, u32 queue, u64 visible_ns) {
  if (!dcb || size_bytes < 4)
    return size_bytes / 4;
  base::LockGuard<base::Mutex> lock(g_mutex);
  render::Renderer& renderer = render::DefaultRenderer();
  StartRendererOnce(renderer);
  render::SyncGuestDirect();
  const u32 words = size_bytes / 4;
  const u64 submission = ++g_total_submits;
  const bool dump = TraceSubmit(dcb, size_bytes, words, submission);
  render::g_submit_queue = queue;
  auto found = g_ring_queues.find(queue);
  if (found == g_ring_queues.end())
    found = g_ring_queues.emplace(queue, QueueState{}).first;
  g_queue = &found->second;
  g_queue->stall.ring_walk = true;
  g_queue->stall.visible_ns = visible_ns;
  g_queue->stall.stalled = false;
  const u32 done = Walk(renderer, static_cast<const u32*>(dcb), words, dump, 0);
  render::SubmitForPublishedLabels();
  g_queue->stall.ring_walk = false;
  const bool stalled = g_queue->stall.stalled;
  g_queue = &g_graphics_queue;
  render::g_submit_queue = 0;
  if (!stalled) {
    TraceOpcodeCensus(dcb, words, submission);
    if (dump)
      TraceWalkDone();
    return words;
  }
  return done;
}

static void ProcessGraphicsDcb(const void* dcb,
                               u32 size_bytes,
                               u32 tag,
                               u64 visible_ns) {
  DELTA_ZONE("ps5.submit_dcb");
  if (!dcb || size_bytes < 4)
    return;
  const u64 t_enter = NowNs();
  base::LockGuard<base::Mutex> lock(g_mutex);
  const u64 t_held = NowNs();
  g_stats.ns_dcb_lock += t_held - t_enter;
  g_queue = &g_graphics_queue;
  g_queue->stall.stalled = false;
  g_queue->stall.visible_ns = visible_ns;
  render::g_submit_queue = tag;
  render::Renderer& renderer = render::DefaultRenderer();
  StartRendererOnce(renderer);
  render::SyncGuestDirect();
  TraceRegShadowScan();

  const u32 words = size_bytes / 4;
  const u64 submission = ++g_total_submits;
  const bool dump = TraceSubmit(dcb, size_bytes, words, submission);
  Walk(renderer, static_cast<const u32*>(dcb), words, dump, 0);
  render::SubmitForPublishedLabels();
  TraceOpcodeCensus(dcb, words, submission);
  if (dump)
    TraceWalkDone();
  g_stats.ns_dcb += NowNs() - t_held;
  render::g_dcb_n++;
  render::g_submit_queue = 0;
}

void SubmitDcb(const void* dcb, u32 size_bytes, u32 tag) {
  if (!dcb || size_bytes < 4)
    return;
  GraphicsJob job;
  const auto* words = static_cast<const u32*>(dcb);
  job.words.assign(words, words + size_bytes / 4);
  job.tag = tag;
  EnqueueGraphics(base::move(job));
}

void SubmitCcb(const void* ccb, u32 size_bytes) {
  SubmitDcb(ccb, size_bytes);
}

static void FinishGraphicsFlip(u64 scanout_base) {
  rdna::NextProgramGeneration();
  gpu::FrameMemoryGeneration();
  render::Renderer& renderer = render::DefaultRenderer();
  if (!g_frame_active || !renderer.available())
    return;
  render::EndFrame(renderer, scanout_base);
  g_frame_active = false;
}

static void ProcessGraphicsFlip(u64 scanout_base) {
  base::LockGuard<base::Mutex> lock(g_mutex);
  FinishGraphicsFlip(scanout_base);
}

void EndFrame(u64 scanout_base) {
  {
    base::LockGuard<base::Mutex> lock(g_graphics_lock);
    if (!g_graphics_started) {
      ProcessGraphicsFlip(scanout_base);
      return;
    }
  }
  GraphicsJob job;
  job.scanout = scanout_base;
  job.flip = true;
  const u64 sequence = EnqueueGraphics(base::move(job));
  if (!sequence)
    return;
  base::LockGuard<base::Mutex> lock(g_graphics_lock);
  guest::Wait(g_graphics_wake, g_graphics_lock,
              [&] { return g_graphics_done >= sequence; });
}

namespace {
void RunGraphics() {
  for (;;) {
    GraphicsJob job;
    {
      base::LockGuard<base::Mutex> lock(g_graphics_lock);
      if (!guest::Wait(
              g_graphics_wake, g_graphics_lock,
              [] { return g_graphics_head < g_graphics_jobs.size(); }) ||
          guest::Stopping())
        return;
      job = base::move(g_graphics_jobs[g_graphics_head++]);
      if (g_graphics_head == g_graphics_jobs.size()) {
        g_graphics_jobs.clear();
        g_graphics_head = 0;
      }
    }
    if (job.flip)
      ProcessGraphicsFlip(job.scanout);
    else
      ProcessGraphicsDcb(job.words.data(), job.words.size() * 4, job.tag,
                         job.enqueued_ns);
    {
      base::LockGuard<base::Mutex> lock(g_graphics_lock);
      g_graphics_done = job.sequence;
    }
    g_graphics_wake.NotifyAll();
  }
}

u64 EnqueueGraphics(GraphicsJob job) {
  base::LockGuard<base::Mutex> lock(g_graphics_lock);
  if (guest::Stopping())
    return 0;
  if (!g_graphics_started) {
    if (!guest::SpawnThread("agc-graphics", RunGraphics)) {
      BASE_LOGW("agc", "cannot start graphics submission worker");
      guest::RequestStop();
      return 0;
    }
    g_graphics_started = true;
  }
  const u64 sequence = job.sequence = ++g_graphics_sequence;
  job.enqueued_ns = base::TickClock::NowNs();
  g_graphics_jobs.push_back(base::move(job));
  g_graphics_wake.NotifyOne();
  return sequence;
}

const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_graphics_jobs);
  guest::ResetResource(g_graphics_head);
  guest::ResetResource(g_graphics_sequence);
  guest::ResetResource(g_graphics_done);
  guest::ResetResource(g_graphics_started);
  g_queue = &g_graphics_queue;
  guest::ResetResource(g_graphics_queue);
  guest::ResetResource(g_ring_queues);
  guest::ResetResource(g_renderer_started);
  guest::ResetResource(g_frame_active);
  guest::ResetResource(g_total_submits);
  guest::ResetResource(g_parked);
  guest::ResetResource(g_label_history);
  guest::ResetResource(g_label_history_n);
});
}  // namespace

}  // namespace gpu::ps5

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_agc_submit(u64 dcb_base, u32 size_bytes) {
  gpu::ps5::SubmitDcb(reinterpret_cast<const void*>(dcb_base), size_bytes);
}
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_agc_submit_tagged(u64 dcb_base,
                                             u32 size_bytes,
                                             u32 tag) {
  gpu::ps5::SubmitDcb(reinterpret_cast<const void*>(dcb_base), size_bytes, tag);
}
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" u32 prosperity_agc_submit_ring(u64 dcb_base,
                                          u32 size_bytes,
                                          u32 queue,
                                          u64 visible_ns) {
  return gpu::ps5::SubmitDcbRing(reinterpret_cast<const void*>(dcb_base),
                                 size_bytes, queue, visible_ns);
}
// NOLINTEND(readability-identifier-naming)

extern "C" void prosperity_agc_flip(u64 scanout_base) {
  gpu::ps5::EndFrame(scanout_base);
}
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_note_aperture(u64 base, u64 size) {
  gpu::ps5::NoteGpuPool(base, size);
}
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" int prosperity_gpu_is_aperture(u64 address) {
  return gpu::ps5::IsGpuAddress(address) ? 1 : 0;
}
// NOLINTEND(readability-identifier-naming)
