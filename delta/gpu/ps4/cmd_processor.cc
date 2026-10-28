/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/ps4/cmd_processor.h"
#include "base/arch.h"
#include "base/logging.h"
#include "gpu/ps4/render_queue.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"

#include <cstring>

#include "host_memory/host_memory.h"
#include "options/options.h"

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/compute_dispatch.h"
#include "gpu/ps4/draw_state.h"
#include "gpu/ps4/gpu_labels.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/render/debug/trace.h"
#include "gpu/render/draw/draw_recomp.h"
#include "gpu/render/image/bindless.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/ps4/pm4.h"
#include "gpu/render/memory/guest_direct.h"
#include "gpu/render/renderer.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(bool, kCeOn, "DELTA_GPU_CE", true);
DELTA_OPTION(u32, kWaitTraceFrom, "DELTA_GPU_WAITTRACE", 0);
DELTA_OPTION(bool, kRenderThread, "DELTA_GPU_RENDER_THREAD", true);
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
DELTA_OPTION(bool, kFenceFlush, "DELTA_GPU_CS_FENCE_FLUSH", true);
DELTA_OPTION(bool, kPrefetchShaders, "DELTA_GPU_PREFETCH_SHADERS", true);
}  // namespace

namespace gpu::render {
u32 g_submit_queue = 0;
u32 g_dcb_n = 0;
}  // namespace gpu::render

namespace gpu::ps4 {
namespace {

base::Mutex g_mutex;
Regs g_regs;
base::Atomic<u64> g_total_submits{0};
base::Atomic<u64> g_total_draws{0};
bool g_renderer_started = false;
bool g_frame_active = false;
u32 g_presented_frames = 0;

struct IndexState {
  u32 type = 0;
  u64 base = 0;
  u64 indirect_base = 0;
  u64 dispatch_indirect_base = 0;
  u32 num_instances = 1;
  u32 max = 0;
};
IndexState g_index;

u64 g_ce_counter = 0;
u64 g_de_counter = 0;

constexpr u32 kMaxIbDepth = 8;

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

bool IsLabelAddress(u64 address) {
  return address >= 0x10000ull && address < kGuestEnd;
}

bool GpuWriteMapped(const char* packet, u64 address, u64 bytes) {
  ForgetKnownGpuWrites(address, bytes);
  if (host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(address), bytes))
    return true;
  static base::Atomic<u32> reported{0};
  if (reported.fetch_add(1) < 16)
    BASE_LOGW("gpu", "{} writes unmapped {:#x}+{:#x}, dropped", packet,
              (unsigned long long)address, (unsigned long long)bytes);
  return false;
}

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

u64 GpuClockTimestamp() {
  return static_cast<u64>((base::TickClock::NowNs()));
}

// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe();

void WriteEventLabel(const char* packet,
                     u64 address,
                     u32 data_sel,
                     u64 value,
                     u32 int_sel = 0) {
  if (data_sel && kFenceFlush && render::CsTracksGuestWrites())
    render::FlushCsWrites(render::DefaultRenderer(), "fence");
  const u64 written = data_sel >= 3 ? GpuClockTimestamp() : value;
  WriteAfterGpu(
      [=] {
        if (data_sel == 1)
          WriteLabel(address, written, false);
        else if (data_sel >= 2)
          WriteLabel(address, written, true);
        TraceLabelWrite(packet, address, data_sel, value);
        if (int_sel)
          // NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
          prosperity_gpu_end_of_pipe();
      },
      address,
      data_sel == 1 ? 4
      : data_sel    ? 8
                    : 0,
      written);
}

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

void SetRegs(u32 base, const u32* body, u32 count) {
  if (!count)
    return;
  const u32 first = Pm4SetRegAddress(base, body[0]);
  const u32 values = count - 1;
  for (u32 i = 0; i < values; i++)
    if (first + i < kRegFileSize)
      g_regs[first + i] = body[1 + i];
  NoteRegisterWrites(first, &body[1], values, base);
}

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
      return true;
  }
}

const volatile u32* WaitRegMemTarget(const u32* body) {
  if ((body[0] >> 4) & 1) {
    const u64 address =
        ((static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
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

void (*g_unknown_wait_hook)(u64 address) = nullptr;
base::Atomic<u64> g_watched_pages[4] = {~0ull, ~0ull, ~0ull, ~0ull};

void NoteGpuWriteNear(const char* packet, u64 address, u64 value) {
  if (!g_unknown_wait_hook)
    return;
  for (auto& page : g_watched_pages)
    if (page.load(base::memory_order_relaxed) == (address & ~0xFFFull)) {
      BASE_LOGI("waitwatch", "{} writes {:#x} = {:#x} on a watched page",
                packet, (unsigned long long)address, (unsigned long long)value);
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
  const auto passes = [&](u32 value) {
    u32 held = 0;
    return WaitRegMemPasses(body, value) ||
           (HeldWriteValue(reinterpret_cast<u64>(polled), &held) &&
            WaitRegMemPasses(body, held));
  };
  if (!polled) {
    NoteUnknownWait(body);
    return;
  }

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
  const auto addressable = [](u64 address) {
    return address >= 0x1000000ull && address < kGuestEnd;
  };
  bool copied = false;
  if (!kNoCopy && bytes && (src_sel == 1 || dst_sel == 1)) {
    copied =
        render::DmaGds(renderer, src_sel, dst_sel, src, dst, bytes, body[1]);
    TraceDmaData(control, body[5] & ~0x1fffffu, src_sel, dst_sel, src, dst,
                 bytes, copied);
    return;
  }
  if (!kNoCopy && src_is_memory && dst_is_memory && bytes &&
      bytes <= 0x1000000u && src != dst && addressable(src) &&
      addressable(src + bytes) && addressable(dst) &&
      addressable(dst + bytes) && GpuWriteMapped("DMA_DATA", dst, bytes) &&
      host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(src), bytes)) {
    NoteGpuWriteNear("DMA_DATA", dst, bytes);
    if (render::GuestBytesSettled(src, bytes) &&
        render::GuestBytesSettled(dst, bytes)) {
      std::memcpy(reinterpret_cast<void*>(dst),
                  reinterpret_cast<const void*>(src), bytes);
      copied = true;
    } else {
      copied = render::ApplyMemoryCopy(renderer, dst, src, bytes);
    }
  }
  if (!kNoCopy && src_sel == 2 && dst_is_memory && bytes &&
      bytes <= 0x8000000u && addressable(dst) && addressable(dst + bytes) &&
      GpuWriteMapped("DMA_DATA fill", dst, bytes)) {
    render::ApplyMemoryFill(renderer, dst, bytes, body[1]);
    copied = true;
  }
  TraceDmaData(control, body[5] & ~0x1fffffu, src_sel, dst_sel, src, dst, bytes,
               copied);
  TraceAddrWatch("DMA_DATA", dst, bytes, body[1], 8);
}

void HandleWriteData(const u32* body, u32 count) {
  if (count >= 3)
    TraceAddrWatch("WRITE_DATA",
                   (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1],
                   (count - 3) * 4u, count >= 4 ? body[3] : 0, 12);
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u32 dwords = count - 3;
  NoteGpuWriteNear("WRITE_DATA", address, body[3]);
  if (IsLabelAddress(address) && IsLabelAddress(address + (u64)dwords * 4) &&
      GpuWriteMapped("WRITE_DATA", address, (u64)dwords * 4)) {
    g_fence_labels.Note(address);
    const auto write = [address,
                        words = base::Vector<u32>(body + 3, body + count)] {
      std::memcpy(reinterpret_cast<void*>(address), words.data(),
                  words.size() * 4);
    };
    // Shadow of the Colossus frees a compute buffer once the WRITE_DATA after
    // its dispatch lands; published early, the dispatch's writes landed on
    // the heap's free-tree links instead.
    if (HeldWriteOverlaps(address, (u64)dwords * 4) ||
        render::GuestWritesPending())
      WriteAfterGpu(write);
    else
      write();
  }
  TraceDataWrite(address, dwords, dwords ? body[3] : 0);
}

void HandleEventWriteEop(const u32* body, u32 count) {
  if (count >= 3)
    TraceAddrWatch("EVENT_WRITE_EOP",
                   (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1], 8,
                   count >= 4 ? body[3] : 0, 8);
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const u64 value = static_cast<u64>(body[3]) |
                    (static_cast<u64>(count >= 5 ? body[4] : 0) << 32);
  WriteEventLabel("EOP", address, (body[2] >> 29) & 0x7, value,
                  (body[2] >> 24) & 0x3);
}

void HandleReleaseMem(const u32* body, u32 count) {
  if (count >= 4)
    TraceAddrWatch("RELEASE_MEM",
                   (static_cast<u64>(body[3] & 0xFFFF) << 32) | body[2], 8,
                   count >= 5 ? body[4] : 0, 8);
  if (count < 5)
    return;
  const u64 address =
      (static_cast<u64>(body[3] & 0xFFFF) << 32) | (body[2] & ~0x3u);
  const u64 value = static_cast<u64>(body[4]) |
                    (static_cast<u64>(count >= 6 ? body[5] : 0) << 32);
  WriteEventLabel("RELEASE_MEM", address, (body[1] >> 29) & 0x7, value,
                  (body[1] >> 24) & 0x7);
}

void HandleEventWrite(const u32* body, u32 count) {
  if (count < 3)
    return;
  const u32 event_index = (body[0] >> 8) & 0xF;
  if (event_index < 1 || event_index > 3)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x7u);
  TraceAddrWatch("EVENT_WRITE", address, 8, body[0], 8);
  if (!IsLabelAddress(address) || !IsLabelAddress(address + 8))
    return;
  constexpr u32 kRenderBackends = 8;
  const u64 span = static_cast<u64>(kRenderBackends) * 16;
  if (!IsGuestRange(address, span) ||
      !host_memory::IsMemoryRangeMapped(reinterpret_cast<const void*>(address),
                                        span))
    return;
  static base::Atomic<u64> samples{0};
  const u64 value = (1ull << 63) | (samples.fetch_add(1) + 1);
  g_fence_labels.Note(address);
  WriteAfterGpu(
      [=] {
        for (u32 rb = 0; rb < kRenderBackends; rb++)
          WriteLabel(address + rb * 16, value, true);
        TraceLabelWrite("EVENT_WRITE", address, 2, value);
      },
      address);
}

void HandleEventWriteEos(const u32* body, u32 count) {
  if (count >= 3)
    TraceAddrWatch("EVENT_WRITE_EOS",
                   (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1], 8,
                   count >= 4 ? body[3] : 0, 8);
  if (count < 4)
    return;
  const u64 address =
      (static_cast<u64>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  WriteAfterGpu(
      [address, value = body[3]] {
        WriteLabel(address, value, false);
        TraceEosLabel(address, value);
      },
      address, 4, body[3]);
}

void HandleDrawPacket(render::Renderer& renderer,
                      u32 op,
                      const u32* body,
                      u32 count) {
  DELTA_ZONE("ps4.draw_packet");
  g_total_draws.fetch_add(1);
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
    packet.index_max = g_index.max;
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

void NoteLateLabel(u64 address, const char* packet);

void InOrder(void (*handle)(const u32*, u32),
             const u32* body,
             u32 count,
             u64 label,
             u64 label_bytes) {
  NoteLateLabel(label, handle == HandleWriteData ? "WRITE_DATA" : "label");
  RenderQueue& queue = GuestRenderQueue();
  if (!queue.running()) {
    handle(body, count);
    return;
  }
  if (label && IsLabelAddress(label)) {
    g_fence_labels.Note(label);
    const u32 sel =
        handle == HandleEventWriteEop && count >= 4 ? (body[2] >> 29) & 0x7
        : handle == HandleReleaseMem && count >= 5  ? (body[1] >> 29) & 0x7
                                                    : 0;
    const u32 at = handle == HandleEventWriteEop ? 3 : 4;
    if (handle == HandleEventWriteEos && count >= 4)
      queue.NotePendingValue(label, 4, body[3]);
    else if (handle == HandleWriteData && (count == 4 || count == 5))
      queue.NotePendingValue(
          label, (count - 3) * 4,
          body[3] | (count == 5 ? static_cast<u64>(body[4]) << 32 : 0));
    else if (sel == 1 || (sel == 2 && count > at + 1))
      queue.NotePendingValue(
          label, sel * 4,
          body[at] | (sel == 2 ? static_cast<u64>(body[at + 1]) << 32 : 0));
    else
      queue.NotePendingWrite(label, label_bytes);
  }
  queue.PushCall([handle, words = base::Vector<u32>(body, body + count)] {
    handle(words.data(), static_cast<u32>(words.size()));
  });
}

u64 PacketAddress(const u32* body, u32 at) {
  return ((static_cast<u64>(body[at + 1] & 0xFFFF) << 32) | body[at]) & ~3ull;
}

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

struct FailedWait {
  u64 address = 0;
  base::TimeTicks when;
};
FailedWait g_failed_waits[2048];
u32 g_failed_next = 0;
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
            depth, (unsigned long long)address, body[0] & 7, body[3], value,
            WaitRegMemPasses(body, value) ? "pass" : "FAIL",
            g_fence_labels.Contains(address) ? "ours" : "foreign", last);
}

bool IbChains(const u32* body, u32 count) {
  const bool chains = count >= 3 && ((body[2] >> 20) & 1);
  if (chains) {
    static base::Atomic<u64> n{0};
    if (n.fetch_add(1) == 0)
      BASE_LOGI("gpu", "first chained indirect buffer");
  }
  return chains;
}

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
        break;
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
    case IT_WRITE_CONST_RAM: {
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
    case IT_LOAD_CONST_RAM: {
      if (!kCeOn || count < 4)
        break;
      {
        static int shown = 0;
        if (CeTraceOn() && shown < 6) {
          shown++;
          BASE_LOGI("ce", "load raw count={} body={:08x} {:08x} {:08x} {:08x}",
                    count, body[0], body[1], body[2], count > 3 ? body[3] : 0u);
        }
      }
      const u64 address = (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
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
    case IT_DUMP_CONST_RAM_OFFSET: {
      if (!kCeOn || count < 4)
        break;
      const u32 offset = body[0] & 0xFFFF, dwords = body[1] & 0x7FFF;
      const u64 address = (static_cast<u64>(body[3] & 0xFFFF) << 32) | body[2];
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
    case IT_INCREMENT_CE_COUNTER:
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
  RunCe(renderer, [] { return false; }, true);
  g_ce.packets.clear();
  g_ce.pos = 0;
}

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
      i += 1;
      continue;
    }
    if (type == Pm4Type::kType0) {
      const u32 count = Pm4Count(hdr);
      const u32 base = Pm4Type0Reg(hdr);
      const u32 available = base::Min(count, words - i - 1);
      for (u32 k = 0; k < available; k++)
        if (base + k < kRegFileSize)
          g_regs[base + k] = p[i + 1 + k];
      NoteRegisterWrites(base, &p[i + 1], available, 0);
      i += 1 + count;
      continue;
    }
    if (type != Pm4Type::kType3) {
      TraceDesync(i, words, static_cast<u32>(type), hdr, dump);
      break;
    }

    const u32 op = Pm4Opcode(hdr);
    const u32 count = Pm4Count(hdr);
    const u32* body = &p[i + 1];
    NotePacket(op);
    const auto op_start =
        time_packets ? base::TimeTicks::Now() : base::TimeTicks{};
    if (dump)
      TraceDcbPacket(i, op, count);
    if (i + 1 + count > words)
      break;
    switch (op) {
      case IT_DISPATCH_DIRECT:
        DispatchComputeDirect(renderer, g_regs, body, count);
        break;
      case IT_DISPATCH_INDIRECT:
        DispatchComputeIndirect(renderer, g_regs, body, count,
                                g_index.dispatch_indirect_base);
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
      case IT_INDEX_BASE:
        if (count >= 2)
          g_index.base = (static_cast<u64>(body[1] & 0xFF) << 32) | body[0];
        break;
      case IT_SET_BASE:
        if (count >= 3 && (body[0] & 0xF) == 1) {
          const u64 base =
              (static_cast<u64>(body[2] & 0xFF) << 32) | (body[1] & ~0x3u);
          if (hdr & 2)
            g_index.dispatch_indirect_base = base;
          else
            g_index.indirect_base = base;
        }
        break;
      case IT_INDEX_BUFFER_SIZE:
        if (count >= 1)
          g_index.max = body[0];
        break;
      case IT_NUM_INSTANCES:
        g_index.num_instances = (count >= 1 && body[0]) ? body[0] : 1;
        break;
      case IT_WAIT_REG_MEM:
        do {
          if (count >= 5 && GuestRenderQueue().running()) {
            const volatile u32* polled = WaitRegMemTarget(body);
            u32 queued = 0;
            if (polled &&
                GuestRenderQueue().PendingWriteValue(
                    reinterpret_cast<u64>(polled), &queued) &&
                WaitRegMemPasses(body, queued))
              break;
          }
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
            u32 held = 0;
            if (WaitRegMemPasses(body, *polled) ||
                (HeldWriteValue(reinterpret_cast<u64>(polled), &held) &&
                 WaitRegMemPasses(body, held)))
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
        if (count == 4 || count == 5)
          GuestRenderQueue().NotePendingValue(
              PacketAddress(body, 1), (count - 3) * 4,
              body[3] | (count == 5 ? static_cast<u64>(body[4]) << 32 : 0));
        else if (count > 5)
          GuestRenderQueue().NotePendingWrite(PacketAddress(body, 1),
                                              (count - 3) * 4ull);
        InOrder(HandleWriteData, body, count,
                count >= 3 ? PacketAddress(body, 1) : 0,
                count >= 4 ? (count - 3) * 4ull : 4);
        break;
      case IT_EVENT_WRITE_EOP:
        InOrder(HandleEventWriteEop, body, count,
                count >= 3 ? PacketAddress(body, 1) : 0, 8);
        break;
      case IT_RELEASE_MEM:
        InOrder(HandleReleaseMem, body, count,
                count >= 4 ? PacketAddress(body, 2) : 0, 8);
        break;
      case IT_EVENT_WRITE:
        InOrder(HandleEventWrite, body, count,
                count >= 3 ? PacketAddress(body, 1) & ~7ull : 0, 128);
        break;
      case IT_EVENT_WRITE_EOS:
        InOrder(HandleEventWriteEos, body, count,
                count >= 3 ? PacketAddress(body, 1) : 0, 128);
        break;
      case IT_INDIRECT_BUFFER:
      case IT_INDIRECT_BUFFER_CNST: {
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
        RunCe(
            renderer,
            [] {
              const u32 ahead = (g_ce_counter - g_de_counter) & 0xFFFFFF;
              return ahead != 0 && ahead < 0x800000;
            },
            false);
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

struct RegUndo {
  u32 off, old;
};
base::Vector<RegUndo> g_prefetch_undo;

void PrefetchSet(Regs& regs, u32 off, u32 value) {
  if (off >= kRegFileSize || regs[off] == value)
    return;
  g_prefetch_undo.push_back({off, regs[off]});
  regs[off] = value;
}

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
        PrefetchSet(regs, base + k, p[i + 1 + k]);
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
      case IT_DISPATCH_INDIRECT:
        PrefetchComputeDispatch(regs);
        break;
      default:
        if (IsDraw(op))
          PrefetchDrawShaders(regs);
        break;
    }
    if (reg_base && count) {
      const u32 first = Pm4SetRegAddress(reg_base, body[0]);
      for (u32 k = 1; k < count; k++)
        PrefetchSet(regs, first + k - 1, body[k]);
    }
    i += 1 + count;
  }
}

struct ScopedWalkTimer {
  base::TimeTicks start = base::TimeTicks::Now();
  ~ScopedWalkTimer() {
    g_stats.ns_dcb +=
        ((base::TimeTicks::Now() - start).InMicroseconds() * 1000);
    render::g_dcb_n++;
  }
};

rhi::TextureView* BindlessTexture(u32 kind,
                                  const u32* words,
                                  rhi::TextureState* state) {
  render::DrawInfo::DrawTex t;
  FillDrawTex(t, gcn::DecodeTImage(words));
  t.arrayed = (kind & 1) != 0;
  return render::BindlessTextureView(t, state);
}

rhi::Sampler* BindlessSampler(u32 kind, const u32* words) {
  render::DrawInfo::DrawTex t;
  std::memcpy(t.sampler, words, sizeof(t.sampler));
  t.sampler_valid = true;
  t.depth_compare = (kind & 1) != 0;
  return render::SamplerForTex(t, rhi::Format::kRGBA8Unorm);
}

void StartRendererOnce(render::Renderer& renderer) {
  if (g_renderer_started)
    return;
  g_renderer_started = true;
  render::Init(renderer);
  render::bindless::SetResolvers(&BindlessTexture, &BindlessSampler);
  gcn::g_bindless = renderer.available() && render::bindless::Supported();
  render::RecordComputeInFrame();
  if (LabelsAtGpu())
    render::DeferLabelsToGpu();
  GuestWriteTracker().Enable();
  gcn::g_flush_guest_range = [](u64 address, u64 bytes) {
    SpeculateWalkRead(render::DefaultRenderer(), address, bytes);
  };
  gcn::g_overlay_guest_range = [](u64 address, u32 dwords, u32* values) {
    if (OverlayKnownGpuWrite(address, dwords * 4ull,
                             reinterpret_cast<u8*>(values)))
      return;
    if (GuestRenderQueue().running())
      GuestRenderQueue().OverlayPendingValues(address, dwords * 4ull,
                                              reinterpret_cast<u8*>(values));
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
  DELTA_ZONE("ps4.end_frame");
  base::LockGuard<base::Mutex> lock(g_mutex);
  gcn::NextProgramCacheGeneration();
  gpu::FrameMemoryGeneration();
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
  DELTA_ZONE("ps4.submit_ccb");
  if (!ccb || size_bytes < 4)
    return;
  base::LockGuard<base::Mutex> lock(g_mutex);
  const u32 words = size_bytes / 4;
  TraceCcbSubmit(size_bytes, words);
  FlattenCcb(static_cast<const u32*>(ccb), words, 0);
  TraceCcbHistogram(words);
}

void SubmitDcb(const void* dcb, u32 size_bytes) {
  DELTA_ZONE("ps4.submit_dcb");
  if (!dcb || size_bytes < 4)
    return;
  ScopedWalkTimer timer;
  const auto lock_start = base::TimeTicks::Now();
  base::LockGuard<base::Mutex> lock(g_mutex);
  g_stats.ns_dcb_lock +=
      ((base::TimeTicks::Now() - lock_start).InMicroseconds() * 1000);

  render::Renderer& renderer = render::DefaultRenderer();
  StartRendererOnce(renderer);
  if (gcn::VsProbeShader() && !gcn::g_vs_probe && renderer.available()) {
    OwnRenderer("vs-probe");
    render::trace::InitVsProbe();
  }

  if (gcn::DirectGcnEnabled() && renderer.available()) {
    if (!gcn::g_direct_table) {
      OwnRenderer("direct-init");
      render::InitGuestDirect();
    }
    if (GuestRenderQueue().running())
      GuestRenderQueue().PushCall(&render::SyncGuestDirect);
    else
      render::SyncGuestDirect();
  }

  const auto* p = static_cast<const u32*>(dcb);
  const u32 words = size_bytes / 4;
  const u64 submission = g_total_submits.fetch_add(1) + 1;
  TraceSubmit(dcb, size_bytes, words, submission, g_total_draws.load());
  const bool dump = ShouldDumpDcb(size_bytes);
  TraceDcbStat(words);

  if (kPrefetchShaders && renderer.available()) {
    PrefetchWalk(g_regs, p, words, 0);
    for (u32 k = g_prefetch_undo.size(); k-- > 0;)
      g_regs[g_prefetch_undo[k].off] = g_prefetch_undo[k].old;
    g_prefetch_undo.clear();
  }
  const u32 walked = WalkDcb(renderer, p, words, 0, dump);
  FinishCe(renderer);
  if (GuestRenderQueue().running())
    GuestRenderQueue().PushCall(&SubmitHeldWrites);
  else
    SubmitHeldWrites();

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

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_total_submits);
  guest::ResetResource(g_total_draws);
  guest::ResetResource(g_presented_frames);
  guest::ResetResource(g_ce_counter);
  guest::ResetResource(g_de_counter);
  guest::ResetResource(g_fence_labels);
  guest::ResetResource(g_label_writes);
  guest::ResetResource(g_ce);
  guest::ResetResource(g_regs);
  guest::ResetResource(g_renderer_started);
  guest::ResetResource(g_frame_active);
  guest::ResetResource(g_index);
  guest::ResetResource(g_const_ram);
  guest::ResetResource(g_failed_waits);
  guest::ResetResource(g_failed_next);
});
}  // namespace

void StopRenderQueue() {
  GuestRenderQueue().Stop();
}

void WaitForFrame() {
  GuestRenderQueue().Drain("display flip");
}

}  // namespace gpu::ps4
