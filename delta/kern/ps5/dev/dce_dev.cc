#include "kern/ps5/dev/dce_dev.h"

#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include "base/logging.h"
#include "base/time/time.h"
#include "guest/display.h"
#include "guest/session.h"
#include "host/window.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event.h"
#include "kern/lv2/sys_mem.h"
#include "options/options.h"

extern "C" void prosperity_agc_flip(u64 scanout_base);

namespace kern {
namespace {
DELTA_OPTION(bool, kDceTracePs5, "DELTA_DCE_PS5_TRACE", false);
constexpr size_t kPoolSize = 0x4000;
// PS5 uses pointer/length payloads inside these input-only ioctls.
struct Query {
  u32 op;
  u32 port;
  u32 flags;
  u32 size;
  u64 data;
  u64 value;
};
struct Request {
  u32 op;
  u32 flags;
  u32 handle;
  u32 size;
  u64 data;
  u64 value;
  u64 result;
  u64 result_size;
};
struct Port {
  u8* pool = nullptr;
  u64 buffers[16]{};
  u64 scanout = 0;
  u64 count = 0;
  u64 time = 0;
  u64 tsc = 0;
  u64 submit_tsc = 0;
  i64 arg = 0;
  i32 index = -1;
  u32 next_context = 0;
};
struct PendingFlip {
  u64 context, scanout;
  int index, mode;
  i64 arg;
  bool ready = false;
};
std::deque<PendingFlip> g_pending_flips;
u64 GuestTsc() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  return static_cast<u64>(base::TickClock::NowNs()) * 16 / 10;
#endif
}
std::mutex g_mutex;
Port g_port;
const guest::SessionReset g_reset([] {
  std::lock_guard lock(g_mutex);
  g_port = {};
  g_pending_flips.clear();
});
}  // namespace

bool DcePs5Active() {
  std::lock_guard lock(g_mutex);
  return g_port.pool != nullptr;
}

u64 DcePs5ScanoutBuffer() {
  std::lock_guard lock(g_mutex);
  return g_port.scanout;
}

bool DcePs5IsDisplayBuffer(u64 address) {
  if (!address)
    return false;
  std::lock_guard lock(g_mutex);
  for (u64 buffer : g_port.buffers)
    if (buffer == address)
      return true;
  return false;
}

bool DcePs5BeginEopFlip(u64 context, u64* scanout) {
  std::lock_guard lock(g_mutex);
  for (auto& flip : g_pending_flips) {
    if (flip.context == context && !flip.ready) {
      flip.ready = true;
      *scanout = flip.scanout;
      return true;
    }
  }
  return false;
}

void DcePs5CompleteEopFlip(u64 context) {
  PendingFlip flip{};
  u8* pool;
  {
    std::lock_guard lock(g_mutex);
    auto it = g_pending_flips.begin();
    while (it != g_pending_flips.end() && it->context != context)
      ++it;
    if (it == g_pending_flips.end() || !it->ready)
      return;
    flip = *it;
    pool = g_port.pool;
    g_pending_flips.erase(it);
  }
  guest::display::WaitFlip(flip.mode, [=] {
    {
      std::lock_guard lock(g_mutex);
      g_port.scanout = flip.scanout;
      g_port.index = flip.index;
      g_port.arg = flip.arg;
      g_port.time = static_cast<u64>(base::TickClock::NowNs());
      g_port.tsc = GuestTsc();
      ++g_port.count;
      if (flip.index >= 0)
        reinterpret_cast<volatile u64*>(pool + 0xa0)[flip.index] = 0;
    }
    NoteFlip();
  });
}

DceDevicePs5::DceDevicePs5(ObjectTable& objects) : Device(objects) {}

bool DceDevicePs5::Init(const char*, u32, u32) {
  std::lock_guard lock(g_mutex);
  pool_ = g_port.pool ? g_port.pool : AllocLowGuest(kPoolSize);
  g_port.pool = pool_;
  return pool_ != nullptr;
}

u8* DceDevicePs5::Map(void*, size_t size, u32, u32, size_t offset) {
  return pool_ && offset < kPoolSize && size <= kPoolSize - offset
             ? pool_ + offset
             : reinterpret_cast<u8*>(-1);
}

i32 DceDevicePs5::Ioctl(u32 command, void* args) {
  if (!args)
    return -static_cast<i32>(SysError::eFAULT);
  if (command == 0x80208218) {
    auto& q = *static_cast<Query*>(args);
    auto* out = reinterpret_cast<u32*>(q.data);
    if (q.size > 0x100 || !host_memory::IsMemoryRangeMapped(out, q.size))
      return -static_cast<i32>(SysError::eFAULT);
    if (q.op == 1 && q.size == 8) {
      out[0] = 1;
      out[1] = 0;
      return 0;
    }
    if (q.op == 0xd && q.size == 0x30) {
      std::memset(out, 0, q.size);
      out[0] = out[2] = 1920;
      out[1] = out[3] = 1080;
      reinterpret_cast<u64*>(out)[3] = 60;
      return 0;
    }
    return -static_cast<i32>(SysError::eINVAL);
  }
  if (command != 0x80308217)
    return -static_cast<i32>(SysError::eINVAL);
  auto& r = *static_cast<Request*>(args);
  auto* p = reinterpret_cast<u64*>(r.data);
  if (r.handle != 1 || !pool_)
    return -static_cast<i32>(SysError::eINVAL);
  if (r.size > 0x100 || r.result_size > 8)
    return -static_cast<i32>(SysError::eINVAL);
  if ((r.size && !host_memory::IsMemoryRangeMapped(p, r.size)) ||
      (r.result_size && !host_memory::IsMemoryRangeMapped(
                            reinterpret_cast<void*>(r.result), r.result_size)))
    return -static_cast<i32>(SysError::eFAULT);
  if (kDceTracePs5) {
    static std::atomic<u32> calls[256]{};
    if (r.op < 256 && calls[r.op].fetch_add(1) < 4) {
      BASE_LOGI("dce5", "op={:#x} handle={} size={:#x} value={:#x}", r.op,
                r.handle, r.size, r.value);
      for (u32 i = 0; p && i < r.size / 8; ++i)
        BASE_LOGI("dce5", "  [{}] {:#018x}", i, p[i]);
    }
  }
  switch (r.op) {
    case 9:
      if (r.size != 8 || r.result_size != 8)
        return -static_cast<i32>(SysError::eINVAL);
      // video output client caches this address and adds 0xa0 for its labels.
      *p = reinterpret_cast<u64>(pool_);
      *reinterpret_cast<u64*>(r.result) = kPoolSize;
      break;
    case 0x38:
      if (r.size != 0x60 || !static_cast<u32>(p[3]) || !(p[3] >> 32))
        return -static_cast<i32>(SysError::eINVAL);
      host::Init("prosperity", static_cast<u32>(p[3]),
                 static_cast<u32>(p[3] >> 32));
      break;
    case 4: {
      if (r.size || r.value >= 16)
        return -static_cast<i32>(SysError::eINVAL);
      std::lock_guard lock(g_mutex);
      if (g_port.scanout == g_port.buffers[r.value])
        g_port.scanout = 0;
      g_port.buffers[r.value] = 0;
      reinterpret_cast<u64*>(pool_ + 0xa0)[r.value] = 0;
      break;
    }
    case 5:
      if (r.size || r.value >= 16)
        return -static_cast<i32>(SysError::eINVAL);
      break;
    case 6:
      if (r.value > 2)
        return -static_cast<i32>(SysError::eINVAL);
      guest::display::SetFlipRate(static_cast<int>(r.value));
      break;
    case 0x37: {
      if (r.value >= 16 || r.size != 0x30)
        return -static_cast<i32>(SysError::eINVAL);
      std::lock_guard lock(g_mutex);
      g_port.buffers[r.value] = p[2];
      if (!g_port.scanout)
        g_port.scanout = p[2];
      break;
    }
    case 0x36: {
      if (r.size != 0x48)
        return -static_cast<i32>(SysError::eINVAL);
      const u32 flip_mode = static_cast<u32>(p[2]);
      const u32 submission = static_cast<u32>(p[2] >> 32);
      const bool blank = p[1] == ~u64(0) || p[1] == ~u32(0);
      if ((!blank && p[1] >= 16) || !flip_mode || flip_mode > 9 ||
          flip_mode == 7 ||
          (submission != 0 && submission != 1 && submission != 5 &&
           submission != 9))
        return -static_cast<i32>(SysError::eINVAL);
      const int index = blank ? -1 : static_cast<int>(p[1]);
      const int mode = (flip_mode - 1) % 3 == 1 ? 2 : 1;
      const i64 arg = static_cast<i64>(p[3]);
      auto* label = reinterpret_cast<u64*>(p[6]);
      if (label && !host_memory::IsMemoryRangeMapped(label, sizeof(*label)))
        return -static_cast<i32>(SysError::eFAULT);
      u64 scanout;
      {
        std::lock_guard lock(g_mutex);
        scanout = blank ? 0 : g_port.buffers[index];
        if (!blank && !scanout)
          return -static_cast<i32>(SysError::eINVAL);
        if (p[5] == 1) {
          if (!label)
            return -static_cast<i32>(SysError::eINVAL);
          const u64 context = ++g_port.next_context;
          g_pending_flips.push_back({context, scanout, index, mode, arg});
          *label = context;
          if (r.result && r.result_size == 8)
            *reinterpret_cast<u64*>(r.result) = 0;
          g_port.submit_tsc = GuestTsc();
          break;
        }
        g_port.scanout = scanout;
        g_port.submit_tsc = GuestTsc();
      }
      if (!blank)
        prosperity_agc_flip(scanout);
      if (r.result && r.result_size == 8)
        *reinterpret_cast<u64*>(r.result) = 0;
      // AGC passes a stack label, so completion must precede returning.
      if (!guest::display::WaitFlip(mode, [=] {
            {
              std::lock_guard lock(g_mutex);
              g_port.index = index;
              g_port.arg = arg;
              g_port.time = static_cast<u64>(base::TickClock::NowNs());
              g_port.tsc = GuestTsc();
              ++g_port.count;
            }
            if (label)
              *label = 1;
            NoteFlip();
          }))
        return -static_cast<i32>(SysError::eINVAL);
      break;
    }
    case 0xa: {
      if (r.size != 0x80)
        return -static_cast<i32>(SysError::eINVAL);
      std::lock_guard lock(g_mutex);
      std::memset(p, 0, r.size);
      p[0] = g_port.arg;
      p[2] = g_port.count;
      p[3] = g_port.time;
      p[4] = g_port.tsc;
      p[7] = g_port.submit_tsc;
      reinterpret_cast<i32*>(p)[10] = g_port.index;
      break;
    }
    case 0xb:
      if (r.size < 0x20)
        return -static_cast<i32>(SysError::eINVAL);
      std::memset(p, 0, r.size);
      p[0] = guest::display::VblankCount();
      p[1] = guest::display::VblankTimeNs();
      p[2] = GuestTsc();
      break;
    case 0xc:
      if (r.size != 0x40)
        return -static_cast<i32>(SysError::eINVAL);
      std::memset(p, 0, r.size);
      break;
    case 0x33:
      return r.value == reinterpret_cast<u64>(pool_ + 0xa0)
                 ? 0
                 : -static_cast<i32>(SysError::eINVAL);
    case 0x40:
      break;
    default:
      return -static_cast<i32>(SysError::eINVAL);
  }
  return 0;
}
}  // namespace kern

extern "C" bool prosperity_ps5_is_display_buffer(u64 address) {
  return kern::DcePs5IsDisplayBuffer(address);
}

extern "C" bool prosperity_ps5_begin_eop_flip(u64 context, u64* scanout) {
  return kern::DcePs5BeginEopFlip(context, scanout);
}

extern "C" void prosperity_ps5_complete_eop_flip(u64 context) {
  kern::DcePs5CompleteEopFlip(context);
}
