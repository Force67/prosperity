#include "kern/ps4/audio_in.h"

#include <cstdio>
#include <cstring>

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/strings/number_parse.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "guest/session.h"
#include "kern/lv2/sys_event_flag.h"

namespace kern::ps4 {
namespace {
constexpr size_t kPorts = 8, kHeader = 0x78, kStride = 0x90;
constexpr size_t kExtent = kHeader + kPorts * kStride;
constexpr i32 kInvalidHandle = static_cast<i32>(0x80260101u);
constexpr i32 kNotOpen = static_cast<i32>(0x80260109u);

struct Control {
  u8* base = nullptr;
  size_t size = 0;
  u32 open = 0;
  u64 due[kPorts]{};
};
base::Mutex g_mutex;
base::HashMap<u32, Control> g_controls;
base::Atomic<bool> g_started{false};

void CaptureMain() {
  for (;;) {
    const u64 now = base::TickClock::NowNs() / 1000;
    {
      base::LockGuard<base::Mutex> lock(g_mutex);
      for (auto& [pid, control] : g_controls) {
        if (!control.base || control.size < kExtent)
          continue;
        for (u32 index = 0; index < kPorts; ++index) {
          if (!(control.open & (1u << index)) || now < control.due[index])
            continue;
          auto* slot = control.base + kHeader + index * kStride;
          auto word = [&](size_t at)->volatile u32& {
            return *reinterpret_cast<volatile u32*>(slot + at);
          };
          const u32 grain = word(8), rate = word(0x1c);
          const u32 channels = word(0x20);
          if (!grain || grain > 1024 || (rate != 16000 && rate != 48000) ||
              !channels || channels > 2)
            continue;
          // No input device is connected. The real library writes silence
          // when connection state is zero; it owns the destination buffer.
          word(0x28) = 0;
          word(0x2c) = grain * channels;
          word(0x0c) = 0;
          base::atomic_thread_fence(base::memory_order_release);
          char flag[32];
          std::snprintf(flag, sizeof(flag), "/evfAin%08x", pid);
          EvfSetByNameSubstr(flag, 1ull << index);
          const u64 period = u64(grain) * 1000000 / rate;
          control.due[index] += period;
          if (control.due[index] <= now)
            control.due[index] = now + period;
        }
      }
    }
    guest::SleepForMilliseconds(1);
  }
}
const guest::SessionReset g_reset([] {
  base::LockGuard<base::Mutex> lock(g_mutex);
  g_controls.clear();
  g_started.store(false);
});
}  // namespace

void AudioInNoticeShm(const char* name, u8* base, size_t size) {
  constexpr const char* prefix = "/shmAin";
  if (!name || std::strncmp(name, prefix, 7) || std::strlen(name) != 15)
    return;
  u64 pid;
  const char* end = nullptr;
  if (!base::ParseUnsigned(name + 7, pid, 16, &end) || *end || pid > UINT32_MAX)
    return;
  {
    base::LockGuard<base::Mutex> lock(g_mutex);
    auto& control = g_controls[pid];
    if (control.base != base || size < kExtent)
      control = {};
    control.base = base;
    control.size = size;
  }
  if (base && size >= kExtent && !g_started.exchange(true))
    guest::SpawnThread("audio_capture", CaptureMain);
}

void AudioInForgetRange(const void* base, size_t size) {
  const auto first = reinterpret_cast<uintptr_t>(base);
  base::LockGuard<base::Mutex> lock(g_mutex);
  for (auto it = g_controls.begin(); it != g_controls.end();) {
    const auto other = reinterpret_cast<uintptr_t>(it->second.base);
    if (first < other + it->second.size && other < first + size)
      it = g_controls.erase(it);
    else
      ++it;
  }
}

i32 SetAudioInPort(u32 pid, u32 handle, bool open) {
  const u32 index = handle & 0xff;
  if ((handle & 0x7f000000) != 0x30000000 || index >= kPorts)
    return kInvalidHandle;
  base::LockGuard<base::Mutex> lock(g_mutex);
  auto it = g_controls.find(pid);
  if (it == g_controls.end() || !it->second.base || it->second.size < kExtent)
    return kNotOpen;
  auto& control = it->second;
  char flag[32];
  std::snprintf(flag, sizeof(flag), "/evfAin%08x", pid);
  EvfClearByNameSubstr(flag, ~(1ull << index));
  if (open) {
    control.open |= 1u << index;
    const auto* slot = control.base + kHeader + index * kStride;
    u32 grain, rate;
    std::memcpy(&grain, slot + 8, 4);
    std::memcpy(&rate, slot + 0x1c, 4);
    control.due[index] = base::TickClock::NowNs() / 1000 +
                         (rate ? u64(grain) * 1000000 / rate : 0);
  } else {
    control.open &= ~(1u << index);
    control.due[index] = 0;
  }
  return 0;
}
}  // namespace kern::ps4
