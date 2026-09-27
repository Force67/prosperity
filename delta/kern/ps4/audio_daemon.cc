/*
 * PS4Delta : PS4 emulation and research project
 *
 * System audio daemon stand-in for LLE libSceAudioOut. See audio_daemon.h.
 *
 * THE PROTOCOL (reverse-engineered from the 11.00 module and cross-checked
 * against live memory; the field table below is the part that is established –
 * see the "unverified" notes for the part that is not).
 *
 * On the first sceAudioOutOpen the module creates:
 *
 *   "/shm_<pid>_C"          the control block. 0x20-byte header followed by 26
 *                           port slots of 0x250 bytes; slot k at 0x20+k*0x250.
 *                           NOTE: it is ftruncate'd to only 0xf28 and relies on
 *                           page rounding, so never size it from the ftruncate
 *                           length; the real extent is 0x20+26*0x250 = 0x3c40.
 *   "/shm_<pid>_<idx>_A"    one per open port, always 0x10000 bytes, holding
 *                           exactly ONE block of samples at offset 0. Not a
 * ring and not cursor'd: submit() memcpy's to offset 0 every time. 0x10000 is
 * just the worst case (2048 grain x 8ch x 4B).
 *
 * Per-slot fields this consumer uses:
 *   +0x00 u64  handshake token. The guest writes the slot's own guest address
 *              here after filling the sample region; its submit() returns BUSY
 *              (0x80260002) for as long as it is non-zero. THE DAEMON MUST ZERO
 *              IT: that single store is the whole reason LLE was silent.
 *   +0x08 u32  bytes per frame (2/4/8/12/16/32)
 *   +0x0c 8xf  per-channel gain, 1.0 == SCE volume 32768 == 0 dB
 *   +0x2c u32  sample type: 0 = int16, 1 = float32, 2 = 32-bit
 *   +0x34 u32  sample rate (the module only ever accepts 48000)
 *   +0x50 u32  port index, same as <idx> in the shm name and the event-flag bit
 *   +0x60 u32  grain: frames per block, multiple of 256 in 256..2048
 *   +0x90 u32  port state (3 once a block has been submitted)
 *
 * Control flow: the FIRST sceAudioOutOutput submits without asking. Every later
 * one finds the token still set, gets BUSY, and parks in
 * sceKernelWaitEventFlag("sceAudioOutMix<pid>", 1<<idx, AND|CLEARPAT) with an
 * infinite timeout. The module imports no SetEventFlag: setting bit idx is the
 * daemon's "block taken, you may write again" grant.
 *
 * So the daemon owes each port exactly three things, in this order:
 *   1. copy grain*bytesPerFrame bytes out of the head of the _A region
 *   2. write 0 to slot+0x00
 *   3. set bit idx of the mix event flag
 * and it must PACE that at grain/rate seconds per port, because on hardware the
 * back-pressure of step 3 is what keeps the title's audio thread in real time.
 * Consume as fast as the guest can produce and it free-runs.
 *
 * NOT ESTABLISHED, deliberately not guessed here:
 *   - sample type has only ever been observed as 0 (both Isaac ports are S16).
 *     1/float32 is implemented from the module's own format table but has never
 *     run; 2/32-bit is a real format (the kernel's PCM path supports 32-bit)
 *     but the host sink has no mapping for it, so such a port is drained (the
 *     handshake is still completed, so the title cannot wedge) and not played.
 *   - whether the per-channel gains at +0x0c are the consumer's job. The
 * samples are memcpy'd verbatim, so they almost certainly are; we apply them as
 * one uniform gain (the sink has no per-channel control). Isaac's are all 1.0,
 *     so this path is unexercised.
 *   - multi-port mixing. Hardware sums all open ports into one device; we give
 *     each guest port its own sink port, which is exactly what the HLE shim
 * does for the same title, so the two paths stay comparable.
 *   - slot +0x30, +0x38, +0x3c, +0x54, +0x58, +0x5c, +0x84, +0x8c and the two
 *     header words are read by nothing here; they are zero (or constant) for
 *     every port observed and their meaning is unknown.
 */

#include "kern/ps4/audio_daemon.h"
#include "base/arch.h"

#include "base/logging.h"
#include "base/strings/number_parse.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/ps4/audio_sink.h"
#include "kern/ps5/audio_queue.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kAudioDaemon, "DELTA_AUDIO_DAEMON", true);
DELTA_OPTION(bool, kAudioTrace, "DELTA_AUDIO_TRACE", false);
DELTA_OPTION(bool, kAudiomixAck, "DELTA_AUDIOMIX_ACK", false);
}  // namespace

namespace krnl {
namespace {

constexpr size_t kCtlHdr = 0x20;
constexpr size_t kSlotStride = 0x250;
constexpr size_t kSlots = 26;  // handle check is `idx <= 0x19`
constexpr size_t kCtlExtent = kCtlHdr + kSlots * kSlotStride;

enum : size_t {
  kOffToken = 0x00,
  kOffBpf = 0x08,
  kOffGain = 0x0c,
  kOffType = 0x2c,
  kOffRate = 0x34,
  kOffIndex = 0x50,
  kOffGrain = 0x60,
  kOffState = 0x90,
};

// The mix flag is opened by name as "sceAudioOutMix<pid>", but the module
// formats that pid differently from the one in the shm names (observed:
// "sceAudioOutMix1337" alongside "/shm_4919_C"), so match the stem instead of
// reconstructing a name we would have to guess the formatting of.
constexpr const char* kMixFlagStem = "sceAudioOutMix";

struct Region {
  u8* base = nullptr;
  size_t size = 0;
};

struct Port {
  int sink = -1;  // AudioSink handle
  u32 bpf = 0, type = 0, rate = 0, grain = 0, channels = 0;
  float gain = 1.f;
  u64 next_due_us = 0;
  u64 blocks = 0, dropped = 0;
  float peak = 0.f, peak_max = 0.f;
  bool configured = false;      // sink decision taken for the current shape
  bool bad_format = false;      // logged once
  bool warned_no_area = false;  // logged once
};

base::Mutex g_m;
Region g_ctl;
base::HashMap<int, Region> g_area;
base::Atomic<bool> g_started{false};

// AudioOut2 uses monotonically increasing producer/consumer block counters
// (unlike AudioOut's single-buffer token); offsets from the 01.14.00 module,
// checked against Skyrim's queues. Sample layout not decoded yet, so this is a
// paced null sink that releases only submitted blocks.
struct Ps5Container {
  Region region;
  base::String flag;
  u32 index = 0;
  u64 due = 0;
  u64 blocks = 0;
};
base::HashMap<base::String, Ps5Container> g_ps5_containers;

u64 DrainPs5(u64 now) {
  base::LockGuard<base::Mutex> lk(g_m);
  u64 wait = 2000;
  for (auto& [name, c] : g_ps5_containers) {
    if (now < c.due) {
      wait = base::Min(wait, c.due - now);
      continue;
    }
    const u32 frames = ps5::ConsumeAudioOut2Block(c.region.base, c.region.size);
    if (!frames)
      continue;
    // AudioOut2's container clock is 48 kHz. One counter step releases one
    // grain across all the container's channel ports.
    EvfSetByNameSubstr(c.flag.c_str(), 1ull << c.index);
    c.due = now + u64(frames) * 1000000 / 48000;
    if (!c.blocks++)
      BASE_LOGI("audiod",
                "AudioOut2 container {}: paced null sink, {} frames "
                "per block (playback not implemented)",
                name.c_str(), frames);
  }
  return wait;
}

bool TraceOn() {
  return kAudioTrace;
}

// DELTA_AUDIO_DAEMON=0 turns the daemon off, leaving LLE libSceAudioOut silent
// (the pre-daemon behaviour), useful when bisecting an LLE hang, since the
// daemon is what unblocks the title's audio thread.
bool Enabled() {
  return kAudioDaemon;
}

u64 NowUs() {
  return (u64)(base::TickClock::NowNs() / 1000);
}

// A slot is only believed if every field it declares is one the module could
// have written. This is what keeps the daemon from acting on some other
// subsystem's shm that happens to end in "_C": a mis-identified region reads as
// nonsense here and is left completely alone (no token clear, no event set).
bool PlausibleSlot(u32 bpf, u32 type, u32 rate, u32 grain) {
  if (rate != 48000)  // sceAudioOutOpen rejects every other frequency
    return false;
  if (grain < 256 || grain > 2048 || (grain & 0xFF) != 0)
    return false;
  if (type > 2)
    return false;
  // The module's format jump table yields whole frames only (s16: 2 B,
  // f32/32-bit: 4 B; 8 channels max); cross-checking rejects shapes it cannot
  // produce, e.g. bpf=32+s16 and bpf=2+f32, which the old value list let
  // through as nonsense sinks.
  const u32 bps = type == 0 ? 2u : 4u;
  if (bpf < bps || bpf % bps != 0 || bpf / bps > 8)
    return false;
  return true;
}

float BlockPeak(const u8* p, size_t bytes, u32 type) {
  float peak = 0.f;
  if (type == 1) {
    const float* s = reinterpret_cast<const float*>(p);
    for (size_t i = 0; i < bytes / 4; i++) {
      const float a = s[i] < 0 ? -s[i] : s[i];
      if (a > peak)
        peak = a;
    }
  } else {
    const i16* s = reinterpret_cast<const i16*>(p);
    for (size_t i = 0; i < bytes / 2; i++) {
      const float a = (s[i] < 0 ? -(float)s[i] : (float)s[i]) / 32768.f;
      if (a > peak)
        peak = a;
    }
  }
  return peak;
}

void DaemonMain() {
  Port ports[kSlots];
  base::Vector<u8> block;
  u64 last_trace = NowUs();

  for (;;) {
    Region ctl;
    base::HashMap<int, Region> areas;
    {
      base::LockGuard<base::Mutex> lk(g_m);
      ctl = g_ctl;
      areas = g_area;
    }

    const u64 now = NowUs();
    u64 sleep_us = DrainPs5(now);

    if (ctl.base && ctl.size >= kCtlExtent) {
      for (size_t k = 0; k < kSlots; k++) {
        Port& p = ports[k];
        const u8* slot = ctl.base + kCtlHdr + k * kSlotStride;
        auto word = [&](size_t o) {
          return *reinterpret_cast<const volatile u32*>(slot + o);
        };

        // Only the low half of the token is tested by the module's submit(),
        // and it is the last thing written, so an acquire fence here pairs with
        // the guest's release and guarantees the block is fully visible.
        if (word(kOffToken) == 0)
          continue;
        base::atomic_thread_fence(base::memory_order_acquire);

        const u32 bpf = word(kOffBpf), type = word(kOffType);
        const u32 rate = word(kOffRate), grain = word(kOffGrain);
        if (!PlausibleSlot(bpf, type, rate, grain)) {
          if (!p.bad_format) {
            p.bad_format = true;
            BASE_LOGI("audiod",
                      "slot {} declares bpf={} type={} rate={} grain={}; not "
                      "a shape this module can produce, leaving it alone",
                      k, bpf, type, rate, grain);
          }
          continue;
        }

        // The slot's recorded port index is also the event-flag bit. A title
        // waiting on SEVERAL bits with AND|CLEARPAT (SotC: 0x1000c0) releases
        // only when all are set, so granting the wrong bit deadlocks the whole
        // port set, not one port. Verify.
        if (word(kOffIndex) != k) {
          if (!p.bad_format) {
            p.bad_format = true;
            BASE_LOGI("audiod",
                      "slot {} declares port index {}; refusing to grant, the "
                      "event-flag bit would be wrong",
                      k, word(kOffIndex));
          }
          continue;
        }

        auto it = areas.find((int)k);
        const size_t need = (size_t)grain * bpf;
        if (it == areas.end() || !it->second.base || it->second.size < need) {
          // Same stakes as above: a pending block we never take is a grant we
          // never give, and under a multi-bit AND wait that hangs every port on
          // that thread. Say so once instead of skipping in silence.
          if (!p.warned_no_area) {
            p.warned_no_area = true;
            BASE_LOGI("audiod",
                      "port {} has a pending block but its sample region is {} "
                      "(need {} bytes); not granting",
                      k, it == areas.end() ? "not mapped" : "too small", need);
          }
          continue;  // sample region not mapped yet: come back next tick
        }

        // Pace to real time. The guest is parked on our event-flag grant, so
        // simply not granting yet IS the back-pressure hardware applies.
        if (p.next_due_us && now < p.next_due_us) {
          const u64 wait = p.next_due_us - now;
          if (wait < sleep_us)
            sleep_us = wait;
          continue;
        }

        // (Re)open the sink when a port first appears or changes shape.
        const u32 channels = bpf / (type == 0 ? 2u : 4u);
        if (!p.configured || p.bpf != bpf || p.type != type || p.rate != rate ||
            p.channels != channels) {
          if (p.sink >= 0) {
            if (auto f = ps4::GetAudioSink().close)
              f(p.sink);
            p.sink = -1;
          }
          p.bpf = bpf;
          p.type = type;
          p.rate = rate;
          p.grain = grain;
          p.channels = channels;
          p.configured = true;
          if (type == 0 || type == 1) {
            p.sink = ps4::GetAudioSink().open
                         ? ps4::GetAudioSink().open(rate, channels,
                                                    type == 1 ? 1 : 0)
                         : -1;
            BASE_LOGI("audiod",
                      "port {} -> sink {}: {}ch {} {}Hz grain={} (bpf={})", k,
                      p.sink, channels, type == 1 ? "f32" : "s16", rate, grain,
                      bpf);
          } else {
            // Sample type 2 ("32-bit") is real (the kernel PCM path supports
            // it) but has no host sink mapping; drain the port rather than
            // guess a conversion.
            BASE_LOGI("audiod",
                      "port {} sample type {} is unverified; draining without "
                      "playback",
                      k, type);
          }
        }
        p.grain = grain;

        block.assign(it->second.base, it->second.base + need);

        // Release the guest: zero the token, then grant the mix bit. Order
        // matters, since the module re-tests the token as soon as it wakes.
        *reinterpret_cast<volatile u64*>(const_cast<u8*>(slot) + kOffToken) = 0;
        base::atomic_thread_fence(base::memory_order_release);
        EvfSetByNameSubstr(kMixFlagStem, 1ull << k);

        const u64 period = (u64)grain * 1000000ull / rate;
        p.next_due_us = (p.next_due_us ? p.next_due_us : now) + period;
        if (p.next_due_us <
            now)  // fell behind (host stall): resync, don't burst
          p.next_due_us = now;

        p.blocks++;
        if (p.sink >= 0) {
          // Per-channel gains, applied as one uniform gain (the sink has no
          // per-channel control). Hardware's mixer is what applies these; the
          // guest memcpy's its buffer verbatim.
          const float* g = reinterpret_cast<const float*>(slot + kOffGain);
          float want = 0.f;
          for (u32 c = 0; c < channels && c < 8; c++)
            if (g[c] > want)
              want = g[c];
          if (!(want >= 0.f) || want > 1.f)  // NaN or out of range: leave as is
            want = p.gain;
          if (want < p.gain - 0.001f || want > p.gain + 0.001f) {
            p.gain = want;
            if (auto f = ps4::GetAudioSink().volume)
              f(p.sink, want);
          }
          if (auto f = ps4::GetAudioSink().output)
            f(p.sink, block.data(), grain);
        } else {
          p.dropped++;
        }

        if (TraceOn()) {
          const float pk = BlockPeak(block.data(), need, type);
          p.peak = pk;
          if (pk > p.peak_max)
            p.peak_max = pk;
        }
      }
    }

    if (TraceOn() && now - last_trace > 2000000) {
      last_trace = now;
      for (size_t k = 0; k < kSlots; k++) {
        const Port& p = ports[k];
        if (!p.blocks)
          continue;
        BASE_LOGI("audiod",
                  "port={} sink={} {}ch {} {}Hz grain={} blocks={} dropped={} "
                  "peak={:.3f} peakMax={:.3f}",
                  k, p.sink, p.channels, p.type == 1 ? "f32" : "s16", p.rate,
                  p.grain, (unsigned long long)p.blocks,
                  (unsigned long long)p.dropped, p.peak, p.peak_max);
      }
    }

    if (sleep_us < 250)
      sleep_us =
          250;  // never spin: worst case ~4k wakeups/s, all of them cheap
    base::SleepForMicroseconds(sleep_us);
  }
}

// "/shm_<pid>_<idx>_A" -> idx, or -1.
int ParsePortIndex(const base::String& n) {
  if (n.size() < 4 || n.compare(n.size() - 2, 2, "_A") != 0)
    return -1;
  const size_t e = n.size() - 2;  // the '_' of "_A"
  const size_t s = n.rfind('_', e - 1);
  if (s == base::String::npos || s + 1 >= e)
    return -1;
  int v = 0;
  for (size_t i = s + 1; i < e; i++) {
    if (n[i] < '0' || n[i] > '9')
      return -1;
    v = v * 10 + (n[i] - '0');
  }
  return v;
}

}  // namespace

void AudioDaemonForgetRange(const void* base, size_t size) {
  const auto first = reinterpret_cast<uintptr_t>(base);
  base::LockGuard<base::Mutex> lk(g_m);
  for (auto it = g_ps5_containers.begin(); it != g_ps5_containers.end();) {
    const auto other = reinterpret_cast<uintptr_t>(it->second.region.base);
    if (first < other + it->second.region.size && other < first + size)
      it = g_ps5_containers.erase(it);
    else
      ++it;
  }
}

void AudioDaemonNoticeShm(const char* name, u8* base, size_t size) {
  if (!name || !Enabled())
    return;
  const base::String n(name);
  constexpr base::StringRef kPs5Stem = "/SceAuOut2ContShm";
  const bool ps5 = n.starts_with(kPs5Stem);
  if (!ps5 && n.compare(0, 5, "/shm_") != 0)
    return;

  bool start = false;
  {
    base::LockGuard<base::Mutex> lk(g_m);
    if (ps5) {
      const size_t split = n.rfind("0x");
      if (split == base::String::npos || split <= kPs5Stem.size())
        return;
      u64 index;
      const char* end = nullptr;
      if (!base::ParseUnsigned(n.c_str() + split + 2, index, 16, &end) ||
          end != n.c_str() + n.size() || index >= 64)
        return;
      auto& c = g_ps5_containers[n];
      if (c.region.base != base)
        c = {};
      c.region = {base, size};
      c.index = index;
      c.flag = "SceAuOut2ContPushEvf" +
               n.substr(kPs5Stem.size(), split - kPs5Stem.size());
      start = base != nullptr;
    } else if (n.size() > 2 && n.compare(n.size() - 2, 2, "_C") == 0) {
      g_ctl = {base, size};
      start = base != nullptr;
    } else {
      const int idx = ParsePortIndex(n);
      if (idx < 0 || idx >= (int)kSlots)
        return;
      g_area[idx] = {base, size};
    }
  }

  if (!start)
    return;
  bool expected = false;
  if (!g_started.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("audiod",
            "'{}' is an LLE audio control block; starting the system "
            "audio daemon stand-in",
            name);
  if (kAudiomixAck)
    BASE_LOGI("audiod",
              "WARNING: DELTA_AUDIOMIX_ACK is set. That research aid fakes the "
              "mix-flag grant on a timer, which races the daemon's real grant "
              "and breaks pacing.");
  base::SpawnDetachedThread("audio_daemon", DaemonMain);
}

}  // namespace krnl

namespace krnl::ps4 {
namespace {
AudioSink g_sink;
}
void SetAudioSink(const AudioSink& sink) {
  g_sink = sink;
}
const AudioSink& GetAudioSink() {
  return g_sink;
}
}  // namespace krnl::ps4
