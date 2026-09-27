/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceVideoOut implementation. See lib_sce_video_out.h for why this
 * overrides the LLE module.
 */

#include "runtime/vprx/ps4/lib_sce_video_out/lib_sce_video_out.h"
#include "base/arch.h"
#include "guest_abi.h"

#include <cstdio>
#include <cstring>

#include "base/logging.h"

#include "base/atomic.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "gpu/ps4/cmd_processor.h"
#include "host/window.h"
#include "kern/lv2/sys_event.h"
#include "kern/lv2/sys_mem.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kVoFail, "DELTA_VO_FAIL", nullptr);
DELTA_OPTION(bool, kVoNostomp, "DELTA_VO_NOSTOMP", false);
}  // namespace

// PS5 present bridge: forwards the flip to the AGC command processor's
// vk::endFrame (gpu/ps5/cmd_processor.cpp).
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_agc_flip(u64 scanout_base);

using namespace kern;

namespace {

// SCE pixel formats we care about. A8R8G8B8 is BGRA in little-endian memory;
// A8B8G8R8 is RGBA. The high bit marks the family, bit 0x2200 the channel
// order.
constexpr u32 kFmtA8R8G8B8_SRGB = 0x80000000u;

// PS4 videoout kernel-event filter. Real EVFILT_DISPLAY is -13 (the vblank pump
// already uses it). Flip completions are a distinct source, so give them their
// own filter so the 60 Hz vblank pump never spuriously fires a flip knote.
constexpr i16 kFilterFlip = -10;
constexpr i16 kFilterVblank = -13;

// videoout event ids returned by sceVideoOutGetEventId.
constexpr int kEventFlip = 0;
constexpr int kEventVblank = 1;

// SceVideoOutResolutionStatus, 0x30 bytes.
struct ResolutionStatus {
  i32 width;
  i32 height;
  i32 pane_width;
  i32 pane_height;
  u64 refresh_rate;
  float screen_size_in_inch;
  u16 flags;
  u16 reserved0;
  u32 reserved1[3];
};

// SceVideoOutFlipStatus, 0x40 bytes.
struct FlipStatus {
  u64 count;
  u64 process_time;
  u64 tsc;
  i64 flip_arg;
  u64 submit_tsc;
  u64 reserved0;
  i32 gc_queue_num;
  i32 flip_pending_num;
  i32 current_buffer;
  u32 reserved1;
};

// SceVideoOutVblankStatus, 0x30 bytes.
struct VblankStatus {
  u64 count;
  u64 process_time;
  u64 tsc;
  u64 reserved[1];
  u8 flags;
  u8 pad[7];
};

// SceVideoOutBufferAttribute, 0x28 bytes.
struct BufferAttribute {
  u32 pixel_format;
  i32 tiling_mode;
  i32 aspect_ratio;
  u32 width;
  u32 height;
  u32 pitch_in_pixel;
  u32 option;
  u32 reserved0;
  u64 reserved1;
};

constexpr int kMaxBuffers = 16;
constexpr int kHandleBase = 1;

struct VideoPort {
  bool open = false;
  int flip_rate = 0;
  u32 width = 1920;
  u32 height = 1080;
  u32 pitch = 1920;  // in pixels
  u32 pixel_format = kFmtA8R8G8B8_SRGB;
  void* buffers[kMaxBuffers] = {};
  int buffer_count = 0;

  // flip bookkeeping (read back via sceVideoOutGetFlipStatus).
  base::Atomic<u64> flip_count{0};
  base::Atomic<u64> submit_count{0};
  i64 last_flip_arg = -1;
  int current_buffer = -1;
  // When the last flip was submitted and when it completed. Zero here is not
  // harmless: a title that decides a per-frame resource is retired by comparing
  // its own submit stamp against the flip's never sees one advance, so it
  // allocates a fresh one every frame instead of recycling.
  base::Atomic<u64> last_submit_tsc{0};
  base::Atomic<u64> last_flip_tsc{0};
  base::Atomic<u64> last_process_time{0};

  // equeue (by handle) a flip/vblank event was registered on, so SubmitFlip can
  // wake exactly that queue. Isaac uses one display port + one equeue.
  int flip_equeue = -1;
  void* flip_udata = nullptr;
  int vblank_equeue = -1;
  void* vblank_udata = nullptr;

  // Flip labels handed out via GetBufferLabelAddress MUST live in
  // GUEST-addressable memory: the title embeds the address in PM4 and the
  // command processor's label range check rightly refuses host .bss (SotC's
  // render fence never landed).
  u64* labels = nullptr;
};

// Monotonic nanoseconds, used for the flip/vblank timestamps the SCE structs
// carry (processTime is documented as microseconds, tsc as a raw counter).
static u64 NowNs() {
  return (u64)base::TickClock::NowNs();
}

u64* VideoLabels();  // fwd (needs g_mtx/g_port below)

base::Mutex g_mtx;
VideoPort g_port;  // single display port is enough for Isaac

// Guest-visible 16-slot label block, allocated on first use (either the pump
// or the title asking for the address can get here first).
u64* VideoLabels() {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!g_port.labels)
    g_port.labels =
        reinterpret_cast<u64*>(kern::AllocLowGuest(16 * sizeof(u64)));
  return g_port.labels;
}
base::Atomic<bool> g_gfx_up{false};

base::Atomic<int> g_gfx_state{0};  // 0=untried, 1=up, 2=failed

bool EnsureGfx(u32 w, u32 h) {
  int st = g_gfx_state.load();
  if (st == 1)
    return true;
  if (st == 2)
    return false;  // tried once and failed; don't spam retries every frame
  base::LockGuard<base::Mutex> lk(g_mtx);
  st = g_gfx_state.load();
  if (st != 0)
    return st == 1;
  if (!host::Init("prosperity", w, h)) {
    BASE_LOGI("videoout", "host::Init FAILED (no window this run)");
    g_gfx_state.store(2);
    return false;
  }
  g_gfx_state.store(1);
  g_gfx_up.store(true);
  BASE_LOGI("videoout", "gfx window up ({}x{})", w, h);
  return true;
}

Equeue* FindEqueue(int handle) {
  auto* p = Process::GetActive();
  if (!p)
    return nullptr;
  auto* obj = p->GetObjTable().Get(static_cast<u32>(handle));
  if (!obj || obj->type() != Object::Type::kEqueue)
    return nullptr;
  return static_cast<Equeue*>(obj);
}

// Present the most recently flipped scanout buffer to the window.
void PresentScanout() {
  void* fb;
  u32 w, h, pitch, fmt;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    int idx = g_port.current_buffer >= 0 ? g_port.current_buffer : 0;
    fb = (idx < kMaxBuffers) ? g_port.buffers[idx] : nullptr;
    w = g_port.width;
    h = g_port.height;
    pitch = g_port.pitch;
    fmt = g_port.pixel_format;
  }
  if (!fb || !EnsureGfx(w, h))
    return;
  auto pf =
      (fmt & 0x2200u) ? host::PixelFormat::kRgba8 : host::PixelFormat::kBgra8;
  host::Present(fb, w, h, pitch * 4, pf);
  host::PumpEvents();
}

base::Atomic<bool> g_flip_pump_started{false};

// The game flips through Gnm (a PM4 prepareFlip), then blocks in kevent on the
// equeue from sceVideoOutAddFlipEvent. No GPU yet, so synthesize completion: a
// ~60 Hz pump presenting the current scanout buffer and posting the flip event
// to registrants.
void StartFlipPump() {
  bool expected = false;
  if (!g_flip_pump_started.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("videoout", "flip pump started (60 Hz)");
  base::SpawnDetachedThread("libSceVideoOut", [] {
    for (;;) {
      base::SleepForMicroseconds(16667);
      // NB: do NOT present here. The window is driven solely by the GPU
      // renderer on the submit thread; the window has one swapchain/command
      // buffer and a present from this pump thread races it, intermittently
      // deadlocking Vulkan. This pump only synthesizes flip completion (labels
      // + events); the scanout buffer is never CPU-written.
      u64 c = g_port.flip_count.fetch_add(1) + 1;
      // Mark flip completion in the buffer labels (Gnm's prepareFlip writes
      // label = GetBufferLabelAddress + index*8; the game busy-polls to recycle
      // buffers). With no GPU we write a monotonic flip count satisfying ">=
      // submitted id". DELTA_VO_NOSTOMP: leave labels to the title's own GPU
      // fence writes (the command processor lands them in this guest-visible
      // block); the stomp overwrites the EXACT flip-arg a real Gnm flip
      // protocol may compare against.
      if (!kVoNostomp)
        if (u64* lb = VideoLabels())
          for (int i = 0; i < 16; i++)
            lb[i] = c;
      // post the flip-complete event to whichever equeue holds a flip knote.
      TriggerAllEqueues(kEventFlip, kFilterFlip, static_cast<i64>(c));
    }
  });
}

}  // namespace

// DIAGNOSTIC: inject a failure return into a named HLE setup function to find
// which real-videoout return value makes Isaac skip its command-buffer
// creation. DELTA_VO_FAIL=open|regbuf|fliprate|addflip[:<hex retval>] (default
// retval -1).
static int FailInject(const char* name) {
  const char* f = kVoFail;
  if (!f || std::strncmp(f, name, std::strlen(name)) != 0)
    return 0;  // 0 = don't inject
  const char* c = std::strchr(f, ':');
  return c ? (int)std::strtol(c + 1, nullptr, 0) : -1;
}

extern "C" {

int PS4ABI sceVideoOutOpen(int user_id,
                           int bus_type,
                           int index,
                           const void* param) {
  BASE_LOGI("videoout", "open user={} bus={} idx={}", user_id, bus_type, index);
  if (int r = FailInject("open")) {
    BASE_LOGI("vofail", "open -> {}", r);
    return r;
  }
  base::LockGuard<base::Mutex> lk(g_mtx);
  g_port.open = true;
  // bring the window up early so the user sees something while the game inits.
  // (do it outside the lock-sensitive window path on first flip if init is
  // heavy)
  return kHandleBase;
}

int PS4ABI sceVideoOutClose(int handle) {
  BASE_LOGI("videoout", "close h={}", handle);
  base::LockGuard<base::Mutex> lk(g_mtx);
  g_port.open = false;
  return 0;
}

int PS4ABI sceVideoOutGetResolutionStatus(int handle, void* status) {
  if (!status)
    return -1;
  auto* s = static_cast<ResolutionStatus*>(status);
  std::memset(s, 0, sizeof(*s));
  s->width = static_cast<i32>(g_port.width);
  s->height = static_cast<i32>(g_port.height);
  s->pane_width = s->width;
  s->pane_height = s->height;
  s->refresh_rate = 1;  // SCE_VIDEO_OUT_REFRESH_RATE_59_94HZ
  s->screen_size_in_inch = 50.0f;
  s->flags = 0;
  return 0;
}

int PS4ABI sceVideoOutSetBufferAttribute(void* attribute,
                                         u32 pixel_format,
                                         u32 tiling_mode,
                                         u32 aspect_ratio,
                                         u32 width,
                                         u32 height,
                                         u32 pitch_in_pixel) {
  if (!attribute)
    return -1;
  auto* a = static_cast<BufferAttribute*>(attribute);
  std::memset(a, 0, sizeof(*a));
  a->pixel_format = pixel_format;
  a->tiling_mode = static_cast<i32>(tiling_mode);
  a->aspect_ratio = static_cast<i32>(aspect_ratio);
  a->width = width;
  a->height = height;
  a->pitch_in_pixel = pitch_in_pixel;
  BASE_LOGI("videoout", "setBufferAttribute fmt={:#x} tiling={} {}x{} pitch={}",
            pixel_format, tiling_mode, width, height, pitch_in_pixel);
  return 0;
}

int PS4ABI sceVideoOutRegisterBuffers(int handle,
                                      int start_index,
                                      void* const* addresses,
                                      int buffer_num,
                                      const void* attribute) {
  if (int r = FailInject("regbuf")) {
    BASE_LOGI("vofail", "regbuf -> {}", r);
    return r;
  }
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (attribute) {
    auto* a = static_cast<const BufferAttribute*>(attribute);
    g_port.width = a->width ? a->width : g_port.width;
    g_port.height = a->height ? a->height : g_port.height;
    g_port.pitch = a->pitch_in_pixel ? a->pitch_in_pixel : g_port.width;
    g_port.pixel_format = a->pixel_format;
  }
  int n = 0;
  for (int i = 0; i < buffer_num && (start_index + i) < kMaxBuffers; i++) {
    g_port.buffers[start_index + i] = addresses ? addresses[i] : nullptr;
    n++;
  }
  g_port.buffer_count = start_index + n;
  BASE_LOGI("videoout",
            "registerBuffers start={} num={} -> {}x{} pitch={} fmt={:#x} "
            "(buf0={:p})",
            start_index, buffer_num, g_port.width, g_port.height, g_port.pitch,
            g_port.pixel_format, addresses ? addresses[0] : nullptr);
  return 0;
}

int PS4ABI sceVideoOutUnregisterBuffers(int handle, int attribute_index) {
  BASE_LOGI("videoout", "unregisterBuffers idx={}", attribute_index);
  return 0;
}

int PS4ABI sceVideoOutSetFlipRate(int handle, int rate) {
  BASE_LOGI("videoout", "setFlipRate {}", rate);
  if (int r = FailInject("fliprate")) {
    BASE_LOGI("vofail", "fliprate -> {}", r);
    return r;
  }
  g_port.flip_rate = rate;
  return 0;
}

int PS4ABI sceVideoOutAddFlipEvent(int eq_handle, int handle, void* udata) {
  BASE_LOGI("videoout", "addFlipEvent eq={} h={} udata={:p}", eq_handle, handle,
            udata);
  if (int r = FailInject("addflip")) {
    BASE_LOGI("vofail", "addflip -> {}", r);
    return r;
  }
  auto* eq = FindEqueue(eq_handle);
  if (!eq)
    return -1;
  g_port.flip_equeue = eq_handle;
  g_port.flip_udata = udata;
  eq->AddEvent(static_cast<u64>(kEventFlip), kFilterFlip, udata);
  StartFlipPump();
  return 0;
}

int PS4ABI sceVideoOutDeleteFlipEvent(int eq_handle, int handle) {
  BASE_LOGI("videoout", "deleteFlipEvent eq={} h={}", eq_handle, handle);
  auto* eq = FindEqueue(eq_handle);
  if (eq)
    eq->RemoveEvent(static_cast<u64>(kEventFlip), kFilterFlip);
  g_port.flip_equeue = -1;
  return 0;
}

int PS4ABI sceVideoOutAddVblankEvent(int eq_handle, int handle, void* udata) {
  BASE_LOGI("videoout", "addVblankEvent eq={} h={} udata={:p}", eq_handle,
            handle, udata);
  auto* eq = FindEqueue(eq_handle);
  if (!eq)
    return -1;
  g_port.vblank_equeue = eq_handle;
  g_port.vblank_udata = udata;
  // vblank rides the existing 60 Hz EVFILT_DISPLAY pump (ident wildcard).
  eq->AddEvent(static_cast<u64>(kEventVblank), kFilterVblank, udata);
  return 0;
}

int PS4ABI sceVideoOutGetEventCount(const void* event) {
  return 1;
}

int PS4ABI sceVideoOutGetEventId(const void* event) {
  if (!event)
    return kEventFlip;
  // event is a SceKernelEvent (kevent_t). distinguish by filter.
  auto* ev = static_cast<const kevent_t*>(event);
  if (ev->filter == kFilterVblank)
    return kEventVblank;
  return kEventFlip;
}

int PS4ABI sceVideoOutGetEventData(const void* event, i64* data) {
  if (!event || !data)
    return -1;
  auto* ev = static_cast<const kevent_t*>(event);
  *data = ev->data;
  return 0;
}

int PS4ABI sceVideoOutSubmitFlip(int handle,
                                 int buffer_index,
                                 int flip_mode,
                                 i64 flip_arg) {
  void* fb = nullptr;
  u32 w, h, pitch, fmt;
  int eq_handle;
  void* udata;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    if (buffer_index >= 0 && buffer_index < kMaxBuffers)
      fb = g_port.buffers[buffer_index];
    w = g_port.width;
    h = g_port.height;
    pitch = g_port.pitch;
    fmt = g_port.pixel_format;
    g_port.current_buffer = buffer_index;
    g_port.last_flip_arg = flip_arg;
    const u64 t = NowNs();
    g_port.last_submit_tsc.store(t);
    g_port.last_flip_tsc.store(t);
    g_port.last_process_time.store(t / 1000);
    g_port.submit_count.fetch_add(1);
  }

  // present the scanout buffer (guest pointers are identity-mapped, so the
  // guest address is directly readable on the host). Until the Gnm->Vulkan
  // path detiles real GPU output this is the linear scanout contents.
  if (fb && EnsureGfx(w, h)) {
    auto pf =
        (fmt & 0x2200u) ? host::PixelFormat::kRgba8 : host::PixelFormat::kBgra8;
    host::Present(fb, w, h, pitch * 4, pf);
    host::PumpEvents();
  }

  // flip "completes" immediately: bump the count and wake the flip equeue.
  g_port.flip_count.fetch_add(1);
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    eq_handle = g_port.flip_equeue;
    udata = g_port.flip_udata;
  }
  if (eq_handle >= 0) {
    if (auto* eq = FindEqueue(eq_handle))
      eq->Trigger(kEventFlip, kFilterFlip,
                  static_cast<i64>(g_port.flip_count.load()));
  }
  return 0;
}

int PS4ABI sceVideoOutSubmitFlipEop(int handle,
                                    int buffer_index,
                                    int flip_mode,
                                    i64 flip_arg,
                                    void* eop_label) {
  // The real libSceGnmDriver (LLE) flips through this internal videoout entry,
  // the EOP-label variant (eopLabel = the GPU completion label we don't need;
  // our submit is synchronous). Present the GPU's render target here
  // (endFrame), not the raw guest scanout buffer the title never CPU-writes;
  // then complete like SubmitFlip.
  u64 scanout = 0;
  int eq_handle;
  void* udata;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    if (buffer_index >= 0 && buffer_index < kMaxBuffers) {
      scanout = reinterpret_cast<u64>(g_port.buffers[buffer_index]);
      g_port.current_buffer = buffer_index;
      const u64 t = NowNs();
      g_port.last_submit_tsc.store(t);
      g_port.last_flip_tsc.store(t);
      g_port.last_process_time.store(t / 1000);
    }
    g_port.last_flip_arg = flip_arg;
    g_port.submit_count.fetch_add(1);
    eq_handle = g_port.flip_equeue;
    udata = g_port.flip_udata;
  }
  // endFrame takes the GPU's own lock; call it outside g_mtx. It presents the
  // RT matching `scanout`, falling back to the last RT rendered when it isn't
  // one. On PS5 the frame was rendered by the AGC command processor (gpu::ps5),
  // so route the present there; the PS4 Gnm path uses gpu::ps4::EndFrame.
  auto* active = Process::GetActive();
  if (active && active->GetPlatform() == Process::Platform::kPs5)
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
    prosperity_agc_flip(scanout);
  else
    gpu::ps4::EndFrame(scanout);
  (void)udata;

  g_port.flip_count.fetch_add(1);
  if (eq_handle >= 0) {
    if (auto* eq = FindEqueue(eq_handle))
      eq->Trigger(kEventFlip, kFilterFlip,
                  static_cast<i64>(g_port.flip_count.load()));
  }
  return 0;
}

int PS4ABI sceVideoOutGetFlipStatus(int handle, void* status) {
  if (!status)
    return -1;
  auto* s = static_cast<FlipStatus*>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flip_count.load();
  s->flip_arg = g_port.last_flip_arg;
  s->current_buffer = g_port.current_buffer;
  s->gc_queue_num = 0;
  // pending = submitted but not yet "completed"; we complete synchronously.
  s->flip_pending_num = 0;
  s->tsc = g_port.last_flip_tsc.load();
  s->submit_tsc = g_port.last_submit_tsc.load();
  s->process_time = g_port.last_process_time.load();
  return 0;
}

int PS4ABI sceVideoOutIsFlipPending(int handle) {
  return 0;  // never pending: flips complete synchronously
}

int PS4ABI sceVideoOutGetVblankStatus(int handle, void* status) {
  if (!status)
    return -1;
  auto* s = static_cast<VblankStatus*>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flip_count.load();
  s->flags = 0;
  s->tsc = g_port.last_flip_tsc.load();
  s->process_time = g_port.last_process_time.load();
  return 0;
}

int PS4ABI sceVideoOutWaitVblank(int handle) {
  return 0;
}

int PS4ABI sceVideoOutGetBufferLabelAddress(int handle, uintptr_t* label) {
  // Gnm's flip path checks `eax == 0` for success (then reads *label to build a
  // GPU completion-label write). Returning the slot count here makes Gnm treat
  // the flip request as failed ("flip request failed"). Return 0 = success.
  if (label)
    *label = reinterpret_cast<uintptr_t>(VideoLabels());
  return 0;
}

int PS4ABI sceVideoOutSetWindowModeMargins(int handle, int top, int bottom) {
  return 0;
}

int PS4ABI sceVideoOutColorSettingsSetGamma_(void* settings, float gamma) {
  return 0;
}

int PS4ABI sceVideoOutModeSetAny_(int handle, void* arg) {
  return 0;
}

// Bridge for the HLE Gnm driver: the game flips via sceGnmSubmitAndFlip-
// CommandBuffers (a GPU prepareFlip packet), so record the target scanout
// buffer here; the flip pump then presents it and posts the flip-complete
// event.
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
void prosperity_videoout_set_flip(int buffer_index, i64 flip_arg) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (buffer_index >= 0 && buffer_index < kMaxBuffers)
    g_port.current_buffer = buffer_index;
  g_port.last_flip_arg = flip_arg;
}
// NOLINTEND(readability-identifier-naming)

// The guest scanout buffer address for a registered buffer index (the GPU
// render target the flip displays). Used by the GPU renderer to present the
// right render target.
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
u64 prosperity_videoout_buffer(int buffer_index) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (buffer_index >= 0 && buffer_index < kMaxBuffers)
    return reinterpret_cast<u64>(g_port.buffers[buffer_index]);
  return 0;
}
// NOLINTEND(readability-identifier-naming)

}  // extern "C"
