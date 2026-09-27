/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 (Prospero) copy of the HLE libSceVideoOut. Prospero exports some
 * functions under different NIDs than PS4 (sceVideoOutSetBufferAttribute =
 * PjS5uASwcV8, sceVideoOutRegisterBuffers = rKBUtgRrtbk, whose ABI also gains
 * an extra `option` arg) and its LLE .sprx never registers its display port in
 * our env. This is a dedicated PS5 copy with its own port state and functions,
 * so its behaviour can diverge from the PS4 HLE without touching PS4 titles.
 * Registered in the PS5-only registry (MODULE_INIT_PS5); the ps5Layout import
 * resolver force-routes libSceVideoOut here. NIDs decoded from the PPSA03311
 * (Isaac) eboot import table.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"  // PS4ABI (via <guest_abi.h>), MODULE_INIT_PS5

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/logging.h"

#include "host/window.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/sys_event.h"
#include "kern/process.h"

#include "base/atomic.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "kern/lv2/sys_mem.h"  // AllocLowGuest
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kVoNostomp, "DELTA_VO_NOSTOMP", false);
}  // namespace

// PS5 present bridge: forwards the flip to the AGC command processor's
// vk::endFrame (gpu/ps5/cmd_processor.cpp).
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_agc_flip(u64 scanout_base);

// The guest address of the display buffer the game most recently flipped
// (sceVideoOutSubmitFlip*'s bufferIndex resolved through the registered-buffer
// table). The PS5 /dev/gc AGC flip ioctls carry no buffer field, so they read
// the scanout target here instead of presenting whichever RT was drawn last.
static base::Atomic<u64> g_current_scanout{0};
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" u64 prosperity_ps5_scanout_base() {
  return g_current_scanout.load(base::memory_order_relaxed);
}
// NOLINTEND(readability-identifier-naming)

using namespace kern;

namespace {

constexpr u32 kFmtA8R8G8B8_SRGB = 0x80000000u;
constexpr i16 kFilterFlip = -10;
constexpr i16 kFilterVblank = -13;
constexpr int kEventFlip = 0;
constexpr int kEventVblank = 1;

struct ResolutionStatus {
  i32 width, height, pane_width, pane_height;
  u64 refresh_rate;
  float screen_size_in_inch;
  u16 flags, reserved0;
  u32 reserved1[3];
};

struct FlipStatus {
  u64 count, process_time, tsc;
  i64 flip_arg;
  u64 submit_tsc, reserved0;
  i32 gc_queue_num, flip_pending_num, current_buffer;
  u32 reserved1;
};

struct VblankStatus {
  u64 count, process_time, tsc, reserved[1];
  u8 flags, pad[7];
};

// SceVideoOutBufferAttribute, 0x28 bytes (shared layout with PS4).
struct BufferAttribute {
  u32 pixel_format;
  i32 tiling_mode, aspect_ratio;
  u32 width, height, pitch_in_pixel, option, reserved0;
  u64 reserved1;
};

constexpr int kMaxBuffers = 16;
constexpr int kHandleBase = 1;

struct VideoPort {
  bool open = false;
  int flip_rate = 0;
  u32 width = 1920, height = 1080, pitch = 1920;
  u32 pixel_format = kFmtA8R8G8B8_SRGB;
  void* buffers[kMaxBuffers] = {};
  int buffer_count = 0;
  base::Atomic<u64> flip_count{0};
  base::Atomic<u64> submit_count{0};
  i64 last_flip_arg = -1;
  int current_buffer = -1;
  int flip_equeue = -1;
  void* flip_udata = nullptr;
  int vblank_equeue = -1;
  void* vblank_udata = nullptr;
};

base::Mutex g_mtx;
VideoPort g_port;  // dedicated PS5 port state

// The 16 flip labels sceVideoOutGetBufferLabelAddress hands to the title. They
// must live in GUEST-addressable memory: the title embeds the address in the
// PM4 it builds, and the command processor's label range check rightly refuses
// to write host .bss.
u64* VideoLabels() {
  static u64* labels =
      reinterpret_cast<u64*>(kern::AllocLowGuest(16 * sizeof(u64)));
  return labels;
}

base::Atomic<int> g_gfx_state{0};  // 0=untried, 1=up, 2=failed

bool EnsureGfx(u32 w, u32 h) {
  int st = g_gfx_state.load();
  if (st == 1)
    return true;
  if (st == 2)
    return false;
  base::LockGuard<base::Mutex> lk(g_mtx);
  st = g_gfx_state.load();
  if (st != 0)
    return st == 1;
  if (!host::Init("prosperity", w, h)) {
    BASE_LOGI("videoout/ps5", "host::Init FAILED (no window this run)");
    g_gfx_state.store(2);
    return false;
  }
  g_gfx_state.store(1);
  BASE_LOGI("videoout/ps5", "gfx window up ({}x{})", w, h);
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

base::Atomic<bool> g_flip_pump_started{false};

// Synthesize flip completion (labels + events) so a title that flips via
// Gnm/AGC and blocks on the flip equeue keeps advancing. Does NOT present (the
// GPU renderer owns the swapchain; presenting here would race it).
void StartFlipPump() {
  bool expected = false;
  if (!g_flip_pump_started.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("videoout/ps5", "flip pump started (60 Hz)");
  base::SpawnDetachedThread("libSceVideoOut_", [] {
    for (;;) {
      base::SleepForMicroseconds(16667);
      u64 c = g_port.flip_count.fetch_add(1) + 1;
      // The label is a flip-completion flag, not a counter: the title leaves it
      // at 0 when it queues a flip and waits for the display controller to
      // write 1, then clears it again. bgfx's AGC backend spins on `*label ==
      // 1` exactly, so an incrementing value satisfies it once and never again.
      // DELTA_VO_NOSTOMP leaves the labels to the title's own GPU fence writes.
      if (!kVoNostomp) {
        u64* labels = VideoLabels();
        for (int i = 0; i < 16; i++)
          labels[i] = 1;
      }
      TriggerAllEqueues(kEventFlip, kFilterFlip, static_cast<i64>(c));
    }
  });
}

int PS4ABI VideoOutOpen(int user_id, int bus_type, int index, const void*) {
  BASE_LOGI("videoout/ps5", "open user={} bus={} idx={}", user_id, bus_type,
            index);
  base::LockGuard<base::Mutex> lk(g_mtx);
  g_port.open = true;
  return kHandleBase;
}

int PS4ABI VideoOutClose(int handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  g_port.open = false;
  return 0;
}

int PS4ABI VideoOutGetResolutionStatus(int, void* status) {
  if (!status)
    return -1;
  auto* s = static_cast<ResolutionStatus*>(status);
  std::memset(s, 0, sizeof(*s));
  s->width = static_cast<i32>(g_port.width);
  s->height = static_cast<i32>(g_port.height);
  s->pane_width = s->width;
  s->pane_height = s->height;
  s->refresh_rate = 1;
  s->screen_size_in_inch = 50.0f;
  return 0;
}

// Prospero drops PS4's aspectRatio argument, so width/height/pitch each sit one
// register earlier. Read as the PS4 shape, Skyrim's 3840x2160 attribute decoded
// as "2160x0 pitch 0" and every buffer registered at the wrong size.
int PS4ABI VideoOutSetBufferAttribute(void* attribute,
                                      u32 pixel_format,
                                      u32 tiling_mode,
                                      u32 width,
                                      u32 height,
                                      u32 pitch_in_pixel) {
  if (!attribute)
    return -1;
  auto* a = static_cast<BufferAttribute*>(attribute);
  std::memset(a, 0, sizeof(*a));
  a->pixel_format = pixel_format;
  a->tiling_mode = static_cast<i32>(tiling_mode);
  a->width = width;
  a->height = height;
  a->pitch_in_pixel = pitch_in_pixel;
  BASE_LOGI("videoout/ps5",
            "setBufferAttribute fmt={:#x} tiling={} {}x{} pitch={}",
            pixel_format, tiling_mode, width, height, pitch_in_pixel);
  return 0;
}

// The PS5 sceVideoOutRegisterBuffers2 `buffers` arg is an array of 32-byte
// SceVideoOutBuffers descriptors (base VA at offset 0), not raw void* pointers
// as on PS4. Stride over the descriptors; the base is the display buffer
// address.
constexpr int kBufDescStride = 4;  // u64s per descriptor (0x20 bytes)

// PS5 ABI: extra `option` arg before the descriptor array vs PS4.
int PS4ABI VideoOutRegisterBuffers(int,
                                   int start_index,
                                   int option,
                                   void* const* buffers,
                                   int buffer_num,
                                   const void* attribute) {
  (void)option;
  u32 w, h;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    if (attribute) {
      auto* a = static_cast<const BufferAttribute*>(attribute);
      g_port.width = a->width ? a->width : g_port.width;
      g_port.height = a->height ? a->height : g_port.height;
      g_port.pitch = a->pitch_in_pixel ? a->pitch_in_pixel : g_port.width;
      g_port.pixel_format = a->pixel_format;
    }
    const u64* desc = reinterpret_cast<const u64*>(buffers);
    int n = 0;
    for (int i = 0; i < buffer_num && (start_index + i) < kMaxBuffers; i++) {
      g_port.buffers[start_index + i] =
          desc ? reinterpret_cast<void*>(desc[i * kBufDescStride]) : nullptr;
      n++;
    }
    g_port.buffer_count = start_index + n;
    w = g_port.width;
    h = g_port.height;
    BASE_LOGI("videoout/ps5",
              "registerBuffers start={} num={} -> {}x{} pitch={} fmt={:#x} "
              "(buf0={:p} buf1={:p})",
              start_index, buffer_num, g_port.width, g_port.height,
              g_port.pitch, g_port.pixel_format, g_port.buffers[start_index],
              buffer_num > 1 ? g_port.buffers[start_index + 1] : nullptr);
    // A title that flips through AGC never calls sceVideoOutSubmitFlip, so it
    // never names a scanout buffer either. Default to the first one it just
    // registered, or the AGC flip ioctls present a null address.
    if (!g_current_scanout.load(base::memory_order_relaxed))
      g_current_scanout.store(
          reinterpret_cast<u64>(g_port.buffers[start_index]),
          base::memory_order_relaxed);
  }
  // Registering display buffers is the title committing to present, whichever
  // flip path it uses. Bring the window up here rather than in submitFlip,
  // which the AGC titles never reach: without it there is no swapchain to
  // present to.
  EnsureGfx(w, h);
  return 0;
}

int PS4ABI VideoOutUnregisterBuffers(int, int) {
  return 0;
}

int PS4ABI VideoOutSetFlipRate(int, int rate) {
  g_port.flip_rate = rate;
  return 0;
}

int PS4ABI VideoOutAddFlipEvent(int eq_handle, int, void* udata) {
  auto* eq = FindEqueue(eq_handle);
  if (!eq)
    return -1;
  g_port.flip_equeue = eq_handle;
  g_port.flip_udata = udata;
  eq->AddEvent(static_cast<u64>(kEventFlip), kFilterFlip, udata);
  StartFlipPump();
  return 0;
}

int PS4ABI VideoOutDeleteFlipEvent(int eq_handle, int) {
  auto* eq = FindEqueue(eq_handle);
  if (eq)
    eq->RemoveEvent(static_cast<u64>(kEventFlip), kFilterFlip);
  g_port.flip_equeue = -1;
  return 0;
}

int PS4ABI VideoOutAddVblankEvent(int eq_handle, int, void* udata) {
  auto* eq = FindEqueue(eq_handle);
  if (!eq)
    return -1;
  g_port.vblank_equeue = eq_handle;
  g_port.vblank_udata = udata;
  eq->AddEvent(static_cast<u64>(kEventVblank), kFilterVblank, udata);
  return 0;
}

int PS4ABI VideoOutGetEventCount(const void*) {
  return 1;
}

int PS4ABI VideoOutGetEventId(const void* event) {
  if (!event)
    return kEventFlip;
  auto* ev = static_cast<const kevent_t*>(event);
  return ev->filter == kFilterVblank ? kEventVblank : kEventFlip;
}

int PS4ABI VideoOutGetEventData(const void* event, i64* data) {
  if (!event || !data)
    return -1;
  *data = static_cast<const kevent_t*>(event)->data;
  return 0;
}

// Whether a title flips through VideoOut at all, and how often, is the
// first thing to know when nothing reaches the screen; the AGC path flips
// somewhere else entirely.
static void TraceSubmit(const char* what, int buffer_index, i64 flip_arg) {
  static base::Atomic<u64> n{0};
  const u64 i = n.fetch_add(1);
  if (i < 3 || (i % 600) == 0)
    BASE_LOGI("videoout/ps5", "{} #{} buffer={} arg={}", what,
              (unsigned long long)i, buffer_index, (long long)flip_arg);
}

int PS4ABI VideoOutSubmitFlip(int, int buffer_index, int, i64 flip_arg) {
  TraceSubmit("submitFlip", buffer_index, flip_arg);
  void* fb = nullptr;
  u32 w, h, pitch, fmt;
  int eq_handle;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    if (buffer_index >= 0 && buffer_index < kMaxBuffers)
      fb = g_port.buffers[buffer_index];
    w = g_port.width;
    h = g_port.height;
    pitch = g_port.pitch;
    fmt = g_port.pixel_format;
    g_port.current_buffer = buffer_index;
    g_current_scanout.store(reinterpret_cast<u64>(fb),
                            base::memory_order_relaxed);
    g_port.last_flip_arg = flip_arg;
    g_port.submit_count.fetch_add(1);
  }
  if (fb && EnsureGfx(w, h)) {
    auto pf =
        (fmt & 0x2200u) ? host::PixelFormat::kRgba8 : host::PixelFormat::kBgra8;
    host::Present(fb, w, h, pitch * 4, pf);
    host::PumpEvents();
  }
  g_port.flip_count.fetch_add(1);
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    eq_handle = g_port.flip_equeue;
  }
  if (eq_handle >= 0)
    if (auto* eq = FindEqueue(eq_handle))
      eq->Trigger(kEventFlip, kFilterFlip,
                  static_cast<i64>(g_port.flip_count.load()));
  return 0;
}

// The EOP variant the AGC path flips through. eopLabel is the GPU completion
// label: the title queues the flip, then spins until the display controller
// writes 1 there. Our present is synchronous, so the flip is already done by
// the time we return. Write the label or the title waits on it forever (bgfx's
// RendererContextAGC parks on `*label == 1` and never submits another frame).
int PS4ABI VideoOutSubmitFlipEop(int,
                                 int buffer_index,
                                 int,
                                 i64 flip_arg,
                                 void* eop_label) {
  TraceSubmit("submitFlipEop", buffer_index, flip_arg);
  u64 scanout = 0;
  int eq_handle;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    if (buffer_index >= 0 && buffer_index < kMaxBuffers) {
      scanout = reinterpret_cast<u64>(g_port.buffers[buffer_index]);
      g_port.current_buffer = buffer_index;
    }
    g_current_scanout.store(scanout, base::memory_order_relaxed);
    g_port.last_flip_arg = flip_arg;
    g_port.submit_count.fetch_add(1);
    eq_handle = g_port.flip_equeue;
  }
  // PS5 always presents through the AGC command processor's render target.
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
  prosperity_agc_flip(scanout);
  if (host_memory::IsMemoryRangeMapped(eop_label, sizeof(u64)))
    *static_cast<volatile u64*>(eop_label) = 1;
  g_port.flip_count.fetch_add(1);
  if (eq_handle >= 0)
    if (auto* eq = FindEqueue(eq_handle))
      eq->Trigger(kEventFlip, kFilterFlip,
                  static_cast<i64>(g_port.flip_count.load()));
  return 0;
}

int PS4ABI VideoOutGetFlipStatus(int, void* status) {
  if (!status)
    return -1;
  auto* s = static_cast<FlipStatus*>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flip_count.load();
  s->flip_arg = g_port.last_flip_arg;
  s->current_buffer = g_port.current_buffer;
  return 0;
}

int PS4ABI VideoOutIsFlipPending(int) {
  return 0;
}

int PS4ABI VideoOutGetVblankStatus(int, void* status) {
  if (!status)
    return -1;
  auto* s = static_cast<VblankStatus*>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flip_count.load();
  return 0;
}

int PS4ABI VideoOutWaitVblank(int) {
  return 0;
}

int PS4ABI VideoOutGetBufferLabelAddress(int, uintptr_t* label) {
  // Taking the label address means the title drives flips through AGC and waits
  // on these labels rather than on the flip equeue, so it never calls
  // sceVideoOutAddFlipEvent. Start the pump here too, or nothing advances the
  // labels and the first frame waits forever (Minecraft, PPSA17221).
  if (label)
    *label = reinterpret_cast<uintptr_t>(VideoLabels());
  StartFlipPump();
  return 0;
}

int PS4ABI VideoOutSetWindowModeMargins(int, int, int) {
  return 0;
}
int PS4ABI VideoOutColorSettingsSetGamma(void*, float) {
  return 0;
}
int PS4ABI VideoOutModeSetAny(int, void*) {
  return 0;
}

}  // namespace

// Is this address one of the display buffers the title registered? The AGC
// frame end uses it to tell "this frame rendered straight into a display
// buffer" (present that one) from "it rendered an offscreen pass" (present
// whatever the last flip named).
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" bool prosperity_ps5_is_display_buffer(u64 addr) {
  if (!addr)
    return false;
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (int i = 0; i < g_port.buffer_count && i < kMaxBuffers; i++)
    if (reinterpret_cast<u64>(g_port.buffers[i]) == addr)
      return true;
  return false;
}
// NOLINTEND(readability-identifier-naming)

static const runtime::FuncInfo functions[] = {
    {0x529DFA3D393AF3B1, (void*)&VideoOutOpen},                 // Up36PTk687E
    {0xBAAB951F8FC3BBBF, (void*)&VideoOutClose},                // uquVH4-Du78
    {0xEA43E78F9D53EB66, (void*)&VideoOutGetResolutionStatus},  // 6kPnj51T62Y
    {0x8BAFEC47DD56B7FE,
     (void*)&VideoOutSetBufferAttribute},  // i6-sR91Wt-4 (PS4 NID)
    {0x3E34B9B804B0715F,
     (void*)&VideoOutSetBufferAttribute},  // PjS5uASwcV8 (PS5 NID)
    {0xACA054B6046BB5B9,
     (void*)&VideoOutRegisterBuffers},  // rKBUtgRrtbk (PS5 NID+ABI)
    {0x379283B642238C9E, (void*)&VideoOutUnregisterBuffers},      // N5KDtkIjjJ4
    {0x0818AEE26084D430, (void*)&VideoOutSetFlipRate},            // CBiu4mCE1DA
    {0x1D7CE32BDC88DF49, (void*)&VideoOutAddFlipEvent},           // HXzjK9yI30k
    {0xFCECE7D05D401518, (void*)&VideoOutDeleteFlipEvent},        // -Ozn0F1AFRg
    {0x5EBBBDDB01C94668, (void*)&VideoOutAddVblankEvent},         // Xru92wHJRmg
    {0x32DE101C793190E7, (void*)&VideoOutGetEventCount},          // Mt4QHHkxkOc
    {0x536249B52A8D2992, (void*)&VideoOutGetEventId},             // U2JJtSqNKZI
    {0xAD651370A7645334, (void*)&VideoOutGetEventData},           // rWUTcKdkUzQ
    {0x538E8DC0E889A72B, (void*)&VideoOutSubmitFlip},             // U46NwOiJpys
    {0x8FCC65FBDD80D2AE, (void*)&VideoOutSubmitFlipEop},          // j8xl+92A0q4
    {0x49B537770A7CD254, (void*)&VideoOutGetFlipStatus},          // SbU3dwp80lQ
    {0xCE05E27C74FD12B6, (void*)&VideoOutIsFlipPending},          // zgXifHT9ErY
    {0xD456412B2F0778D5, (void*)&VideoOutGetVblankStatus},        // 1FZBKy8HeNU
    {0x8FA45A01495A2EFD, (void*)&VideoOutWaitVblank},             // j6RaAUlaLv0
    {0x39C4326D07A31C46, (void*)&VideoOutGetBufferLabelAddress},  // OcQybQejHEY
    {0x313C71ACE09E4A28, (void*)&VideoOutSetWindowModeMargins},   // MTxxrOCeSig
    {0x0D886159B2527918, (void*)&VideoOutColorSettingsSetGamma},  // DYhhWbJSeRg
    {0xA63903B20C658BA7, (void*)&VideoOutModeSetAny},             // pjkDsgxli6c
};

MODULE_INIT_PS5(libSceVideoOut);

extern "C" int g_vprx_anchor_ps5_lib_sce_video_out = 1;
