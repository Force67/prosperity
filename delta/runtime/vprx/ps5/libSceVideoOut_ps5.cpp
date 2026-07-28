/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 (Prospero) copy of the HLE libSceVideoOut. Prospero exports some functions
 * under different NIDs than PS4 (sceVideoOutSetBufferAttribute = PjS5uASwcV8,
 * sceVideoOutRegisterBuffers = rKBUtgRrtbk, whose ABI also gains an extra `option`
 * arg) and its LLE .sprx never registers its display port in our env. This is a
 * dedicated PS5 copy -- own port state, own functions -- so its behaviour can
 * diverge from the PS4 HLE without touching PS4 titles. Registered in the PS5-only
 * registry (MODULE_INIT_PS5); the ps5Layout import resolver force-routes
 * libSceVideoOut here. NIDs decoded from the PPSA03311 (Isaac) eboot import table.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "gfx/gfx.h"
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_event.h"
#include "kern/ps4/lv2/sys_mem.h"  // allocLowGuest

// PS5 present bridge: forwards the flip to the AGC command processor's
// vk::endFrame (gpu/ps5/cmd_processor.cpp).
extern "C" void prosperity_agc_flip(uint64_t scanoutBase);

// The guest address of the display buffer the game most recently flipped
// (sceVideoOutSubmitFlip*'s bufferIndex resolved through the registered-buffer
// table). The PS5 /dev/gc AGC flip ioctls carry no buffer field, so they read
// the scanout target here instead of presenting whichever RT was drawn last.
static std::atomic<uint64_t> g_currentScanout{0};
extern "C" uint64_t prosperity_ps5_scanout_base() {
  return g_currentScanout.load(std::memory_order_relaxed);
}

using namespace krnl;

namespace {

constexpr uint32_t kFmtA8R8G8B8_SRGB = 0x80000000u;
constexpr int16_t kFilterFlip = -10;
constexpr int16_t kFilterVblank = -13;
constexpr int kEventFlip = 0;
constexpr int kEventVblank = 1;

struct ResolutionStatus {
  int32_t width, height, paneWidth, paneHeight;
  uint64_t refreshRate;
  float screenSizeInInch;
  uint16_t flags, reserved0;
  uint32_t reserved1[3];
};

struct FlipStatus {
  uint64_t count, processTime, tsc;
  int64_t flipArg;
  uint64_t submitTsc, reserved0;
  int32_t gcQueueNum, flipPendingNum, currentBuffer;
  uint32_t reserved1;
};

struct VblankStatus {
  uint64_t count, processTime, tsc, reserved[1];
  uint8_t flags, pad[7];
};

// SceVideoOutBufferAttribute, 0x28 bytes (shared layout with PS4).
struct BufferAttribute {
  uint32_t pixelFormat;
  int32_t tilingMode, aspectRatio;
  uint32_t width, height, pitchInPixel, option, reserved0;
  uint64_t reserved1;
};

constexpr int kMaxBuffers = 16;
constexpr int kHandleBase = 1;

struct VideoPort {
  bool open = false;
  int flipRate = 0;
  uint32_t width = 1920, height = 1080, pitch = 1920;
  uint32_t pixelFormat = kFmtA8R8G8B8_SRGB;
  void *buffers[kMaxBuffers] = {};
  int bufferCount = 0;
  std::atomic<uint64_t> flipCount{0};
  std::atomic<uint64_t> submitCount{0};
  int64_t lastFlipArg = -1;
  int currentBuffer = -1;
  int flipEqueue = -1;
  void *flipUdata = nullptr;
  int vblankEqueue = -1;
  void *vblankUdata = nullptr;
};

std::mutex g_mtx;
VideoPort g_port;  // dedicated PS5 port state

// The 16 flip labels sceVideoOutGetBufferLabelAddress hands to the title. They
// must live in GUEST-addressable memory: the title embeds the address in the PM4
// it builds, and the command processor's label range check rightly refuses to
// write host .bss.
uint64_t *videoLabels() {
  static uint64_t *labels =
      reinterpret_cast<uint64_t *>(krnl::allocLowGuest(16 * sizeof(uint64_t)));
  return labels;
}

std::atomic<int> g_gfxState{0};  // 0=untried, 1=up, 2=failed

bool ensureGfx(uint32_t w, uint32_t h) {
  int st = g_gfxState.load();
  if (st == 1) return true;
  if (st == 2) return false;
  std::lock_guard<std::mutex> lk(g_mtx);
  st = g_gfxState.load();
  if (st != 0) return st == 1;
  if (!gfx::init("prosperity", w, h)) {
    std::printf("[videoout/ps5] gfx::init FAILED (no window this run)\n");
    g_gfxState.store(2);
    return false;
  }
  g_gfxState.store(1);
  std::printf("[videoout/ps5] gfx window up (%ux%u)\n", w, h);
  return true;
}

equeue *findEqueue(int handle) {
  auto *p = proc::getActive();
  if (!p) return nullptr;
  auto *obj = p->getObjTable().get(static_cast<uint32_t>(handle));
  if (!obj || obj->type() != kObject::oType::equeue) return nullptr;
  return static_cast<equeue *>(obj);
}

std::atomic<bool> g_flipPumpStarted{false};

// Synthesize flip completion (labels + events) so a title that flips via Gnm/AGC
// and blocks on the flip equeue keeps advancing. Does NOT present (the GPU
// renderer owns the swapchain; presenting here would race it).
void startFlipPump() {
  bool expected = false;
  if (!g_flipPumpStarted.compare_exchange_strong(expected, true)) return;
  std::printf("[videoout/ps5] flip pump started (60 Hz)\n");
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::microseconds(16667));
      uint64_t c = g_port.flipCount.fetch_add(1) + 1;
      // DELTA_VO_NOSTOMP: leave the labels to the title's own GPU fence writes.
      // The blanket stomp satisfies a ">= submitted id" poll, but a title whose
      // protocol is "label == 0 means the buffer is free" reads it as every
      // buffer permanently busy.
      static const bool noStomp = std::getenv("DELTA_VO_NOSTOMP") != nullptr;
      if (!noStomp) {
        uint64_t *labels = videoLabels();
        for (int i = 0; i < 16; i++) labels[i] = c;
      }
      triggerAllEqueues(kEventFlip, kFilterFlip, static_cast<int64_t>(c));
    }
  }).detach();
}

int PS4ABI vOpen(int userId, int busType, int index, const void *) {
  std::printf("[videoout/ps5] open user=%d bus=%d idx=%d\n", userId, busType, index);
  std::lock_guard<std::mutex> lk(g_mtx);
  g_port.open = true;
  return kHandleBase;
}

int PS4ABI vClose(int handle) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_port.open = false;
  return 0;
}

int PS4ABI vGetResolutionStatus(int, void *status) {
  if (!status) return -1;
  auto *s = static_cast<ResolutionStatus *>(status);
  std::memset(s, 0, sizeof(*s));
  s->width = static_cast<int32_t>(g_port.width);
  s->height = static_cast<int32_t>(g_port.height);
  s->paneWidth = s->width;
  s->paneHeight = s->height;
  s->refreshRate = 1;
  s->screenSizeInInch = 50.0f;
  return 0;
}

// Prospero drops PS4's aspectRatio argument, so width/height/pitch each sit one
// register earlier. Read as the PS4 shape, Skyrim's 3840x2160 attribute decoded
// as "2160x0 pitch 0" and every buffer registered at the wrong size.
int PS4ABI vSetBufferAttribute(void *attribute, uint32_t pixelFormat,
                               uint32_t tilingMode, uint32_t width,
                               uint32_t height, uint32_t pitchInPixel) {
  if (!attribute) return -1;
  auto *a = static_cast<BufferAttribute *>(attribute);
  std::memset(a, 0, sizeof(*a));
  a->pixelFormat = pixelFormat;
  a->tilingMode = static_cast<int32_t>(tilingMode);
  a->width = width;
  a->height = height;
  a->pitchInPixel = pitchInPixel;
  std::printf("[videoout/ps5] setBufferAttribute fmt=%#x tiling=%u %ux%u pitch=%u\n",
              pixelFormat, tilingMode, width, height, pitchInPixel);
  return 0;
}

// The PS5 sceVideoOutRegisterBuffers2 `buffers` arg is an array of 32-byte
// SceVideoOutBuffers descriptors (base VA at offset 0), not raw void* pointers
// as on PS4. Stride over the descriptors; the base is the display buffer address.
constexpr int kBufDescStride = 4;  // u64s per descriptor (0x20 bytes)

// PS5 ABI: extra `option` arg before the descriptor array vs PS4.
int PS4ABI vRegisterBuffers(int, int startIndex, int option, void *const *buffers,
                            int bufferNum, const void *attribute) {
  (void)option;
  uint32_t w, h;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (attribute) {
      auto *a = static_cast<const BufferAttribute *>(attribute);
      g_port.width = a->width ? a->width : g_port.width;
      g_port.height = a->height ? a->height : g_port.height;
      g_port.pitch = a->pitchInPixel ? a->pitchInPixel : g_port.width;
      g_port.pixelFormat = a->pixelFormat;
    }
    const uint64_t *desc = reinterpret_cast<const uint64_t *>(buffers);
    int n = 0;
    for (int i = 0; i < bufferNum && (startIndex + i) < kMaxBuffers; i++) {
      g_port.buffers[startIndex + i] =
          desc ? reinterpret_cast<void *>(desc[i * kBufDescStride]) : nullptr;
      n++;
    }
    g_port.bufferCount = startIndex + n;
    w = g_port.width;
    h = g_port.height;
    std::printf("[videoout/ps5] registerBuffers start=%d num=%d -> %ux%u pitch=%u "
                "fmt=%#x (buf0=%p buf1=%p)\n",
                startIndex, bufferNum, g_port.width, g_port.height, g_port.pitch,
                g_port.pixelFormat, g_port.buffers[startIndex],
                bufferNum > 1 ? g_port.buffers[startIndex + 1] : nullptr);
    // A title that flips through AGC never calls sceVideoOutSubmitFlip, so it
    // never names a scanout buffer either. Default to the first one it just
    // registered, or the AGC flip ioctls present a null address.
    if (!g_currentScanout.load(std::memory_order_relaxed))
      g_currentScanout.store(
          reinterpret_cast<uint64_t>(g_port.buffers[startIndex]),
          std::memory_order_relaxed);
  }
  // Registering display buffers is the title committing to present, whichever
  // flip path it uses. Bring the window up here rather than in submitFlip, which
  // the AGC titles never reach: without it there is no swapchain to present to.
  ensureGfx(w, h);
  return 0;
}

int PS4ABI vUnregisterBuffers(int, int) { return 0; }

int PS4ABI vSetFlipRate(int, int rate) {
  g_port.flipRate = rate;
  return 0;
}

int PS4ABI vAddFlipEvent(int eqHandle, int, void *udata) {
  auto *eq = findEqueue(eqHandle);
  if (!eq) return -1;
  g_port.flipEqueue = eqHandle;
  g_port.flipUdata = udata;
  eq->addEvent(static_cast<uint64_t>(kEventFlip), kFilterFlip, udata);
  startFlipPump();
  return 0;
}

int PS4ABI vDeleteFlipEvent(int eqHandle, int) {
  auto *eq = findEqueue(eqHandle);
  if (eq) eq->removeEvent(static_cast<uint64_t>(kEventFlip), kFilterFlip);
  g_port.flipEqueue = -1;
  return 0;
}

int PS4ABI vAddVblankEvent(int eqHandle, int, void *udata) {
  auto *eq = findEqueue(eqHandle);
  if (!eq) return -1;
  g_port.vblankEqueue = eqHandle;
  g_port.vblankUdata = udata;
  eq->addEvent(static_cast<uint64_t>(kEventVblank), kFilterVblank, udata);
  return 0;
}

int PS4ABI vGetEventCount(const void *) { return 1; }

int PS4ABI vGetEventId(const void *event) {
  if (!event) return kEventFlip;
  auto *ev = static_cast<const kevent_t *>(event);
  return ev->filter == kFilterVblank ? kEventVblank : kEventFlip;
}

int PS4ABI vGetEventData(const void *event, int64_t *data) {
  if (!event || !data) return -1;
  *data = static_cast<const kevent_t *>(event)->data;
  return 0;
}

int PS4ABI vSubmitFlip(int, int bufferIndex, int, int64_t flipArg) {
  void *fb = nullptr;
  uint32_t w, h, pitch, fmt;
  int eqHandle;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (bufferIndex >= 0 && bufferIndex < kMaxBuffers)
      fb = g_port.buffers[bufferIndex];
    w = g_port.width; h = g_port.height; pitch = g_port.pitch; fmt = g_port.pixelFormat;
    g_port.currentBuffer = bufferIndex;
    g_currentScanout.store(reinterpret_cast<uint64_t>(fb),
                           std::memory_order_relaxed);
    g_port.lastFlipArg = flipArg;
    g_port.submitCount.fetch_add(1);
  }
  if (fb && ensureGfx(w, h)) {
    auto pf = (fmt & 0x2200u) ? gfx::PixelFormat::rgba8 : gfx::PixelFormat::bgra8;
    gfx::present(fb, w, h, pitch * 4, pf);
    gfx::pumpEvents();
  }
  g_port.flipCount.fetch_add(1);
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    eqHandle = g_port.flipEqueue;
  }
  if (eqHandle >= 0)
    if (auto *eq = findEqueue(eqHandle))
      eq->trigger(kEventFlip, kFilterFlip,
                  static_cast<int64_t>(g_port.flipCount.load()));
  return 0;
}

int PS4ABI vSubmitFlipEop(int, int bufferIndex, int, int64_t flipArg, void *) {
  uint64_t scanout = 0;
  int eqHandle;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (bufferIndex >= 0 && bufferIndex < kMaxBuffers) {
      scanout = reinterpret_cast<uint64_t>(g_port.buffers[bufferIndex]);
      g_port.currentBuffer = bufferIndex;
    }
    g_currentScanout.store(scanout, std::memory_order_relaxed);
    g_port.lastFlipArg = flipArg;
    g_port.submitCount.fetch_add(1);
    eqHandle = g_port.flipEqueue;
  }
  // PS5 always presents through the AGC command processor's render target.
  prosperity_agc_flip(scanout);
  g_port.flipCount.fetch_add(1);
  if (eqHandle >= 0)
    if (auto *eq = findEqueue(eqHandle))
      eq->trigger(kEventFlip, kFilterFlip,
                  static_cast<int64_t>(g_port.flipCount.load()));
  return 0;
}

int PS4ABI vGetFlipStatus(int, void *status) {
  if (!status) return -1;
  auto *s = static_cast<FlipStatus *>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flipCount.load();
  s->flipArg = g_port.lastFlipArg;
  s->currentBuffer = g_port.currentBuffer;
  return 0;
}

int PS4ABI vIsFlipPending(int) { return 0; }

int PS4ABI vGetVblankStatus(int, void *status) {
  if (!status) return -1;
  auto *s = static_cast<VblankStatus *>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flipCount.load();
  return 0;
}

int PS4ABI vWaitVblank(int) { return 0; }

int PS4ABI vGetBufferLabelAddress(int, uintptr_t *label) {
  // Taking the label address means the title drives flips through AGC and waits
  // on these labels rather than on the flip equeue, so it never calls
  // sceVideoOutAddFlipEvent. Start the pump here too, or nothing advances the
  // labels and the first frame waits forever (Minecraft, PPSA17221).
  if (label) *label = reinterpret_cast<uintptr_t>(videoLabels());
  startFlipPump();
  return 0;
}

int PS4ABI vSetWindowModeMargins(int, int, int) { return 0; }
int PS4ABI vColorSettingsSetGamma(void *, float) { return 0; }
int PS4ABI vModeSetAny(int, void *) { return 0; }

}  // namespace

static const runtime::funcInfo functions[] = {
    {0x529DFA3D393AF3B1, (void *)&vOpen},                  // Up36PTk687E
    {0xBAAB951F8FC3BBBF, (void *)&vClose},                 // uquVH4-Du78
    {0xEA43E78F9D53EB66, (void *)&vGetResolutionStatus},   // 6kPnj51T62Y
    {0x8BAFEC47DD56B7FE, (void *)&vSetBufferAttribute},    // i6-sR91Wt-4 (PS4 NID)
    {0x3E34B9B804B0715F, (void *)&vSetBufferAttribute},    // PjS5uASwcV8 (PS5 NID)
    {0xACA054B6046BB5B9, (void *)&vRegisterBuffers},       // rKBUtgRrtbk (PS5 NID+ABI)
    {0x379283B642238C9E, (void *)&vUnregisterBuffers},     // N5KDtkIjjJ4
    {0x0818AEE26084D430, (void *)&vSetFlipRate},           // CBiu4mCE1DA
    {0x1D7CE32BDC88DF49, (void *)&vAddFlipEvent},          // HXzjK9yI30k
    {0xFCECE7D05D401518, (void *)&vDeleteFlipEvent},       // -Ozn0F1AFRg
    {0x5EBBBDDB01C94668, (void *)&vAddVblankEvent},        // Xru92wHJRmg
    {0x32DE101C793190E7, (void *)&vGetEventCount},         // Mt4QHHkxkOc
    {0x536249B52A8D2992, (void *)&vGetEventId},            // U2JJtSqNKZI
    {0xAD651370A7645334, (void *)&vGetEventData},          // rWUTcKdkUzQ
    {0x538E8DC0E889A72B, (void *)&vSubmitFlip},            // U46NwOiJpys
    {0x8FCC65FBDD80D2AE, (void *)&vSubmitFlipEop},         // j8xl+92A0q4
    {0x49B537770A7CD254, (void *)&vGetFlipStatus},         // SbU3dwp80lQ
    {0xCE05E27C74FD12B6, (void *)&vIsFlipPending},         // zgXifHT9ErY
    {0xD456412B2F0778D5, (void *)&vGetVblankStatus},       // 1FZBKy8HeNU
    {0x8FA45A01495A2EFD, (void *)&vWaitVblank},            // j6RaAUlaLv0
    {0x39C4326D07A31C46, (void *)&vGetBufferLabelAddress}, // OcQybQejHEY
    {0x313C71ACE09E4A28, (void *)&vSetWindowModeMargins},  // MTxxrOCeSig
    {0x0D886159B2527918, (void *)&vColorSettingsSetGamma}, // DYhhWbJSeRg
    {0xA63903B20C658BA7, (void *)&vModeSetAny},            // pjkDsgxli6c
};

MODULE_INIT_PS5(libSceVideoOut);

extern "C" int vprx_anchor_ps5_libSceVideoOut = 1;
