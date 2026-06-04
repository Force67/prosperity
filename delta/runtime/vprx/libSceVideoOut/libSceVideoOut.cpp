/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceVideoOut implementation. See libSceVideoOut.h for why this overrides
 * the LLE module.
 */

#include "libSceVideoOut.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

#include "gfx/gfx.h"
#include "kern/proc.h"
#include "kern/lv2/sys_event.h"

using namespace krnl;

namespace {

// SCE pixel formats we care about. A8R8G8B8 is BGRA in little-endian memory;
// A8B8G8R8 is RGBA. The high bit marks the family, bit 0x2200 the channel order.
constexpr uint32_t kFmtA8R8G8B8_SRGB = 0x80000000u;

// PS4 videoout kernel-event filter. Real EVFILT_DISPLAY is -13 (the vblank pump
// already uses it). Flip completions are a distinct source, so give them their
// own filter so the 60 Hz vblank pump never spuriously fires a flip knote.
constexpr int16_t kFilterFlip = -10;
constexpr int16_t kFilterVblank = -13;

// videoout event ids returned by sceVideoOutGetEventId.
constexpr int kEventFlip = 0;
constexpr int kEventVblank = 1;

// SceVideoOutResolutionStatus, 0x30 bytes.
struct ResolutionStatus {
  int32_t width;
  int32_t height;
  int32_t paneWidth;
  int32_t paneHeight;
  uint64_t refreshRate;
  float screenSizeInInch;
  uint16_t flags;
  uint16_t reserved0;
  uint32_t reserved1[3];
};

// SceVideoOutFlipStatus, 0x40 bytes.
struct FlipStatus {
  uint64_t count;
  uint64_t processTime;
  uint64_t tsc;
  int64_t flipArg;
  uint64_t submitTsc;
  uint64_t reserved0;
  int32_t gcQueueNum;
  int32_t flipPendingNum;
  int32_t currentBuffer;
  uint32_t reserved1;
};

// SceVideoOutVblankStatus, 0x30 bytes.
struct VblankStatus {
  uint64_t count;
  uint64_t processTime;
  uint64_t tsc;
  uint64_t reserved[1];
  uint8_t flags;
  uint8_t pad[7];
};

// SceVideoOutBufferAttribute, 0x28 bytes.
struct BufferAttribute {
  uint32_t pixelFormat;
  int32_t tilingMode;
  int32_t aspectRatio;
  uint32_t width;
  uint32_t height;
  uint32_t pitchInPixel;
  uint32_t option;
  uint32_t reserved0;
  uint64_t reserved1;
};

constexpr int kMaxBuffers = 16;
constexpr int kHandleBase = 1;

struct VideoPort {
  bool open = false;
  int flipRate = 0;
  uint32_t width = 1920;
  uint32_t height = 1080;
  uint32_t pitch = 1920;       // in pixels
  uint32_t pixelFormat = kFmtA8R8G8B8_SRGB;
  void *buffers[kMaxBuffers] = {};
  int bufferCount = 0;

  // flip bookkeeping (read back via sceVideoOutGetFlipStatus).
  std::atomic<uint64_t> flipCount{0};
  std::atomic<uint64_t> submitCount{0};
  int64_t lastFlipArg = -1;
  int currentBuffer = -1;

  // equeue (by handle) a flip/vblank event was registered on, so SubmitFlip can
  // wake exactly that queue. Isaac uses one display port + one equeue.
  int flipEqueue = -1;
  void *flipUdata = nullptr;
  int vblankEqueue = -1;
  void *vblankUdata = nullptr;

  // a 16-byte guest label region per port (sceVideoOutGetBufferLabelAddress).
  uint64_t labels[16] = {};
};

std::mutex g_mtx;
VideoPort g_port;            // single display port is enough for Isaac
std::atomic<bool> g_gfxUp{false};

std::atomic<int> g_gfxState{0};  // 0=untried, 1=up, 2=failed

bool ensureGfx(uint32_t w, uint32_t h) {
  int st = g_gfxState.load();
  if (st == 1)
    return true;
  if (st == 2)
    return false;  // tried once and failed; don't spam retries every frame
  std::lock_guard<std::mutex> lk(g_mtx);
  st = g_gfxState.load();
  if (st != 0)
    return st == 1;
  if (!gfx::init("prosperity - The Binding of Isaac", w, h)) {
    std::printf("[videoout] gfx::init FAILED (no window this run)\n");
    g_gfxState.store(2);
    return false;
  }
  g_gfxState.store(1);
  g_gfxUp.store(true);
  std::printf("[videoout] gfx window up (%ux%u)\n", w, h);
  return true;
}

equeue *findEqueue(int handle) {
  auto *p = proc::getActive();
  if (!p)
    return nullptr;
  auto *obj = p->getObjTable().get(static_cast<uint32_t>(handle));
  if (!obj || obj->type() != kObject::oType::equeue)
    return nullptr;
  return static_cast<equeue *>(obj);
}

// Present the most recently flipped scanout buffer to the window.
void presentScanout() {
  void *fb;
  uint32_t w, h, pitch, fmt;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    int idx = g_port.currentBuffer >= 0 ? g_port.currentBuffer : 0;
    fb = (idx < kMaxBuffers) ? g_port.buffers[idx] : nullptr;
    w = g_port.width;
    h = g_port.height;
    pitch = g_port.pitch;
    fmt = g_port.pixelFormat;
  }
  if (!fb || !ensureGfx(w, h))
    return;
  auto pf = (fmt & 0x2200u) ? gfx::PixelFormat::rgba8 : gfx::PixelFormat::bgra8;
  gfx::present(fb, w, h, pitch * 4, pf);
  gfx::pumpEvents();
}

std::atomic<bool> g_flipPumpStarted{false};

// The game submits flips through Gnm (a PM4 prepareFlip packet to the GPU), not
// sceVideoOutSubmitFlip, and then blocks in kevent on the equeue it registered
// with sceVideoOutAddFlipEvent waiting for flip completion. We don't run the GPU
// yet, so synthesize that completion: a ~60 Hz pump that presents the current
// scanout buffer and posts the flip event to every equeue that registered one.
void startFlipPump() {
  bool expected = false;
  if (!g_flipPumpStarted.compare_exchange_strong(expected, true))
    return;
  std::printf("[videoout] flip pump started (60 Hz)\n");
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::microseconds(16667));
      // NB: do NOT present here. The window is driven solely by the GPU
      // renderer (Gnm submit -> gpu::endFrame -> gfx::present) on the submit
      // thread. gfx has one swapchain/command buffer and is not thread-safe, so
      // a present from this pump thread races the renderer's present and
      // intermittently deadlocks Vulkan. This pump only synthesizes flip
      // completion (labels + events) to unblock the game's flip wait; the guest
      // scanout buffer it used to blit is never CPU-written by the title anyway.
      uint64_t c = g_port.flipCount.fetch_add(1) + 1;
      // Mark GPU/flip completion in the buffer labels. Gnm's prepareFlip packet
      // tells the GPU to write the buffer's label (sceVideoOutGetBufferLabelAddress
      // + bufferIndex*8) when the flip completes; the game busy-polls that label
      // to recycle buffers. With no real GPU we write it ourselves: a monotonic
      // value (the flip count) satisfies the ">= submitted id" poll so the game
      // stops waiting and submits the next frame.
      for (int i = 0; i < 16; i++)
        g_port.labels[i] = c;
      // post the flip-complete event to whichever equeue holds a flip knote.
      triggerAllEqueues(kEventFlip, kFilterFlip, static_cast<int64_t>(c));
    }
  }).detach();
}

}  // namespace

extern "C" {

int PS4ABI sceVideoOutOpen(int userId, int busType, int index, const void *param) {
  std::printf("[videoout] open user=%d bus=%d idx=%d\n", userId, busType, index);
  std::lock_guard<std::mutex> lk(g_mtx);
  g_port.open = true;
  // bring the window up early so the user sees something while the game inits.
  // (do it outside the lock-sensitive gfx path on first flip if init is heavy)
  return kHandleBase;
}

int PS4ABI sceVideoOutClose(int handle) {
  std::printf("[videoout] close h=%d\n", handle);
  std::lock_guard<std::mutex> lk(g_mtx);
  g_port.open = false;
  return 0;
}

int PS4ABI sceVideoOutGetResolutionStatus(int handle, void *status) {
  if (!status)
    return -1;
  auto *s = static_cast<ResolutionStatus *>(status);
  std::memset(s, 0, sizeof(*s));
  s->width = static_cast<int32_t>(g_port.width);
  s->height = static_cast<int32_t>(g_port.height);
  s->paneWidth = s->width;
  s->paneHeight = s->height;
  s->refreshRate = 1;  // SCE_VIDEO_OUT_REFRESH_RATE_59_94HZ
  s->screenSizeInInch = 50.0f;
  s->flags = 0;
  return 0;
}

int PS4ABI sceVideoOutSetBufferAttribute(void *attribute, uint32_t pixelFormat,
                                         uint32_t tilingMode, uint32_t aspectRatio,
                                         uint32_t width, uint32_t height,
                                         uint32_t pitchInPixel) {
  if (!attribute)
    return -1;
  auto *a = static_cast<BufferAttribute *>(attribute);
  std::memset(a, 0, sizeof(*a));
  a->pixelFormat = pixelFormat;
  a->tilingMode = static_cast<int32_t>(tilingMode);
  a->aspectRatio = static_cast<int32_t>(aspectRatio);
  a->width = width;
  a->height = height;
  a->pitchInPixel = pitchInPixel;
  std::printf("[videoout] setBufferAttribute fmt=%#x tiling=%u %ux%u pitch=%u\n",
              pixelFormat, tilingMode, width, height, pitchInPixel);
  return 0;
}

int PS4ABI sceVideoOutRegisterBuffers(int handle, int startIndex,
                                     void *const *addresses, int bufferNum,
                                     const void *attribute) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (attribute) {
    auto *a = static_cast<const BufferAttribute *>(attribute);
    g_port.width = a->width ? a->width : g_port.width;
    g_port.height = a->height ? a->height : g_port.height;
    g_port.pitch = a->pitchInPixel ? a->pitchInPixel : g_port.width;
    g_port.pixelFormat = a->pixelFormat;
  }
  int n = 0;
  for (int i = 0; i < bufferNum && (startIndex + i) < kMaxBuffers; i++) {
    g_port.buffers[startIndex + i] = addresses ? addresses[i] : nullptr;
    n++;
  }
  g_port.bufferCount = startIndex + n;
  std::printf("[videoout] registerBuffers start=%d num=%d -> %ux%u pitch=%u "
              "fmt=%#x (buf0=%p)\n",
              startIndex, bufferNum, g_port.width, g_port.height, g_port.pitch,
              g_port.pixelFormat, addresses ? addresses[0] : nullptr);
  return 0;
}

int PS4ABI sceVideoOutUnregisterBuffers(int handle, int attributeIndex) {
  std::printf("[videoout] unregisterBuffers idx=%d\n", attributeIndex);
  return 0;
}

int PS4ABI sceVideoOutSetFlipRate(int handle, int rate) {
  std::printf("[videoout] setFlipRate %d\n", rate);
  g_port.flipRate = rate;
  return 0;
}

int PS4ABI sceVideoOutAddFlipEvent(int eqHandle, int handle, void *udata) {
  std::printf("[videoout] addFlipEvent eq=%d h=%d udata=%p\n", eqHandle, handle,
              udata);
  auto *eq = findEqueue(eqHandle);
  if (!eq)
    return -1;
  g_port.flipEqueue = eqHandle;
  g_port.flipUdata = udata;
  eq->addEvent(static_cast<uint64_t>(kEventFlip), kFilterFlip, udata);
  startFlipPump();
  return 0;
}

int PS4ABI sceVideoOutDeleteFlipEvent(int eqHandle, int handle) {
  std::printf("[videoout] deleteFlipEvent eq=%d h=%d\n", eqHandle, handle);
  auto *eq = findEqueue(eqHandle);
  if (eq)
    eq->removeEvent(static_cast<uint64_t>(kEventFlip), kFilterFlip);
  g_port.flipEqueue = -1;
  return 0;
}

int PS4ABI sceVideoOutAddVblankEvent(int eqHandle, int handle, void *udata) {
  std::printf("[videoout] addVblankEvent eq=%d h=%d udata=%p\n", eqHandle, handle,
              udata);
  auto *eq = findEqueue(eqHandle);
  if (!eq)
    return -1;
  g_port.vblankEqueue = eqHandle;
  g_port.vblankUdata = udata;
  // vblank rides the existing 60 Hz EVFILT_DISPLAY pump (ident wildcard).
  eq->addEvent(static_cast<uint64_t>(kEventVblank), kFilterVblank, udata);
  return 0;
}

int PS4ABI sceVideoOutGetEventCount(const void *event) {
  return 1;
}

int PS4ABI sceVideoOutGetEventId(const void *event) {
  if (!event)
    return kEventFlip;
  // event is a SceKernelEvent (kevent_t). distinguish by filter.
  auto *ev = static_cast<const kevent_t *>(event);
  if (ev->filter == kFilterVblank)
    return kEventVblank;
  return kEventFlip;
}

int PS4ABI sceVideoOutGetEventData(const void *event, int64_t *data) {
  if (!event || !data)
    return -1;
  auto *ev = static_cast<const kevent_t *>(event);
  *data = ev->data;
  return 0;
}

int PS4ABI sceVideoOutSubmitFlip(int handle, int bufferIndex, int flipMode,
                                int64_t flipArg) {
  void *fb = nullptr;
  uint32_t w, h, pitch, fmt;
  int eqHandle;
  void *udata;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (bufferIndex >= 0 && bufferIndex < kMaxBuffers)
      fb = g_port.buffers[bufferIndex];
    w = g_port.width;
    h = g_port.height;
    pitch = g_port.pitch;
    fmt = g_port.pixelFormat;
    g_port.currentBuffer = bufferIndex;
    g_port.lastFlipArg = flipArg;
    g_port.submitCount.fetch_add(1);
  }

  // present the scanout buffer (guest pointers are identity-mapped, so the
  // guest address is directly readable on the host). Until the Gnm->Vulkan
  // path detiles real GPU output this is the linear scanout contents.
  if (fb && ensureGfx(w, h)) {
    auto pf = (fmt & 0x2200u) ? gfx::PixelFormat::rgba8 : gfx::PixelFormat::bgra8;
    gfx::present(fb, w, h, pitch * 4, pf);
    gfx::pumpEvents();
  }

  // flip "completes" immediately: bump the count and wake the flip equeue.
  g_port.flipCount.fetch_add(1);
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    eqHandle = g_port.flipEqueue;
    udata = g_port.flipUdata;
  }
  if (eqHandle >= 0) {
    if (auto *eq = findEqueue(eqHandle))
      eq->trigger(kEventFlip, kFilterFlip,
                  static_cast<int64_t>(g_port.flipCount.load()));
  }
  return 0;
}

int PS4ABI sceVideoOutGetFlipStatus(int handle, void *status) {
  if (!status)
    return -1;
  auto *s = static_cast<FlipStatus *>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flipCount.load();
  s->flipArg = g_port.lastFlipArg;
  s->currentBuffer = g_port.currentBuffer;
  s->gcQueueNum = 0;
  // pending = submitted but not yet "completed"; we complete synchronously.
  s->flipPendingNum = 0;
  return 0;
}

int PS4ABI sceVideoOutIsFlipPending(int handle) {
  return 0;  // never pending: flips complete synchronously
}

int PS4ABI sceVideoOutGetVblankStatus(int handle, void *status) {
  if (!status)
    return -1;
  auto *s = static_cast<VblankStatus *>(status);
  std::memset(s, 0, sizeof(*s));
  s->count = g_port.flipCount.load();
  s->flags = 0;
  return 0;
}

int PS4ABI sceVideoOutWaitVblank(int handle) {
  return 0;
}

int PS4ABI sceVideoOutGetBufferLabelAddress(int handle, uintptr_t *label) {
  // Gnm's flip path checks `eax == 0` for success (then reads *label to build a
  // GPU completion-label write). Returning the slot count here makes Gnm treat
  // the flip request as failed ("flip request failed"). Return 0 = success.
  if (label)
    *label = reinterpret_cast<uintptr_t>(&g_port.labels[0]);
  return 0;
}

int PS4ABI sceVideoOutSetWindowModeMargins(int handle, int top, int bottom) {
  return 0;
}

int PS4ABI sceVideoOutColorSettingsSetGamma_(void *settings, float gamma) {
  return 0;
}

int PS4ABI sceVideoOutModeSetAny_(int handle, void *arg) {
  return 0;
}

// Bridge for the HLE Gnm driver: the game flips via sceGnmSubmitAndFlip-
// CommandBuffers (a GPU prepareFlip packet), so record the target scanout buffer
// here; the flip pump then presents it and posts the flip-complete event.
void prosperity_videoout_set_flip(int bufferIndex, int64_t flipArg) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (bufferIndex >= 0 && bufferIndex < kMaxBuffers)
    g_port.currentBuffer = bufferIndex;
  g_port.lastFlipArg = flipArg;
}

// The guest scanout buffer address for a registered buffer index (the GPU
// render target the flip displays). Used by the GPU renderer to present the
// right render target.
uint64_t prosperity_videoout_buffer(int bufferIndex) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (bufferIndex >= 0 && bufferIndex < kMaxBuffers)
    return reinterpret_cast<uint64_t>(g_port.buffers[bufferIndex]);
  return 0;
}

}  // extern "C"
