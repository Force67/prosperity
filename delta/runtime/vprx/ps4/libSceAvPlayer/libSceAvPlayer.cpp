#include "libSceAvPlayer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utl/options.h>

#include <chrono>
#include <thread>

#include "cpu/cpu_backend.h"

namespace {
DELTA_OPTION(bool, kAvpTrace, "DELTA_AVP_TRACE", false);
}  // namespace

// A non-null sentinel handle. The title only ever passes it back to these stubs
// (which ignore it), so any non-null/non-negative value reads as "valid".
namespace {
constexpr int64_t kHandle = 1;

// SceAvPlayerInitData carries the title's event callback at +0x50 (the object
// pointer) and +0x58 (the function). A title that drives playback from those
// events -- rather than by polling IsActive -- never leaves its movie screen
// unless they arrive: Bloodborne opens sprj_opening.mp4, waits for READY, and
// sits on a black frame forever.
uint64_t g_eventObject = 0;
uint64_t g_eventCallback = 0;

// SceAvPlayerEvents. Only the state ones matter for a movie we never decode.
enum : int32_t {
  kStateStop = 0x01,
  kStateReady = 0x02,
  kStatePlay = 0x03,
};

struct PendingEvent {
  int32_t event;
  uint32_t delay_ms;
};

// Runs as a guest thread (see postEvent) so the callback has a real guest
// context and TLS on both backends.
void PS4ABI avpEventThread(void *arg) {
  auto *pending = static_cast<PendingEvent *>(arg);
  std::this_thread::sleep_for(std::chrono::milliseconds(pending->delay_ms));
  const int32_t event = pending->event;
  delete pending;
  if (!g_eventCallback)
    return;
  if (kAvpTrace)
    std::fprintf(stderr, "[avp] event %d -> %#llx\n", event,
                 (unsigned long long)g_eventCallback);
  // eventData is null for the state events.
  cpu::backend().runGuestFunction(g_eventCallback, g_eventObject,
                                  static_cast<uint64_t>(event), 0, 0);
}

// The real player delivers state events from its own thread, once the call that
// queued them has returned. Delivering inside the call instead runs the title's
// handler while its movie object is still half-built: Bloodborne panics out of
// DLLightMutex with "Mutex is not initialized" and takes a null deref.
void postEvent(int32_t event, uint32_t delay_ms) {
  if (!g_eventCallback)
    return;
  const uint64_t fsbase = cpu::currentGuestFsBase();
  // Create the guest thread on THIS thread (FEX requires it) and only run it on
  // the worker, exactly as sys_thr_new does.
  void *gthread = cpu::backend().createGuestThread(
      cpu::makeHostThunk(reinterpret_cast<void *>(&avpEventThread),
                         "avpEvent"),
      new PendingEvent{event, delay_ms}, fsbase);
  if (!gthread)
    return;
  std::thread([gthread] { cpu::backend().runGuestThread(gthread); }).detach();
}

// DELTA_AVP_TRACE: count calls to the hot AvPlayer entrypoints. If a title spins
// on IsActive/GetVideoData (millions of calls) it is WAITING on the movie -> our
// stub must signal "done" some way the title accepts; if it calls each once it
// just skips the movie and the stub is fine.
void avpTrace(const char *fn) {
  if (!kAvpTrace) return;
  static uint64_t n = 0;
  if ((n++ % 100000) == 0)
    std::fprintf(stderr, "[avp] %s (call #%llu)\n", fn, (unsigned long long)n);
}
}

// DELTA_AVP_TRACE: dump the init-data block as 16 pointers so the event-callback
// (a guest code pointer) and its offset can be identified -> lets us fire video
// state events the title waits on (Doom64 stalls after Start with no IsActive poll).
static void takeInitData(const char *fn, const void *initData) {
  if (!initData)
    return;
  auto *p = reinterpret_cast<const uint64_t *>(initData);
  g_eventObject = p[0x50 / 8];
  g_eventCallback = p[0x58 / 8];
  if (!kAvpTrace) return;
  for (int i = 0; i < 16; i++)
    std::fprintf(stderr, "[avp] %s initData[%#x]=%#llx\n", fn, i * 8,
                 (unsigned long long)p[i]);
}

int64_t PS4ABI sceAvPlayerInit(void *initData) {
  takeInitData("Init", initData);
  return kHandle;
}

int64_t PS4ABI sceAvPlayerInitEx(const void *initData, int64_t *handleOut) {
  takeInitData("InitEx", initData);
  if (handleOut)
    *handleOut = kHandle;
  return 0;
}

int PS4ABI sceAvPlayerPostInit(int64_t /*handle*/, void * /*postInitData*/) {
  return 0;
}

int PS4ABI sceAvPlayerAddSource(int64_t /*handle*/, const char *filename) {
  if (kAvpTrace) std::fprintf(stderr, "[avp] AddSource '%s'\n", filename ? filename : "(null)");
  postEvent(kStateReady, 50);  // the source is open, as far as the title cares
  return 0;
}

int PS4ABI sceAvPlayerAddSourceEx(int64_t /*handle*/, uint32_t /*type*/,
                                  void * /*source*/) {
  postEvent(kStateReady, 50);
  return 0;
}

// A zero-length movie: it starts and ends in the same call, which is what the
// polling contract below (IsActive == false) already tells the title.
int PS4ABI sceAvPlayerStart(int64_t /*handle*/) {
  postEvent(kStatePlay, 50);
  postEvent(kStateStop, 150);
  return 0;
}
int PS4ABI sceAvPlayerStop(int64_t /*handle*/) {
  postEvent(kStateStop, 20);
  return 0;
}
int PS4ABI sceAvPlayerClose(int64_t /*handle*/) {
  g_eventCallback = 0;
  g_eventObject = 0;
  return 0;
}

// The key stub: report no active playback so the title's frame loop is skipped.
bool PS4ABI sceAvPlayerIsActive(int64_t /*handle*/) { avpTrace("IsActive"); return false; }

// No frames are ever produced. The bool contract is "false -> no data this
// call", so callers must not read frameInfo; leave it untouched.
bool PS4ABI sceAvPlayerGetVideoData(int64_t /*handle*/, void * /*frameInfo*/) {
  avpTrace("GetVideoData");
  return false;
}
bool PS4ABI sceAvPlayerGetVideoDataEx(int64_t /*handle*/, void * /*frameInfo*/) {
  return false;
}
bool PS4ABI sceAvPlayerGetAudioData(int64_t /*handle*/, void * /*frameInfo*/) {
  return false;
}

uint64_t PS4ABI sceAvPlayerCurrentTime(int64_t /*handle*/) { return 0; }
int PS4ABI sceAvPlayerSetLooping(int64_t /*handle*/, bool /*loop*/) { return 0; }

// No streams in the (absent) movie. With a zero count the title skips its
// per-stream enable/info enumeration.
int PS4ABI sceAvPlayerStreamCount(int64_t /*handle*/) { return 0; }
int PS4ABI sceAvPlayerGetStreamInfo(int64_t /*handle*/, uint32_t /*streamId*/,
                                    void * /*info*/) {
  return -1;
}

int PS4ABI sceAvPlayerControlOk() { avpTrace("ControlOk"); return 0; }
