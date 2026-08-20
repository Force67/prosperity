#include "libSceAvPlayer.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utl/options.h>

#include <atomic>
#include <chrono>
#include <thread>

#include <base/logging.h>

#include "cpu/cpu_backend.h"

namespace {
DELTA_OPTION(bool, kAvpTrace, "DELTA_AVP_TRACE", false);
// DELTA_AVP_NOMOVIE=1: fail every AddSource, so the title behaves as if the
// file were unplayable. Separates "the title is stuck on the movie" from "the
// movie is irrelevant and the stall is elsewhere".
DELTA_OPTION(bool, kAvpNoMovie, "DELTA_AVP_NOMOVIE", false);
}  // namespace

// A non-null sentinel handle. The title only ever passes it back to these stubs
// (which ignore it), so any non-null/non-negative value reads as "valid".
namespace {
constexpr i64 kHandle = 1;

// SceAvPlayerInitData carries the title's event callback at +0x50 (the object
// pointer) and +0x58 (the function). A title that drives playback from those
// events -- rather than by polling IsActive -- never leaves its movie screen
// unless they arrive: Bloodborne opens sprj_opening.mp4, waits for READY, and
// sits on a black frame forever.
u64 g_eventObject = 0;
u64 g_eventCallback = 0;

// SceAvPlayerEvents. Only the state ones matter for a movie we never decode.
enum : i32 {
  kStateStop = 0x01,
  kStateReady = 0x02,
  kStatePlay = 0x03,
};

// How long the stub pretends the (never decoded) movie runs. Long enough that a
// title polling IsActive sees a play->stop transition rather than a movie that
// was over before it asked, short enough that nobody waits on it.
constexpr u64 kMovieMs = 200;

// Set by Start; IsActive and CurrentTime answer against it so the three agree.
std::atomic<i64> g_startMs{0};

i64 nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct PendingEvent {
  i32 event;
  u32 delay_ms;
};

// Runs as a guest thread (see postEvent) so the callback has a real guest
// context and TLS on both backends.
void PS4ABI avpEventThread(void *arg) {
  auto *pending = static_cast<PendingEvent *>(arg);
  std::this_thread::sleep_for(std::chrono::milliseconds(pending->delay_ms));
  const i32 event = pending->event;
  delete pending;
  if (!g_eventCallback)
    return;
  if (kAvpTrace)
    BASE_LOGI("avp", "event {} -> {:#x}", event,
              (unsigned long long)g_eventCallback);
  // eventData is null for the state events.
  cpu::backend().runGuestFunction(g_eventCallback, g_eventObject,
                                  static_cast<u64>(event), 0, 0);
}

// The real player delivers state events from its own thread, once the call that
// queued them has returned. Delivering inside the call instead runs the title's
// handler while its movie object is still half-built: Bloodborne panics out of
// DLLightMutex with "Mutex is not initialized" and takes a null deref.
void postEvent(i32 event, u32 delay_ms) {
  if (!g_eventCallback)
    return;
  const u64 fsbase = cpu::currentGuestFsBase();
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
  static u64 n = 0;
  if ((n++ % 100000) == 0)
    BASE_LOGI("avp", "{} (call #{})", fn, (unsigned long long)n);
}

// The cold entry points: the ORDER a title walks them in is what matters (does
// it enumerate streams? does it ever reach Start?), so log the first few of each
// rather than a sampled count.
void avpStep(const char *fn) {
  if (!kAvpTrace) return;
  BASE_LOGI("avp", "-> {}", fn);
}
}

// DELTA_AVP_TRACE: dump the init-data block as 16 pointers so the event-callback
// (a guest code pointer) and its offset can be identified -> lets us fire video
// state events the title waits on (Doom64 stalls after Start with no IsActive poll).
//
// `eventOff` is where the SceAvPlayerEventReplacement block starts. The Ex form
// of the struct leads with a thisSize field, so its blocks all sit 8 bytes
// later; reading the plain offsets out of it lands on the file replacement's
// size() and on the event object, and calling the latter as the callback jumps
// into a heap object (GTA:SA faults there the moment the movie opens).
static void takeInitData(const char *fn, const void *initData, u32 eventOff) {
  if (!initData)
    return;
  auto *p = reinterpret_cast<const u64 *>(initData);
  g_eventObject = p[eventOff / 8];
  g_eventCallback = p[eventOff / 8 + 1];
  if (!kAvpTrace) return;
  for (int i = 0; i < 16; i++)
    BASE_LOGI("avp", "{} initData[{:#x}]={:#x}", fn, i * 8,
              (unsigned long long)p[i]);
}

i64 PS4ABI sceAvPlayerInit(void *initData) {
  takeInitData("Init", initData, 0x50);
  return kHandle;
}

i64 PS4ABI sceAvPlayerInitEx(const void *initData, i64 *handleOut) {
  takeInitData("InitEx", initData, 0x58);
  if (handleOut)
    *handleOut = kHandle;
  return 0;
}

int PS4ABI sceAvPlayerPostInit(i64 /*handle*/, void * /*postInitData*/) {
  avpStep("PostInit");
  return 0;
}

int PS4ABI sceAvPlayerAddSource(i64 /*handle*/, const char *filename) {
  if (kAvpTrace) BASE_LOGI("avp", "AddSource '{}'", filename ? filename : "(null)");
  if (kAvpNoMovie)
    return -1;
  postEvent(kStateReady, 50);  // the source is open, as far as the title cares
  return 0;
}

int PS4ABI sceAvPlayerAddSourceEx(i64 /*handle*/, u32 /*type*/,
                                  void * /*source*/) {
  avpStep("AddSourceEx");
  postEvent(kStateReady, 50);
  return 0;
}

// A zero-length movie: it starts and ends in the same call, which is what the
// polling contract below (IsActive == false) already tells the title.
int PS4ABI sceAvPlayerStart(i64 /*handle*/) {
  avpStep("Start");
  g_startMs.store(nowMs());
  postEvent(kStatePlay, 10);
  postEvent(kStateStop, static_cast<u32>(kMovieMs));
  return 0;
}
int PS4ABI sceAvPlayerStop(i64 /*handle*/) {
  avpStep("Stop");
  g_startMs.store(0);
  postEvent(kStateStop, 20);
  return 0;
}
int PS4ABI sceAvPlayerClose(i64 /*handle*/) {
  avpStep("Close");
  g_startMs.store(0);
  g_eventCallback = 0;
  g_eventObject = 0;
  return 0;
}

// Active only for the stub movie's length after Start, so a title that gates on
// this sees playback end instead of a player that was never running.
bool PS4ABI sceAvPlayerIsActive(i64 /*handle*/) {
  avpTrace("IsActive");
  const i64 start = g_startMs.load();
  return start != 0 && (u64)(nowMs() - start) < kMovieMs;
}

// No frames are ever produced. The bool contract is "false -> no data this
// call", so callers must not read frameInfo; leave it untouched.
bool PS4ABI sceAvPlayerGetVideoData(i64 /*handle*/, void * /*frameInfo*/) {
  avpTrace("GetVideoData");
  return false;
}
bool PS4ABI sceAvPlayerGetVideoDataEx(i64 /*handle*/, void * /*frameInfo*/) {
  return false;
}
bool PS4ABI sceAvPlayerGetAudioData(i64 /*handle*/, void * /*frameInfo*/) {
  return false;
}

u64 PS4ABI sceAvPlayerCurrentTime(i64 /*handle*/) {
  const i64 start = g_startMs.load();
  if (!start)
    return 0;
  const u64 t = static_cast<u64>(nowMs() - start);
  return t < kMovieMs ? t : kMovieMs;
}
int PS4ABI sceAvPlayerSetLooping(i64 /*handle*/, bool /*loop*/) {
  avpStep("SetLooping");
  return 0;
}

// One video stream. A count of zero looks like "this file has nothing in it"
// and a title that picks a video stream before starting playback then never
// calls Start: GTA:SA opens its intro movie, takes READY, enumerates zero
// streams and leaves the movie layer up -- an opaque black rect over the main
// menu, forever. Reporting a stream lets the title enable it, Start, and take
// the end-of-playback that tears the layer down.
int PS4ABI sceAvPlayerStreamCount(i64 /*handle*/) {
  avpStep("StreamCount");
  return 1;
}

// SceAvPlayerStreamInfo: type, pad, 16 bytes of per-type details, duration and
// startTime in milliseconds. The video details are width/height/aspect and a
// language code.
int PS4ABI sceAvPlayerGetStreamInfo(i64 /*handle*/, u32 streamId,
                                    void *info) {
  if (kAvpTrace) BASE_LOGI("avp", "-> GetStreamInfo({})", streamId);
  if (!info || streamId != 0)
    return -1;
  auto *u32s = static_cast<u32 *>(info);
  std::memset(info, 0, 40);
  u32s[0] = 0;     // SCE_AVPLAYER_VIDEO
  u32s[2] = 1920;  // details.video.width
  u32s[3] = 1080;  // details.video.height
  float aspect = 16.f / 9.f;
  std::memcpy(&u32s[4], &aspect, sizeof(aspect));
  std::memcpy(&u32s[5], "eng", 4);
  auto *u64s = static_cast<u64 *>(info);
  u64s[3] = kMovieMs;  // duration
  u64s[4] = 0;         // startTime
  return 0;
}

int PS4ABI sceAvPlayerEnableStream(i64, u32 id) {
  if (kAvpTrace) BASE_LOGI("avp", "-> EnableStream({})", id);
  return 0;
}
int PS4ABI sceAvPlayerDisableStream(i64, u32 id) {
  if (kAvpTrace) BASE_LOGI("avp", "-> DisableStream({})", id);
  return 0;
}
int PS4ABI sceAvPlayerPause(i64) { avpStep("Pause"); return 0; }
int PS4ABI sceAvPlayerResume(i64) { avpStep("Resume"); return 0; }
int PS4ABI sceAvPlayerJumpToTime(i64, u64) { avpStep("JumpToTime"); return 0; }
int PS4ABI sceAvPlayerSetAvSyncMode(i64, u32) {
  avpStep("SetAvSyncMode");
  return 0;
}
int PS4ABI sceAvPlayerSetTrickSpeed(i64, int) {
  avpStep("SetTrickSpeed");
  return 0;
}
