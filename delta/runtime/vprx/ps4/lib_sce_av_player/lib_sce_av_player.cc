#include "runtime/vprx/ps4/lib_sce_av_player/lib_sce_av_player.h"
#include "base/arch.h"
#include "guest/session.h"
#include "guest_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "options/options.h"

#include "base/logging.h"

#include "base/atomic.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "cpu/backend.h"

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
// events rather than by polling IsActive never leaves its movie screen
// unless they arrive: Bloodborne opens sprj_opening.mp4, waits for READY, and
// sits on a black frame forever.
u64 g_event_object = 0;
u64 g_event_callback = 0;

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
base::Atomic<i64> g_start_ms{0};

i64 NowMs() {
  return base::TickClock::NowNs() / 1000000;
}

struct PendingEvent {
  i32 event;
  u32 delay_ms;
};

// Runs as a guest thread (see postEvent) so the callback has a real guest
// context and TLS on both backends.
void PS4ABI AvpEventThread(void* arg) {
  auto* pending = static_cast<PendingEvent*>(arg);
  guest::SleepForMilliseconds(pending->delay_ms);
  const i32 event = pending->event;
  if (!g_event_callback)
    return;
  if (kAvpTrace)
    BASE_LOGI("avp", "event {} -> {:#x}", event,
              (unsigned long long)g_event_callback);
  // eventData is null for the state events.
  cpu::GetBackend().RunGuestFunction(g_event_callback, g_event_object,
                                     static_cast<u64>(event), 0, 0);
}

// The real player delivers state events from its own thread, once the call that
// queued them has returned. Delivering inside the call instead runs the title's
// handler while its movie object is still half-built: Bloodborne panics out of
// DLLightMutex with "Mutex is not initialized" and takes a null deref.
void PostEvent(i32 event, u32 delay_ms) {
  if (!g_event_callback)
    return;
  const u64 fsbase = cpu::CurrentGuestFsBase();
  // Create the guest thread on THIS thread (FEX requires it) and only run it on
  // the worker, exactly as sys_thr_new does.
  auto pending = std::make_shared<PendingEvent>(PendingEvent{event, delay_ms});
  void* gthread = cpu::GetBackend().CreateGuestThread(
      cpu::MakeHostThunk(reinterpret_cast<void*>(&AvpEventThread), "avpEvent"),
      pending.get(), fsbase);
  if (!gthread)
    return;
  if (!guest::SpawnThread("libSceAvPlayer", [gthread, pending] {
        cpu::GetBackend().RunGuestThread(gthread);
      }))
    cpu::GetBackend().DiscardGuestThread(gthread);
}

// DELTA_AVP_TRACE: count calls to the hot AvPlayer entrypoints. If a title
// spins on IsActive/GetVideoData (millions of calls) it is WAITING on the movie
// -> our stub must signal "done" some way the title accepts; if it calls each
// once it just skips the movie and the stub is fine.
void AvpTrace(const char* fn) {
  if (!kAvpTrace)
    return;
  static u64 n = 0;
  if ((n++ % 100000) == 0)
    BASE_LOGI("avp", "{} (call #{})", fn, (unsigned long long)n);
}

// The cold entry points: the ORDER a title walks them in is what matters (does
// it enumerate streams? does it ever reach Start?), so log the first few of
// each rather than a sampled count.
void AvpStep(const char* fn) {
  if (!kAvpTrace)
    return;
  BASE_LOGI("avp", "-> {}", fn);
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_event_object);
  guest::ResetResource(g_event_callback);
  guest::ResetResource(g_start_ms);
});
}  // namespace

}  // namespace

// DELTA_AVP_TRACE: dump the init-data block as 16 pointers so the
// event-callback (a guest code pointer) and its offset can be identified ->
// lets us fire video state events the title waits on (Doom64 stalls after Start
// with no IsActive poll).
//
// `eventOff` is where the SceAvPlayerEventReplacement block starts. The Ex form
// of the struct leads with a thisSize field, so its blocks all sit 8 bytes
// later; reading the plain offsets out of it lands on the file replacement's
// size() and on the event object, and calling the latter as the callback jumps
// into a heap object (GTA:SA faults there the moment the movie opens).
static void TakeInitData(const char* fn, const void* init_data, u32 event_off) {
  if (!init_data)
    return;
  auto* p = reinterpret_cast<const u64*>(init_data);
  g_event_object = p[event_off / 8];
  g_event_callback = p[event_off / 8 + 1];
  if (!kAvpTrace)
    return;
  for (int i = 0; i < 16; i++)
    BASE_LOGI("avp", "{} initData[{:#x}]={:#x}", fn, i * 8,
              (unsigned long long)p[i]);
}

i64 PS4ABI sceAvPlayerInit(void* init_data) {
  TakeInitData("Init", init_data, 0x50);
  return kHandle;
}

i64 PS4ABI sceAvPlayerInitEx(const void* init_data, i64* handle_out) {
  TakeInitData("InitEx", init_data, 0x58);
  if (handle_out)
    *handle_out = kHandle;
  return 0;
}

int PS4ABI sceAvPlayerPostInit(i64 /*handle*/, void* /*postInitData*/) {
  AvpStep("PostInit");
  return 0;
}

int PS4ABI sceAvPlayerAddSource(i64 /*handle*/, const char* filename) {
  if (kAvpTrace)
    BASE_LOGI("avp", "AddSource '{}'", filename ? filename : "(null)");
  if (kAvpNoMovie)
    return -1;
  PostEvent(kStateReady, 50);  // the source is open, as far as the title cares
  return 0;
}

int PS4ABI sceAvPlayerAddSourceEx(i64 /*handle*/,
                                  u32 /*type*/,
                                  void* /*source*/) {
  AvpStep("AddSourceEx");
  PostEvent(kStateReady, 50);
  return 0;
}

// A zero-length movie: it starts and ends in the same call, which is what the
// polling contract below (IsActive == false) already tells the title.
int PS4ABI sceAvPlayerStart(i64 /*handle*/) {
  AvpStep("Start");
  g_start_ms.store(NowMs());
  PostEvent(kStatePlay, 10);
  PostEvent(kStateStop, static_cast<u32>(kMovieMs));
  return 0;
}
int PS4ABI sceAvPlayerStop(i64 /*handle*/) {
  AvpStep("Stop");
  g_start_ms.store(0);
  PostEvent(kStateStop, 20);
  return 0;
}
int PS4ABI sceAvPlayerClose(i64 /*handle*/) {
  AvpStep("Close");
  g_start_ms.store(0);
  g_event_callback = 0;
  g_event_object = 0;
  return 0;
}

// Active only for the stub movie's length after Start, so a title that gates on
// this sees playback end instead of a player that was never running.
bool PS4ABI sceAvPlayerIsActive(i64 /*handle*/) {
  AvpTrace("IsActive");
  const i64 start = g_start_ms.load();
  return start != 0 && (u64)(NowMs() - start) < kMovieMs;
}

// No frames are ever produced. The bool contract is "false -> no data this
// call", so callers must not read frameInfo; leave it untouched.
bool PS4ABI sceAvPlayerGetVideoData(i64 /*handle*/, void* /*frameInfo*/) {
  AvpTrace("GetVideoData");
  return false;
}
bool PS4ABI sceAvPlayerGetVideoDataEx(i64 /*handle*/, void* /*frameInfo*/) {
  return false;
}
bool PS4ABI sceAvPlayerGetAudioData(i64 /*handle*/, void* /*frameInfo*/) {
  return false;
}

u64 PS4ABI sceAvPlayerCurrentTime(i64 /*handle*/) {
  const i64 start = g_start_ms.load();
  if (!start)
    return 0;
  const u64 t = static_cast<u64>(NowMs() - start);
  return t < kMovieMs ? t : kMovieMs;
}
int PS4ABI sceAvPlayerSetLooping(i64 /*handle*/, bool /*loop*/) {
  AvpStep("SetLooping");
  return 0;
}

// One video stream. A count of zero looks like "this file has nothing in it"
// and a title that picks a video stream before starting playback then never
// calls Start: GTA:SA opens its intro movie, takes READY, enumerates zero
// streams and leaves the movie layer up, an opaque black rect over the main
// menu, forever. Reporting a stream lets the title enable it, Start, and take
// the end-of-playback that tears the layer down.
int PS4ABI sceAvPlayerStreamCount(i64 /*handle*/) {
  AvpStep("StreamCount");
  return 1;
}

// SceAvPlayerStreamInfo: type, pad, 16 bytes of per-type details, duration and
// startTime in milliseconds. The video details are width/height/aspect and a
// language code.
int PS4ABI sceAvPlayerGetStreamInfo(i64 /*handle*/, u32 stream_id, void* info) {
  if (kAvpTrace)
    BASE_LOGI("avp", "-> GetStreamInfo({})", stream_id);
  if (!info || stream_id != 0)
    return -1;
  auto* u32s = static_cast<u32*>(info);
  std::memset(info, 0, 40);
  u32s[0] = 0;     // SCE_AVPLAYER_VIDEO
  u32s[2] = 1920;  // details.video.width
  u32s[3] = 1080;  // details.video.height
  float aspect = 16.f / 9.f;
  std::memcpy(&u32s[4], &aspect, sizeof(aspect));
  std::memcpy(&u32s[5], "eng", 4);
  auto* u64s = static_cast<u64*>(info);
  u64s[3] = kMovieMs;  // duration
  u64s[4] = 0;         // startTime
  return 0;
}

int PS4ABI sceAvPlayerEnableStream(i64, u32 id) {
  if (kAvpTrace)
    BASE_LOGI("avp", "-> EnableStream({})", id);
  return 0;
}
int PS4ABI sceAvPlayerDisableStream(i64, u32 id) {
  if (kAvpTrace)
    BASE_LOGI("avp", "-> DisableStream({})", id);
  return 0;
}
int PS4ABI sceAvPlayerPause(i64) {
  AvpStep("Pause");
  return 0;
}
int PS4ABI sceAvPlayerResume(i64) {
  AvpStep("Resume");
  return 0;
}
int PS4ABI sceAvPlayerJumpToTime(i64, u64) {
  AvpStep("JumpToTime");
  return 0;
}
int PS4ABI sceAvPlayerSetAvSyncMode(i64, u32) {
  AvpStep("SetAvSyncMode");
  return 0;
}
int PS4ABI sceAvPlayerSetTrickSpeed(i64, int) {
  AvpStep("SetTrickSpeed");
  return 0;
}
