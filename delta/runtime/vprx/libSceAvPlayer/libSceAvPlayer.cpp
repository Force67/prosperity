#include "libSceAvPlayer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// A non-null sentinel handle. The title only ever passes it back to these stubs
// (which ignore it), so any non-null/non-negative value reads as "valid".
namespace {
constexpr int64_t kHandle = 1;

// DELTA_AVP_TRACE: count calls to the hot AvPlayer entrypoints. If a title spins
// on IsActive/GetVideoData (millions of calls) it is WAITING on the movie -> our
// stub must signal "done" some way the title accepts; if it calls each once it
// just skips the movie and the stub is fine.
void avpTrace(const char *fn) {
  static const bool on = std::getenv("DELTA_AVP_TRACE") != nullptr;
  if (!on) return;
  static uint64_t n = 0;
  if ((n++ % 100000) == 0)
    std::fprintf(stderr, "[avp] %s (call #%llu)\n", fn, (unsigned long long)n);
}
}

int64_t PS4ABI sceAvPlayerInit(void * /*initData*/) { return kHandle; }

int64_t PS4ABI sceAvPlayerInitEx(const void * /*initData*/, int64_t *handleOut) {
  if (handleOut)
    *handleOut = kHandle;
  return 0;
}

int PS4ABI sceAvPlayerPostInit(int64_t /*handle*/, void * /*postInitData*/) {
  return 0;
}

int PS4ABI sceAvPlayerAddSource(int64_t /*handle*/, const char *filename) {
  static const bool on = std::getenv("DELTA_AVP_TRACE") != nullptr;
  if (on) std::fprintf(stderr, "[avp] AddSource '%s'\n", filename ? filename : "(null)");
  return 0;
}

int PS4ABI sceAvPlayerAddSourceEx(int64_t /*handle*/, uint32_t /*type*/,
                                  void * /*source*/) {
  return 0;
}

int PS4ABI sceAvPlayerStart(int64_t /*handle*/) { return 0; }
int PS4ABI sceAvPlayerStop(int64_t /*handle*/) { return 0; }
int PS4ABI sceAvPlayerClose(int64_t /*handle*/) { return 0; }

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
