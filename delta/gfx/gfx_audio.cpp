/*
 * PS4Delta : PS4 emulation and research project
 *
 * Host audio output bridge (SDL3). See gfx_audio.h.
 */

#include "gfx_audio.h"

#if !defined(__ANDROID__)

#include <SDL3/SDL.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Port {
  SDL_AudioStream *stream = nullptr;
  uint32_t channels = 2;
  int bytesPerSample = 2;  // S16 = 2, F32 = 4
  float gain = 1.0f;
};

std::mutex g_mtx;
std::vector<Port> g_ports;
bool g_init = false;
const bool g_trace = std::getenv("DELTA_AUDIO_TRACE") != nullptr;
uint64_t g_framesOut = 0;

}  // namespace

extern "C" int prosperity_audio_open(uint32_t freq, uint32_t channels, int isFloat) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!g_init) {
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
      std::fprintf(stderr, "[audio] SDL_InitSubSystem(AUDIO) failed: %s\n", SDL_GetError());
      return -1;
    }
    g_init = true;
  }
  SDL_AudioSpec spec{};
  spec.format = isFloat ? SDL_AUDIO_F32 : SDL_AUDIO_S16;
  spec.channels = static_cast<int>(channels);
  spec.freq = static_cast<int>(freq);
  SDL_AudioStream *s = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
                                                 &spec, nullptr, nullptr);
  if (!s) {
    std::fprintf(stderr, "[audio] open stream failed: %s\n", SDL_GetError());
    return -1;
  }
  SDL_ResumeAudioStreamDevice(s);
  Port p;
  p.stream = s;
  p.channels = channels;
  p.bytesPerSample = isFloat ? 4 : 2;
  int h = static_cast<int>(g_ports.size());
  g_ports.push_back(p);
  if (g_trace)
    std::fprintf(stderr, "[audio] open h=%d %uHz %uch %s\n", h, freq, channels,
                 isFloat ? "f32" : "s16");
  return h;
}

extern "C" int prosperity_audio_output(int handle, const void *samples, uint32_t frames) {
  // Snapshot the port under the lock, then operate lock-free. The SDL stream is
  // owned by SDL (stable), and SDL_Get/PutAudioStreamData are thread-safe, so we
  // must NOT hold g_mtx across the backpressure sleep below: another thread can
  // open a port mid-wait (e.g. AvPlayer's audio thread during an intro movie),
  // reallocating g_ports and dangling a held Port& -> crash.
  SDL_AudioStream *stream;
  int channels, bytesPerSample;
  float gain;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (handle < 0 || handle >= static_cast<int>(g_ports.size()))
      return -1;
    Port &p = g_ports[handle];
    if (!p.stream || !samples || !frames)
      return static_cast<int>(frames);
    stream = p.stream;
    channels = p.channels;
    bytesPerSample = p.bytesPerSample;
    gain = p.gain;
  }
  uint32_t bytes = frames * channels * static_cast<uint32_t>(bytesPerSample);
  // Backpressure: the real sceAudioOutOutput blocks until the previous grain has
  // played, which is what paces a title's audio thread to real time. Our queue is
  // non-blocking, so an audio thread (e.g. FMOD's mixer/output threads) that loops
  // on Output runs flat-out, pinning a core at ~100% and starving the game thread.
  // Block here until the device queue drains below a small target so the producer
  // tracks playback. Bounded so a stalled/absent device (headless) can't hang the
  // thread: if it doesn't drain within ~a buffer's worth of time, fall through.
  const uint32_t target = bytes * 3;  // keep ~3 buffers of latency
  for (int i = 0; i < 64; i++) {
    int q = SDL_GetAudioStreamQueued(stream);  // thread-safe
    if (q < 0 || static_cast<uint32_t>(q) <= target)
      break;
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  // Bound latency: if the device queue is STILL many buffers deep after the wait
  // (no real playback to pace against, e.g. headless), drop this buffer rather than
  // accumulate seconds of delay.
  int queued = SDL_GetAudioStreamQueued(stream);
  if (queued >= 0 && static_cast<uint32_t>(queued) < bytes * 8) {
    if (gain >= 0.999f) {
      SDL_PutAudioStreamData(stream, samples, static_cast<int>(bytes));
    } else {
      // Apply gain into a scratch copy so we never mutate guest memory.
      static thread_local std::vector<uint8_t> scratch;
      scratch.resize(bytes);
      uint32_t n = frames * channels;
      if (bytesPerSample == 4) {
        const float *src = static_cast<const float *>(samples);
        float *dst = reinterpret_cast<float *>(scratch.data());
        for (uint32_t i = 0; i < n; i++) dst[i] = src[i] * gain;
      } else {
        const int16_t *src = static_cast<const int16_t *>(samples);
        int16_t *dst = reinterpret_cast<int16_t *>(scratch.data());
        for (uint32_t i = 0; i < n; i++) dst[i] = static_cast<int16_t>(src[i] * gain);
      }
      SDL_PutAudioStreamData(stream, scratch.data(), static_cast<int>(bytes));
    }
  }
  if (g_trace && ((g_framesOut += frames) % (48000 * 2) < frames)) {
    // Peak level over this buffer (confirms real audio vs. silence -> tells PCM
    // working from a missing decode upstream).
    float peak = 0.f;
    uint32_t n = frames * channels;
    if (bytesPerSample == 4) {
      const float *s = static_cast<const float *>(samples);
      for (uint32_t i = 0; i < n; i++) { float a = s[i] < 0 ? -s[i] : s[i]; if (a > peak) peak = a; }
    } else {
      const int16_t *s = static_cast<const int16_t *>(samples);
      for (uint32_t i = 0; i < n; i++) { float a = (s[i] < 0 ? -s[i] : s[i]) / 32768.f; if (a > peak) peak = a; }
    }
    std::fprintf(stderr, "[audio] h=%d ~%lu frames out, queued=%d peak=%.3f\n", handle,
                 (unsigned long)g_framesOut, queued, peak);
  }
  return static_cast<int>(frames);
}

extern "C" void prosperity_audio_volume(int handle, float gain) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (handle < 0 || handle >= static_cast<int>(g_ports.size())) return;
  g_ports[handle].gain = gain < 0.f ? 0.f : gain > 1.f ? 1.f : gain;
}

extern "C" void prosperity_audio_close(int handle) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (handle < 0 || handle >= static_cast<int>(g_ports.size())) return;
  if (g_ports[handle].stream) {
    SDL_DestroyAudioStream(g_ports[handle].stream);
    g_ports[handle].stream = nullptr;
  }
}

#else  // __ANDROID__ : SDL is not linked into the gfx build; no-op for now.

extern "C" int prosperity_audio_open(uint32_t, uint32_t, int) { return -1; }
extern "C" int prosperity_audio_output(int, const void *, uint32_t frames) {
  return static_cast<int>(frames);
}
extern "C" void prosperity_audio_volume(int, float) {}
extern "C" void prosperity_audio_close(int) {}

#endif
