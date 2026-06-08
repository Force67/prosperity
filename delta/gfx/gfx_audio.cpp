/*
 * PS4Delta : PS4 emulation and research project
 *
 * Host audio output bridge (SDL3). See gfx_audio.h.
 */

#include "gfx_audio.h"

#if !defined(__ANDROID__)

#include <SDL3/SDL.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
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
  std::lock_guard<std::mutex> lk(g_mtx);
  if (handle < 0 || handle >= static_cast<int>(g_ports.size()))
    return -1;
  Port &p = g_ports[handle];
  if (!p.stream || !samples || !frames)
    return static_cast<int>(frames);
  uint32_t bytes = frames * p.channels * static_cast<uint32_t>(p.bytesPerSample);
  // Bound latency: if the device queue is already many buffers deep (the producer
  // is outrunning playback, e.g. the game is paced by the 60fps video flip rather
  // than by audio), drop this buffer rather than accumulate seconds of delay.
  int queued = SDL_GetAudioStreamQueued(p.stream);
  if (queued >= 0 && static_cast<uint32_t>(queued) < bytes * 8) {
    if (p.gain >= 0.999f) {
      SDL_PutAudioStreamData(p.stream, samples, static_cast<int>(bytes));
    } else {
      // Apply gain into a scratch copy so we never mutate guest memory.
      static thread_local std::vector<uint8_t> scratch;
      scratch.resize(bytes);
      uint32_t n = frames * p.channels;
      if (p.bytesPerSample == 4) {
        const float *src = static_cast<const float *>(samples);
        float *dst = reinterpret_cast<float *>(scratch.data());
        for (uint32_t i = 0; i < n; i++) dst[i] = src[i] * p.gain;
      } else {
        const int16_t *src = static_cast<const int16_t *>(samples);
        int16_t *dst = reinterpret_cast<int16_t *>(scratch.data());
        for (uint32_t i = 0; i < n; i++) dst[i] = static_cast<int16_t>(src[i] * p.gain);
      }
      SDL_PutAudioStreamData(p.stream, scratch.data(), static_cast<int>(bytes));
    }
  }
  if (g_trace && ((g_framesOut += frames) % (48000 * 2) < frames)) {
    // Peak level over this buffer (confirms real audio vs. silence -> tells PCM
    // working from a missing decode upstream).
    float peak = 0.f;
    uint32_t n = frames * p.channels;
    if (p.bytesPerSample == 4) {
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
