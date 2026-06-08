/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceAudioOut. See libSceAudioOut.h. Bridges to the host SDL3 device via
 * delta_gfx's gfx_audio. Each open port records its grain/format so Output knows
 * how many interleaved frames the guest buffer holds.
 */

#include "libSceAudioOut.h"

#include "gfx/gfx_audio.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

struct Port {
  int bridge = -1;     // gfx_audio handle
  uint32_t grain = 0;  // samples per channel per Output (the open `length`)
  uint32_t channels = 2;
  bool open = false;
};

std::mutex g_mtx;
std::vector<Port> g_ports;  // SCE handle = index + 1

// SceAudioOutParamFormat (param low byte) -> (channels, isFloat).
void decodeFormat(uint32_t param, uint32_t &channels, int &isFloat) {
  switch (param & 0xFF) {
  case 0: channels = 1; isFloat = 0; break;  // S16 mono
  case 1: channels = 2; isFloat = 0; break;  // S16 stereo
  case 2: channels = 8; isFloat = 0; break;  // S16 8ch
  case 3: channels = 1; isFloat = 1; break;  // float mono
  case 4: channels = 2; isFloat = 1; break;  // float stereo
  case 5: channels = 8; isFloat = 1; break;  // float 8ch
  case 6: channels = 8; isFloat = 0; break;  // S16 8ch std
  case 7: channels = 8; isFloat = 1; break;  // float 8ch std
  default: channels = 2; isFloat = 0; break;
  }
}

Port *port(int32_t handle) {
  if (handle <= 0 || handle > static_cast<int32_t>(g_ports.size())) return nullptr;
  Port &p = g_ports[handle - 1];
  return p.open ? &p : nullptr;
}

}  // namespace

extern "C" {

int PS4ABI sceAudioOutInit() { return 0; }

int PS4ABI sceAudioOutInitIpmiGetSession(int32_t) { return 0; }

int PS4ABI sceAudioOutOpen(int32_t /*userId*/, int32_t /*type*/, int32_t /*index*/,
                           uint32_t length, uint32_t freq, uint32_t param) {
  uint32_t channels = 2;
  int isFloat = 0;
  decodeFormat(param, channels, isFloat);
  if (!freq) freq = 48000;
  int bridge = prosperity_audio_open(freq, channels, isFloat);
  std::lock_guard<std::mutex> lk(g_mtx);
  Port p;
  p.bridge = bridge;
  p.grain = length ? length : 256;
  p.channels = channels;
  p.open = true;
  g_ports.push_back(p);
  return static_cast<int>(g_ports.size());  // SCE handle = index + 1 (>0)
}

int PS4ABI sceAudioOutOutput(int32_t handle, const void *ptr) {
  std::lock_guard<std::mutex> lk(g_mtx);
  Port *p = port(handle);
  if (!p) return -1;
  if (!ptr) return 0;  // a null ptr is a "drain" request; nothing to queue
  if (p->bridge >= 0)
    prosperity_audio_output(p->bridge, ptr, p->grain);
  return static_cast<int>(p->grain);
}

// SceAudioOutOutputParam { int32_t handle; void *ptr; } (ptr is 8-aligned, so the
// struct is 16 bytes: handle@0, ptr@8).
struct OutputParam { int32_t handle; uint32_t pad; const void *ptr; };

int PS4ABI sceAudioOutOutputs(void *params, uint32_t num) {
  if (!params) return -1;
  const OutputParam *pp = static_cast<const OutputParam *>(params);
  int last = 0;
  for (uint32_t i = 0; i < num; i++)
    last = sceAudioOutOutput(pp[i].handle, pp[i].ptr);
  return last;
}

int PS4ABI sceAudioOutClose(int32_t handle) {
  std::lock_guard<std::mutex> lk(g_mtx);
  Port *p = port(handle);
  if (!p) return -1;
  if (p->bridge >= 0) prosperity_audio_close(p->bridge);
  p->open = false;
  p->bridge = -1;
  return 0;
}

int PS4ABI sceAudioOutSetVolume(int32_t handle, int32_t /*flag*/, int32_t *vol) {
  std::lock_guard<std::mutex> lk(g_mtx);
  Port *p = port(handle);
  if (!p) return -1;
  if (vol && p->bridge >= 0)  // SCE 0dB == 32768; use channel 0 as the master gain
    prosperity_audio_volume(p->bridge, static_cast<float>(vol[0]) / 32768.0f);
  return 0;
}

int PS4ABI sceAudioOutSetVolumeDc(int32_t, void *) { return 0; }

int PS4ABI sceAudioOutGetPortState(int32_t handle, void *state) {
  // SceAudioOutPortState is 32 bytes; zero it and report a connected port so the
  // caller doesn't read garbage (see the output-buffer convention).
  if (state) {
    std::memset(state, 0, 32);
    std::lock_guard<std::mutex> lk(g_mtx);
    Port *p = port(handle);
    if (p) {
      reinterpret_cast<uint16_t *>(state)[0] = 1;             // output: connected
      reinterpret_cast<uint8_t *>(state)[2] = (uint8_t)p->channels;
    }
  }
  return 0;
}

int64_t PS4ABI sceAudioOutGetLastOutputTime(int32_t) { return 0; }

}  // extern "C"
