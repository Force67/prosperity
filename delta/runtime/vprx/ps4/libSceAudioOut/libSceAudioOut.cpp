/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceAudioOut. See libSceAudioOut.h. Bridges to the host SDL3 device via
 * delta_gfx's gfx_audio. Each open port records its grain/format so Output knows
 * how many interleaved frames the guest buffer holds.
 *
 * This path is verified working end to end: The Binding of Isaac opens two
 * ports and its samples reach SDL with a live signal (peak climbing 0.037 ->
 * 0.071 over a run). A title that comes out silent through here is submitting
 * silence -- SotC does, for the same upstream reason it submits a black frame.
 *
 * ---- what running this module LLE would take -------------------------------
 * The real libSceAudioOut does NOT use an ioctl device, so there is no /dev node
 * to write. Traced with DELTA_LLE=libSceAudioOut it instead:
 *   - opens a named event flag "sceAudioOutMix<pid>"  (pid is ours, 0x1337)
 *   - creates per-port POSIX shm: "/shm_<pid>_C", "/shm_<pid>_7_A",
 *     "/shm_<pid>_20_A" ... (flags 0x202 = O_RDWR|O_CREAT)
 * and then mixes into those regions and signals the flag, expecting the system
 * audio daemon to consume them. We host no such daemon, so the samples stop
 * there: with DELTA_LLE=libSceAudioOut the title runs normally (Isaac: 36 fps,
 * no crash) and our SDL sink is simply never opened.
 *
 * Hosting it means standing in for that daemon: map the shm regions, decode the
 * ring (header layout, read/write cursors, per-port sample format -- none of
 * which is established yet) and push the frames to prosperity_audio_output on
 * the flag. Reverse-engineer the ring against Isaac, NOT against SotC: Isaac is
 * the only title here that currently produces a non-zero signal, so it is the
 * only one where a wrong decode is distinguishable from a correct one.
 */

#include "libSceAudioOut.h"

#include "gfx/gfx_audio.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  // DELTA_AUDIO_TRACE: the raw param array next to how we parse it. The struct
  // stride is the whole ballgame -- misread it and every handle/ptr past the
  // first is garbage, which reads downstream as "one port, silent".
  static const bool trace = std::getenv("DELTA_AUDIO_TRACE") != nullptr;
  static int dumped = 0;
  if (trace && dumped < 4) {
    dumped++;
    const auto *b = static_cast<const uint8_t *>(params);
    std::fprintf(stderr, "[audioparam] num=%u raw:", num);
    for (uint32_t i = 0; i < num * 16 && i < 96; i++)
      std::fprintf(stderr, "%s%02x", (i % 16) ? "" : " ", b[i]);
    std::fprintf(stderr, "\n");
    const OutputParam *q = static_cast<const OutputParam *>(params);
    for (uint32_t i = 0; i < num && i < 6; i++)
      std::fprintf(stderr, "[audioparam]  [%u] handle=%d ptr=%p\n", i,
                   q[i].handle, q[i].ptr);
  }
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
