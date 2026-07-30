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
 * ---- the LLE shared-memory mixer protocol ----------------------------------
 * The real libSceAudioOut does NOT use an ioctl device; there is no /dev node to
 * write. It hands blocks to the system audio daemon through POSIX shm and is
 * woken by a named event flag. Established by disassembling the 11.00 module
 * (/system/common/lib/libSceAudioOut.sprx, a plain FreeBSD ELF -- the exported
 * NIDs decode straight to sceAudioOut* names) and confirmed against a live Isaac
 * run. There is NO ring and NO cursor: it is a one-block-deep handshake.
 *
 * Regions, all created O_RDWR|O_CREAT (0x202) by the module:
 *   "/shm_<pid>_C"        control block. ftruncate'd to 0xf28, but the module
 *                         WRITES up to 0x3c40 and relies on page rounding --
 *                         map at least 0x4000. Do not size it from ftruncate.
 *   "/shm_<pid>_<idx>_A"  one per open port, always 0x10000 bytes: the worst
 *                         case single block (max grain 2048 * 8ch * 4B). The
 *                         module memcpy's to OFFSET 0 every time.
 * <idx> is the audio port index, 0..25, decimal, same number in the shm name,
 * in the control block and in the event flag bit.
 *
 * Control block: 0x20 header, then 26 port slots of 0x250 (slot k is at
 * 0x20 + k*0x250). Header: +0x00 float -70.0, +0x08 u32 0xffffffff, rest zero.
 * Slot fields that matter to a consumer (offsets relative to the slot):
 *   +0x00 u64  HANDSHAKE TOKEN. The module writes the slot's own guest address
 *              here after filling the _A region, and refuses to fill it again
 *              while it is non-zero (it tests only the low u32).
 *   +0x08 u32  bytes per frame: 2/4/16 for S16 1/2/8ch, 4/8/32 for float, ...
 *   +0x0c..0x2b  8 floats, per-channel gain (SCE volume / 32768, so 1.0 = 0 dB)
 *   +0x2c u32  sample type: 0 = int16, 1 = float32, 2 = 32-bit
 *   +0x34 u32  sample rate; the module only accepts 48000
 *   +0x40 u64  sceKernelGetProcessTime of the last accepted block
 *   +0x50 u32  port index (== <idx>)
 *   +0x60 u32  grain, frames per block: multiple of 256, 256..2048
 *   +0x90 u32  state; 3 once a block has been submitted
 * channels = (+0x08) / (type 0 ? 2 : 4). Samples are the game's buffer verbatim,
 * interleaved, memcpy'd -- no conversion, no header, grain*bytesPerFrame bytes.
 *
 * sceAudioOutOutput(handle, ptr) is, in the module:
 *     submit();  if (!BUSY) return;          <- FIRST block needs no permission
 *     for (;;) { sceKernelWaitEventFlag("sceAudioOutMix<pid>", 1<<idx,
 *                                       AND|CLEARPAT, NULL, NULL);
 *                if (submit() != BUSY) break; }
 * where submit() returns BUSY (0x80260002) unless slot+0x00 is zero, and on
 * success does memcpy(portShm, ptr, grain*bpf) then slot+0x00 = &slot.
 *
 * So the daemon's whole job per port is: see slot+0x00 non-zero, take
 * grain*bpf bytes from the head of the _A region, write 0 to slot+0x00, then
 * sceKernelSetEventFlag(mixEvf, 1<<idx). Pace it at grain/48000 s per block and
 * the guest's Output blocks in the same way real hardware paces it.
 *
 * That daemon EXISTS: kern/ps4/audio_daemon.cpp, which consumes the regions and
 * feeds the same gfx_audio sink this shim uses. It starts only when the real
 * module creates the control block, so it can never double up with this file.
 * Under DELTA_LLE=libSceAudioOut Isaac now streams both ports in real time with
 * peaks in the same range as this shim's (see that file's header).
 *
 * Verified end to end against Isaac with the research harness in
 * kern/ps4/lv2/sys_mem.cpp (DELTA_SHM_AUDIO_PROBE, plus DELTA_AUDIOMIX_ACK in
 * sys_event_flag.cpp): performing exactly the above makes Isaac stream
 * continuously on port 7 (bpf=4, type=0, 2ch, 48000, grain 512) with a live
 * signal whose peak climbs like the HLE reference. Decoding the block as
 * interleaved int16 is confirmed statistically, not assumed: per-channel lag-1
 * autocorrelation 0.954 vs 0.523 at the wrong stride, and a float32 reading is
 * 44% denormal garbage.
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
  if (std::getenv("DELTA_AUDIO_TRACE"))
    std::fprintf(stderr, "[audioopen] len=%u freq=%u param=%#x -> %uch %s\n",
                 length, freq, param, channels, isFloat ? "f32" : "s16");
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
