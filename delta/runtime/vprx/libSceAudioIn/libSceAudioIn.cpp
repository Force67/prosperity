/*
 * HLE libSceAudioIn.
 */

#include "libSceAudioIn.h"

#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Port {
  uint32_t grain = 0;
  uint32_t channels = 1;
  uint32_t freq = 16000;
  bool open = false;
};

std::mutex g_mtx;
std::vector<Port> g_ports;

uint32_t channelsFromParam(uint32_t param) {
  switch (param & 0xFF) {
  case 1:
  case 4:
    return 2;
  case 2:
  case 5:
  case 6:
  case 7:
    return 8;
  default:
    return 1;
  }
}

uint32_t bytesPerSample(uint32_t param) {
  switch (param & 0xFF) {
  case 3:
  case 4:
  case 5:
  case 7:
    return 4;
  default:
    return 2;
  }
}

Port *port(int32_t handle) {
  if (handle <= 0 || handle > static_cast<int32_t>(g_ports.size()))
    return nullptr;
  Port &p = g_ports[handle - 1];
  return p.open ? &p : nullptr;
}

int openPort(uint32_t length, uint32_t freq, uint32_t param) {
  std::lock_guard<std::mutex> lk(g_mtx);
  Port p;
  p.grain = length ? length : 256;
  p.channels = channelsFromParam(param);
  p.freq = freq ? freq : 16000;
  p.open = true;
  g_ports.push_back(p);
  return static_cast<int>(g_ports.size());
}

int readSilence(int32_t handle, void *ptr, uint32_t sampleBytes = 2) {
  uint32_t grain, channels, freq;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    Port *p = port(handle);
    if (!p)
      return -1;
    grain = p->grain;
    channels = p->channels;
    freq = p->freq;
  }
  const uint32_t bytes = grain * channels * sampleBytes;
  if (ptr)
    std::memset(ptr, 0, bytes);
  // The real sceAudioInInput blocks until `grain` samples are captured. Pace it
  // (outside the lock) to grain/freq seconds so the title's capture thread
  // doesn't busy-spin at 100% CPU returning instant silence.
  if (freq)
    std::this_thread::sleep_for(
        std::chrono::microseconds(1000000ull * grain / freq));
  return static_cast<int>(grain);
}

void fillStatus(int32_t handle, void *status) {
  if (!status)
    return;
  std::memset(status, 0, 32);
  std::lock_guard<std::mutex> lk(g_mtx);
  if (Port *p = port(handle)) {
    reinterpret_cast<uint32_t *>(status)[0] = 1;
    reinterpret_cast<uint32_t *>(status)[1] = p->grain;
    reinterpret_cast<uint32_t *>(status)[2] = p->channels;
  }
}

}  // namespace

extern "C" {

int PS4ABI sceAudioInInit() { return 0; }

int PS4ABI sceAudioInOpen(int32_t, int32_t, int32_t, uint32_t length,
                          uint32_t freq, uint32_t param) {
  return openPort(length, freq, param);
}

int PS4ABI sceAudioInInput(int32_t handle, void *ptr) {
  return readSilence(handle, ptr);
}

int PS4ABI sceAudioInClose(int32_t handle) {
  std::lock_guard<std::mutex> lk(g_mtx);
  Port *p = port(handle);
  if (!p)
    return -1;
  p->open = false;
  return 0;
}

int PS4ABI sceAudioInGetStatus(int32_t handle, void *status) {
  fillStatus(handle, status);
  return 0;
}

int PS4ABI sceAudioInSetConnections(int32_t, int32_t) { return 0; }

int PS4ABI sceAudioInGetHandleStatus(int32_t handle, void *status) {
  fillStatus(handle, status);
  return 0;
}

int PS4ABI sceAudioInDeviceOpen(int32_t userId, int32_t type, int32_t index,
                                uint32_t length, uint32_t freq,
                                uint32_t param) {
  return sceAudioInOpen(userId, type, index, length, freq, param);
}

int PS4ABI sceAudioInDeviceHqOpen(int32_t userId, int32_t type, int32_t index,
                                  uint32_t length, uint32_t freq,
                                  uint32_t param) {
  return sceAudioInOpen(userId, type, index, length, freq, param);
}

int PS4ABI sceAudioInDeviceRead(int32_t handle, void *ptr) {
  return readSilence(handle, ptr);
}

int PS4ABI sceAudioInDeviceClose(int32_t handle) {
  return sceAudioInClose(handle);
}

int PS4ABI sceAudioInDeviceState(int32_t handle, void *state) {
  fillStatus(handle, state);
  return 0;
}

}  // extern "C"
