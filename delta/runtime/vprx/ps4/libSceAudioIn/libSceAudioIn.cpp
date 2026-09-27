/*
 * HLE libSceAudioIn.
 */

#include "libSceAudioIn.h"
#include "base/arch.h"

#include <cstring>
#include <base/containers/vector.h>
#include <base/threading/lock_guard.h>
#include <base/threading/mutex.h>
#include <base/threading/thread.h>

namespace {

struct Port {
  u32 grain = 0;
  u32 channels = 1;
  u32 freq = 16000;
  bool open = false;
};

base::Mutex g_mtx;
base::Vector<Port> g_ports;

u32 channelsFromParam(u32 param) {
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

u32 bytesPerSample(u32 param) {
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

Port *port(i32 handle) {
  if (handle <= 0 || handle > static_cast<i32>(g_ports.size()))
    return nullptr;
  Port &p = g_ports[handle - 1];
  return p.open ? &p : nullptr;
}

int openPort(u32 length, u32 freq, u32 param) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  Port p;
  p.grain = length ? length : 256;
  p.channels = channelsFromParam(param);
  p.freq = freq ? freq : 16000;
  p.open = true;
  g_ports.push_back(p);
  return static_cast<int>(g_ports.size());
}

int readSilence(i32 handle, void *ptr, u32 sampleBytes = 2) {
  u32 grain, channels, freq;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    Port *p = port(handle);
    if (!p)
      return -1;
    grain = p->grain;
    channels = p->channels;
    freq = p->freq;
  }
  const u32 bytes = grain * channels * sampleBytes;
  if (ptr)
    std::memset(ptr, 0, bytes);
  // The real sceAudioInInput blocks until `grain` samples are captured. Pace it
  // (outside the lock) to grain/freq seconds so the title's capture thread
  // doesn't busy-spin at 100% CPU returning instant silence.
  if (freq)
    base::SleepForMicroseconds(1000000ull * grain / freq);
  return static_cast<int>(grain);
}

void fillStatus(i32 handle, void *status) {
  if (!status)
    return;
  std::memset(status, 0, 32);
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (Port *p = port(handle)) {
    reinterpret_cast<u32 *>(status)[0] = 1;
    reinterpret_cast<u32 *>(status)[1] = p->grain;
    reinterpret_cast<u32 *>(status)[2] = p->channels;
  }
}

}  // namespace

extern "C" {

int PS4ABI sceAudioInInit() { return 0; }

int PS4ABI sceAudioInOpen(i32, i32, i32, u32 length,
                          u32 freq, u32 param) {
  return openPort(length, freq, param);
}

int PS4ABI sceAudioInInput(i32 handle, void *ptr) {
  return readSilence(handle, ptr);
}

int PS4ABI sceAudioInClose(i32 handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  Port *p = port(handle);
  if (!p)
    return -1;
  p->open = false;
  return 0;
}

int PS4ABI sceAudioInGetStatus(i32 handle, void *status) {
  fillStatus(handle, status);
  return 0;
}

int PS4ABI sceAudioInSetConnections(i32, i32) { return 0; }

int PS4ABI sceAudioInGetHandleStatus(i32 handle, void *status) {
  fillStatus(handle, status);
  return 0;
}

int PS4ABI sceAudioInDeviceOpen(i32 userId, i32 type, i32 index,
                                u32 length, u32 freq,
                                u32 param) {
  return sceAudioInOpen(userId, type, index, length, freq, param);
}

int PS4ABI sceAudioInDeviceHqOpen(i32 userId, i32 type, i32 index,
                                  u32 length, u32 freq,
                                  u32 param) {
  return sceAudioInOpen(userId, type, index, length, freq, param);
}

int PS4ABI sceAudioInDeviceRead(i32 handle, void *ptr) {
  return readSilence(handle, ptr);
}

int PS4ABI sceAudioInDeviceClose(i32 handle) {
  return sceAudioInClose(handle);
}

int PS4ABI sceAudioInDeviceState(i32 handle, void *state) {
  fillStatus(handle, state);
  return 0;
}

}  // extern "C"
