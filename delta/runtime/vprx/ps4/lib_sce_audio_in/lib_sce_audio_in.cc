/*
 * HLE libSceAudioIn.
 */

#include "runtime/vprx/ps4/lib_sce_audio_in/lib_sce_audio_in.h"
#include "base/arch.h"
#include "guest_abi.h"

#include <cstring>
#include "base/containers/vector.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"

namespace {

struct Port {
  u32 grain = 0;
  u32 channels = 1;
  u32 freq = 16000;
  bool open = false;
};

base::Mutex g_mtx;
base::Vector<Port> g_ports;

u32 ChannelsFromParam(u32 param) {
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

u32 BytesPerSample(u32 param) {
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

Port* FindPort(i32 handle) {
  if (handle <= 0 || handle > static_cast<i32>(g_ports.size()))
    return nullptr;
  Port& p = g_ports[handle - 1];
  return p.open ? &p : nullptr;
}

int OpenPort(u32 length, u32 freq, u32 param) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  Port p;
  p.grain = length ? length : 256;
  p.channels = ChannelsFromParam(param);
  p.freq = freq ? freq : 16000;
  p.open = true;
  g_ports.push_back(p);
  return static_cast<int>(g_ports.size());
}

int ReadSilence(i32 handle, void* ptr, u32 sample_bytes = 2) {
  u32 grain, channels, freq;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    Port* p = FindPort(handle);
    if (!p)
      return -1;
    grain = p->grain;
    channels = p->channels;
    freq = p->freq;
  }
  const u32 bytes = grain * channels * sample_bytes;
  if (ptr)
    std::memset(ptr, 0, bytes);
  // The real sceAudioInInput blocks until `grain` samples are captured. Pace it
  // (outside the lock) to grain/freq seconds so the title's capture thread
  // doesn't busy-spin at 100% CPU returning instant silence.
  if (freq)
    base::SleepForMicroseconds(1000000ull * grain / freq);
  return static_cast<int>(grain);
}

void FillStatus(i32 handle, void* status) {
  if (!status)
    return;
  std::memset(status, 0, 32);
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (Port* p = FindPort(handle)) {
    reinterpret_cast<u32*>(status)[0] = 1;
    reinterpret_cast<u32*>(status)[1] = p->grain;
    reinterpret_cast<u32*>(status)[2] = p->channels;
  }
}

}  // namespace

extern "C" {

int PS4ABI sceAudioInInit() {
  return 0;
}

int PS4ABI sceAudioInOpen(i32, i32, i32, u32 length, u32 freq, u32 param) {
  return OpenPort(length, freq, param);
}

int PS4ABI sceAudioInInput(i32 handle, void* ptr) {
  return ReadSilence(handle, ptr);
}

int PS4ABI sceAudioInClose(i32 handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  Port* p = FindPort(handle);
  if (!p)
    return -1;
  p->open = false;
  return 0;
}

int PS4ABI sceAudioInGetStatus(i32 handle, void* status) {
  FillStatus(handle, status);
  return 0;
}

int PS4ABI sceAudioInSetConnections(i32, i32) {
  return 0;
}

int PS4ABI sceAudioInGetHandleStatus(i32 handle, void* status) {
  FillStatus(handle, status);
  return 0;
}

int PS4ABI sceAudioInDeviceOpen(i32 user_id,
                                i32 type,
                                i32 index,
                                u32 length,
                                u32 freq,
                                u32 param) {
  return sceAudioInOpen(user_id, type, index, length, freq, param);
}

int PS4ABI sceAudioInDeviceHqOpen(i32 user_id,
                                  i32 type,
                                  i32 index,
                                  u32 length,
                                  u32 freq,
                                  u32 param) {
  return sceAudioInOpen(user_id, type, index, length, freq, param);
}

int PS4ABI sceAudioInDeviceRead(i32 handle, void* ptr) {
  return ReadSilence(handle, ptr);
}

int PS4ABI sceAudioInDeviceClose(i32 handle) {
  return sceAudioInClose(handle);
}

int PS4ABI sceAudioInDeviceState(i32 handle, void* state) {
  FillStatus(handle, state);
  return 0;
}

}  // extern "C"
