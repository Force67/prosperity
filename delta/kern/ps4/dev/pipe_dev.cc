#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/pipe_dev.h"

namespace kern {
struct PipeState {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<u8> bytes;
  bool reader = true, writer = true;
};
namespace {
constexpr size_t kCapacity = 16384;
constexpr size_t kAtomicWrite = 512;
constexpr u16 kRead = 0x01, kWrite = 0x04, kHangup = 0x10;
}  // namespace

PipeDevice::PipeDevice(ObjectTable& objects,
                       std::shared_ptr<PipeState> state,
                       bool reader)
    : Device(objects), state_(std::move(state)), reader_(reader) {}

PipeDevice::~PipeDevice() {
  std::lock_guard lock(state_->mutex);
  (reader_ ? state_->reader : state_->writer) = false;
  state_->changed.notify_all();
}

i64 PipeDevice::Read(void* buffer, size_t length) {
  if (!reader_)
    return -SysError::eBADF;
  if (!length)
    return 0;
  if (!host_memory::IsMemoryRangeMapped(buffer, length))
    return -SysError::eFAULT;
  std::unique_lock lock(state_->mutex);
  while (state_->bytes.empty()) {
    if (!state_->writer)
      return 0;
    if (nonblocking_)
      return -SysError::eAGAIN;
    if (guest::Stopping())
      return -SysError::eINTR;
    state_->changed.wait_for(lock, std::chrono::milliseconds(5));
  }
  const size_t count = std::min(length, state_->bytes.size());
  for (size_t i = 0; i < count; ++i) {
    static_cast<u8*>(buffer)[i] = state_->bytes.front();
    state_->bytes.pop_front();
  }
  state_->changed.notify_all();
  return count;
}

i64 PipeDevice::Write(const void* buffer, size_t length) {
  if (reader_)
    return -SysError::eBADF;
  if (!length)
    return 0;
  if (!host_memory::IsMemoryRangeMapped(buffer, length))
    return -SysError::eFAULT;
  std::unique_lock lock(state_->mutex);
  const size_t needed = length <= kAtomicWrite ? length : 1;
  while (kCapacity - state_->bytes.size() < needed) {
    if (!state_->reader)
      return -SysError::ePIPE;
    if (nonblocking_)
      return -SysError::eAGAIN;
    if (guest::Stopping())
      return -SysError::eINTR;
    state_->changed.wait_for(lock, std::chrono::milliseconds(5));
  }
  if (!state_->reader)
    return -SysError::ePIPE;
  const size_t count = std::min(length, kCapacity - state_->bytes.size());
  const auto* bytes = static_cast<const u8*>(buffer);
  state_->bytes.insert(state_->bytes.end(), bytes, bytes + count);
  state_->changed.notify_all();
  return count;
}

u16 PipeDevice::Poll(u16 events) {
  std::lock_guard lock(state_->mutex);
  if (reader_)
    return ((events & kRead) && !state_->bytes.empty() ? kRead : 0) |
           (!state_->writer ? kHangup : 0);
  return ((events & kWrite) && state_->bytes.size() < kCapacity ? kWrite : 0) |
         (!state_->reader ? kHangup : 0);
}

u32 PipeDevice::GetFlags() const {
  std::lock_guard lock(state_->mutex);
  return (reader_ ? 0 : 1) | (nonblocking_ ? 4 : 0);
}

void PipeDevice::SetFlags(u32 flags) {
  std::lock_guard lock(state_->mutex);
  nonblocking_ = (flags & 4) != 0;
  state_->changed.notify_all();
}

int PipeDevice::Fstat(void* stat) {
  if (!host_memory::IsMemoryRangeMapped(stat, sizeof(SceKernelStat)))
    return -SysError::eFAULT;
  std::lock_guard lock(state_->mutex);
  FillStat(*static_cast<SceKernelStat*>(stat), 0x1000, state_->bytes.size());
  return 0;
}

void CreatePipe(ObjectTable& objects, u32& reader, u32& writer) {
  auto state = std::make_shared<PipeState>();
  reader = (new PipeDevice(objects, state, true))->handle();
  writer = (new PipeDevice(objects, std::move(state), false))->handle();
}
}  // namespace kern
