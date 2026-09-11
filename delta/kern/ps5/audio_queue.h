#pragma once

#include <atomic>
#include <cstddef>

#include "base/arch.h"

namespace krnl::ps5 {

// AudioOut2 container layout in firmware 01.14.00. The producer publishes one
// 64-bit sequence number per grain, shared by every channel port. The daemon
// advances the consumer sequence only after taking a submitted grain.
// Returns its frame count, or zero for an empty/uninitialised/invalid queue.
// The caller owns mapping lifetime and supplies pacing and the event grant.
inline u32 ConsumeAudioOut2Block(u8* memory, size_t size) {
  if (!memory || size < 0x2508)
    return 0;
  const auto produced = std::atomic_ref<u64>(
      *reinterpret_cast<u64*>(memory + 0x1c80)).load(std::memory_order_acquire);
  auto consumer = std::atomic_ref<u64>(*reinterpret_cast<u64*>(memory + 0x2500));
  const u64 consumed = consumer.load(std::memory_order_relaxed);
  const u32 frames = *reinterpret_cast<const volatile u32*>(memory + 0x1c88);
  const u32 capacity = *reinterpret_cast<const volatile u32*>(memory + 0x1c8c);
  const u64 pending = produced - consumed;
  if (!pending || !capacity || capacity > 16 || pending > capacity ||
      !frames || frames > 2048 || (frames & 0x7f))
    return 0;
  consumer.store(consumed + 1, std::memory_order_release);
  return frames;
}

}  // namespace krnl::ps5
