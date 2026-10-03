/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include <cstring>

namespace gpu::render {

struct ComputeWritebackCounts {
  u64 wrote = 0, adopted = 0, conflicts = 0;
};

inline void MergeComputeWritebackBlock(u8* guest, u8* staged, u8* shadow,
                                      ComputeWritebackCounts& counts) {
  const bool gpu = std::memcmp(staged, shadow, 64) != 0;
  const bool cpu = std::memcmp(guest, shadow, 64) != 0;
  if (gpu) {
    // CPU writes can race the block comparison. Only publish changed words.
    counts.conflicts += cpu;
    for (u64 w = 0; w < 64; w += 4) {
      if (std::memcmp(guest + w, shadow + w, 4) != 0) {
        std::memcpy(staged + w, guest + w, 4);
        counts.adopted += 4;
      } else if (std::memcmp(staged + w, shadow + w, 4) != 0) {
        std::memcpy(guest + w, staged + w, 4);
        counts.wrote += 4;
      }
      std::memcpy(shadow + w, staged + w, 4);
    }
  } else if (cpu) {
    std::memcpy(staged, guest, 64);
    std::memcpy(shadow, guest, 64);
    counts.adopted += 64;
  }
}

}  // namespace gpu::render
