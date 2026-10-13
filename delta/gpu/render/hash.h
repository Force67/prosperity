/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

namespace gpu::render {

inline u64 HashWord(u64 h, u64 v) {
  return (h ^ v) * 1099511628211ull;
}

}  // namespace gpu::render
