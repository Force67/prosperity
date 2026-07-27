/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

namespace gpu::vk {

struct ComputeBufferAccess {
  bool read = false;
  bool write = false;
};

constexpr bool NeedsComputeBarrier(ComputeBufferAccess prior,
                                   ComputeBufferAccess current) {
  return prior.write || (prior.read && current.write);
}

}  // namespace gpu::vk
