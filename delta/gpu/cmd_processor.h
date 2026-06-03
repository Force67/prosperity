#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GPU command processor: consumes a PM4 draw command buffer (dcb), tracks the
 * Liverpool register state, and drives the Vulkan renderer on each draw. Entry
 * point for the Gnm HLE submit path.
 */

#include <cstdint>

namespace gpu {

// Process one PM4 draw command buffer (guest GPU address, identity-mapped to a
// host pointer; size in bytes). Walks the packet stream, updating register
// state and issuing draws. Safe to call from the guest's GPU submit thread.
void submitDcb(const void *dcb, uint32_t sizeBytes);

}  // namespace gpu
