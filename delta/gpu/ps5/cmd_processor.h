#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 GPU command processor (scaffold). The PS5 uses an RDNA2 GPU driven by the
 * AGC command format, which is unrelated to the PS4's PM4/GCN path in gpu/ps4/.
 * This is greenfield: it will host an AGC packet parser plus an RDNA2->SPIR-V
 * recompiler, and must expose the same submit API as the PS4 path so the
 * runtime can dispatch on the guest platform. Currently unimplemented stubs.
 *
 * Runtime platform dispatch (ps4 vs ps5 submit) will be wired up when the PS5
 * GPU path is actually built; for now this only compiles as scaffolding.
 */

#include <cstdint>

namespace gpu::ps5 {

// Process one AGC draw command buffer (guest GPU address, size in bytes).
void SubmitDcb(const void* dcb, uint32_t size_bytes);

// Process one AGC constant command buffer.
void SubmitCcb(const void* ccb, uint32_t size_bytes);

// End the current frame and present the render target at `scanout_base`.
void EndFrame(uint64_t scanout_base);

}  // namespace gpu::ps5
