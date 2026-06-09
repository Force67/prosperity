#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN compute-shader CPU interpreter. Executes a dispatched compute shader one
 * invocation at a time on the host, writing its image_store results straight into
 * guest memory. Doom64 builds its level texture atlases with compute shaders that
 * copy/decode a source buffer into the destination image (T#); without running
 * them the atlas the 3D world samples stays zero -> black. Running them here
 * populates the guest atlas so the existing draw path samples real texels.
 *
 * Scope: the straight-line "buffer -> image" copy/decode shaders Doom64 uses
 * (compute thread x/y, load source texels via MUBUF, image_store to the dest T#).
 * Bounds control flow is approximated by clamping/dropping out-of-range stores.
 * Gated behind DELTA_GPU_CSRUN until validated.
 */

#include <cstdint>

namespace gpu::gcn {

// Run a compute dispatch. csAddr = guest CS code; dim* = workgroup counts; tg* =
// threads per workgroup; userSgpr = number of user-data dwords loaded into s0..;
// tgidEnable = which workgroup-id dims land in the sgprs after the user data;
// userData = the 16 COMPUTE_USER_DATA_0 dwords. Returns the number of texels
// actually written (0 if it did nothing -> caller can log/ignore).
uint64_t runComputeShader(uint64_t csAddr, uint32_t dimX, uint32_t dimY,
                          uint32_t dimZ, uint32_t tgX, uint32_t tgY, uint32_t tgZ,
                          uint32_t userSgpr, uint32_t tgidEnable,
                          const uint32_t *userData);

}  // namespace gpu::gcn
