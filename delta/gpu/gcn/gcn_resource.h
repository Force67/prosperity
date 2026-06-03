#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking: extract the V# (buffer) / T# (image) / S# (sampler)
 * "sharps" a shader uses by analysing how it loads them out of the user-data
 * SGPRs. For Isaac's vertex-fetch VS this recovers the vertex-attribute buffers
 * so the renderer can pull real geometry.
 */

#include <cstdint>
#include <vector>

namespace gpu::gcn {

// A decoded vertex-buffer resource (GCN V#, 4 dwords).
struct VBuffer {
  uint64_t base = 0;       // guest address of the vertex data
  uint32_t stride = 0;     // bytes per vertex
  uint32_t numRecords = 0; // vertex count
  uint32_t dfmt = 0;       // data format (nfmt<<...|dfmt)
  uint32_t nfmt = 0;
};

// Decode a V# from 4 consecutive dwords.
VBuffer decodeVBuffer(const uint32_t *p);

// Given a fetch shader (guest code) and the VS user-data SGPRs (16 dwords),
// recover the vertex-attribute buffers it fetches. Returns the V#s in attribute
// order. Best-effort: handles the common Gnm fetch-shader pattern (s_load_dwordx4
// of a V# table pointed to by a user SGPR, then buffer_load_format per attribute).
std::vector<VBuffer> trackVertexBuffers(const uint32_t *fetchCode,
                                        uint32_t maxDwords,
                                        const uint32_t *vsUserData);

}  // namespace gpu::gcn
