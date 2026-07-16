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

// A decoded image resource (GCN T#, 8 dwords).
struct TImage {
  uint64_t base = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pitch = 0;      // surface pitch in pixels (T#.pitch+1)
  uint32_t layers = 1;     // physical array layers (T#.depth+1 for 2D arrays)
  uint32_t baseArray = 0;  // first layer exposed by this descriptor view
  uint32_t viewLayers = 1; // number of layers exposed by this descriptor view
  uint32_t mipLevels = 1;  // physical levels in storage (LAST_LEVEL + 1)
  uint32_t baseMip = 0;     // first level exposed by this descriptor view
  uint32_t viewMips = 1;    // levels exposed by this descriptor view
  uint32_t minLod = 0;      // T# MIN_LOD clamp in U4.8 fixed-point
  uint32_t dfmt = 0;
  uint32_t nfmt = 0;
  uint32_t type = 0;       // SQ_RSRC_IMG_* (9 = 2D, 13 = 2D array)
  uint32_t tilingIdx = 0;  // 8/31 = linear; everything else is tiled (1D micro or 2D macro)
  uint32_t sampler[4] = {}; // S# used by this MIMG sample instruction
  bool pow2Pad = false;     // pad physical mip dimensions/layers to powers of two
  bool samplerValid = false;
  bool arrayed = false;    // MIMG DA bit: address contains an array-layer coordinate
  bool valid = false;
};

// Decode a T# from 8 dwords.
TImage decodeTImage(const uint32_t *p);

// Recover the image(s) a (textured) pixel shader samples, by tracking its
// s_load_dwordx8 of T#s out of the PS user-data tables. Empty if the PS does no
// texture sampling. The result preserves MIMG binding order; unresolved entries
// are returned with valid=false so later bindings are not compacted.
std::vector<TImage> trackTextures(const uint32_t *psCode, uint32_t maxDwords,
                                  const uint32_t *psUserData);

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
