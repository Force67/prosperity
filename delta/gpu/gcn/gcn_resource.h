#pragma once

#include "base/arch.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/memory/shared_pointer.h"
#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

struct RecompiledCs;
struct ShaderAttr;
struct ShaderBuffer;

struct VBuffer {
  u64 base = 0;
  u32 stride = 0;
  u32 num_records = 0;
  u32 dfmt = 0;
  u32 nfmt = 0;
};

struct TImage {
  u64 base = 0;
  u32 width = 0;
  u32 height = 0;
  u32 pitch = 0;
  u32 depth = 1;
  u32 layers = 1;
  u32 base_array = 0;
  u32 view_layers = 1;
  u32 mip_levels = 1;
  u32 base_mip = 0;
  u32 view_mips = 1;
  u32 min_lod = 0;
  u32 dfmt = 0;
  u32 nfmt = 0;
  u32 type = 0;
  u32 tiling_idx = 0;
  u32 dst_sel[4] = {4, 5, 6, 7};
  u32 sampler[4] = {};
  bool pow2_pad = false;
  bool sampler_valid = false;
  bool is_3d = false;
  bool is_1d = false;
  bool arrayed = false;
  bool force_lod_zero = false;
  bool depth_compare = false;
  bool storage = false;
  bool null_descriptor = false;
  bool valid = false;
  u64 src = 0;
};

inline u32 PackDstSel(const u32 dst_sel[4]) {
  return (dst_sel[0] & 7) | ((dst_sel[1] & 7) << 3) | ((dst_sel[2] & 7) << 6) |
         ((dst_sel[3] & 7) << 9);
}

VBuffer DecodeVBuffer(const u32* dwords);

TImage DecodeTImage(const u32* dwords);

struct MimgBindingPlan {
  base::HashMap<u32, u32> binding_by_pc;
  base::Vector<u32> binding_srsrc;
  base::Vector<bool> binding_storage;
};
MimgBindingPlan PlanMimgBindings(const Program& program,
                                 const u8* reachable = nullptr);

class ScalarReplayScope {
 public:
  ScalarReplayScope();
  ~ScalarReplayScope();
  ScalarReplayScope(const ScalarReplayScope&) = delete;
  ScalarReplayScope& operator=(const ScalarReplayScope&) = delete;
};

void TrackTextures(base::Vector<TImage>& out,
                   const base::SharedPointer<const Program>& ps_program,
                   const u32* ps_user_data,
                   bool trace = false,
                   u64 code_base = 0);

inline u64 CbufKey(u32 base_sgpr, bool pointer, u32 version) {
  return (static_cast<u64>(version) << 32) | base_sgpr |
         (pointer ? 0x100u : 0u);
}

using CbufList = base::Vector<base::Pair<u64, VBuffer>>;
inline const VBuffer* FindCbuf(const CbufList& list, u64 key) {
  for (const auto& [k, v] : list)
    if (k == key)
      return &v;
  return nullptr;
}
void ResolveCbuffers(CbufList& out,
                     const base::SharedPointer<const Program>& program,
                     const u32* user_data,
                     u64 code_base = 0);

void ResolveDirectVertexBuffers(
    base::Vector<VBuffer>& out,
    const base::SharedPointer<const Program>& program,
    const base::Vector<ShaderAttr>& attrs,
    const u32* user_data,
    u64 code_base = 0);

void ResolveShaderBuffers(base::Vector<VBuffer>& out,
                          const base::SharedPointer<const Program>& program,
                          const base::Vector<ShaderBuffer>& buffers,
                          const u32* user_data,
                          u64 code_base = 0);

struct ResolvedCsResource {
  bool valid = false;
  u32 descriptor[8] = {};
};
void ResolveCsResources(base::Vector<ResolvedCsResource>& out,
                        const Program& program,
                        const RecompiledCs& plan,
                        const u32* user_data,
                        u64 code_base);

void TrackVertexBuffers(base::Vector<VBuffer>& out,
                        const Program& fetch_program,
                        const u32* vs_user_data);

extern void (*g_flush_guest_range)(u64 address, u64 bytes);
extern void (*g_overlay_guest_range)(u64 address, u32 dwords, u32* values);

extern u64 g_direct_table, g_direct_misses;

// A 2_10_10_10 vertex attribute in anything but UNORM: no portable host vertex
// format decodes it, so it is fetched as a raw R32_UINT and unpacked by the
// vertex shader. In the recompiler's int_attr_mask such an attribute sets bit
// 32 + location; the scaled number formats also set bit 48 + location, and the
// signed ones and the integer ones the usual sint (16 + location) and uint
// (location) bits.
inline bool PackedAttrFormat(u32 dfmt, u32 nfmt) {
  return dfmt == 9 && nfmt >= 1 && nfmt <= 6;
}
inline u64 PackedAttrBits(u32 location, u32 nfmt) {
  u64 bits = 1ull << (32 + location);
  if (nfmt == 2 || nfmt == 3)
    bits |= 1ull << (48 + location);
  if (nfmt == 3 || nfmt == 5)
    bits |= 1ull << (16 + location);
  if (nfmt == 4)
    bits |= 1ull << location;
  return bits;
}

// Vertex-shader register probe (DELTA_GPU_GCN_VSVGPR*): device address of a
// readback buffer of kVsProbeSlots slots, one per (instance, vertex). The
// low VsProbeVertexBits() of a slot index are the vertex.
// Slot words: draw tag 0, draw tag 1, kVsProbeMark | instance, the four probed
// registers, checksum (see ProbeVsCapture). The slot index gives the vertex.
extern u64 g_vs_probe;
constexpr u32 kVsProbeSlotWords = 8;
constexpr u32 kVsProbeSlots = 1u << 19;
constexpr u32 kVsProbeMark = 0xA5000000u;
extern bool g_bindless;

constexpr u32 kBindlessSlots[2] = {16384, 4096};
constexpr u32 kBindlessKeyWords[2] = {8, 4};
constexpr u32 kBindlessStride[2] = {kBindlessKeyWords[0] + 3,
                                    kBindlessKeyWords[1] + 3};
constexpr u32 kBindlessPending = 1u << 31;
constexpr u32 kBindlessProbes = 16;
constexpr u32 kBindlessKinds = 8;
constexpr u32 kBindlessSamplerBinding = kBindlessKinds;
constexpr u32 kBindlessTableBinding = kBindlessKinds + 1;

constexpr u64 kDirectBase = 0;
constexpr u64 kDirectSize = 0x1000000000000ull;
constexpr u64 kDirectFlatSize = 0x10000000000ull;
constexpr u32 kDirectBlockShift = 16;
constexpr u32 kDirectPageShift = 12;
constexpr u64 kDirectLeaf = 1;
constexpr u64 kDirectUnmapped = 2;
constexpr u64 kDirectDirectory = 4;
constexpr u32 kDirectDirectoryShift = 32;
constexpr u32 kDirectMissSlots = 63;
constexpr u64 kDirectBlocks = kDirectSize >> kDirectBlockShift;
constexpr u64 kDirectFlatBlocks = kDirectFlatSize >> kDirectBlockShift;
constexpr u64 kDirectDirectoryEntries = kDirectSize >> kDirectDirectoryShift;
constexpr u64 kDirectDirectoryBlocks =
    1ull << (kDirectDirectoryShift - kDirectBlockShift);
constexpr u64 kDirectTableEntries = kDirectFlatBlocks + kDirectDirectoryEntries;

constexpr u64 GuestDirectIndex(u64 address) {
  if (address - kDirectBase < kDirectSize)
    return (address - kDirectBase) >> kDirectBlockShift;
  return ~0ull;
}

}  // namespace gpu::gcn
