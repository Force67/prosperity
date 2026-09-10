/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 -> GLCompute SPIR-V. See rdna_compute.h.
 */

#include "gpu/ps5/rdna/rdna_compute.h"
#include "base/arch.h"

#include "gpu/guest_memory.h"

namespace gpu::rdna {
u32 ComputeCodeDwords(const u32* code) {
  if (!code)
    return 0;
  const u64 base = reinterpret_cast<u64>(code);
  // Probe only mapped memory, then let the decoder stop at the actual endpgm.
  u32 lo = 0, hi = 64 * 1024 / 4;
  if (gpu::IsReadableRange(base, u64(hi) * 4))
    return hi;
  while (lo < hi) {
    const u32 mid = (lo + hi + 1) / 2;
    if (gpu::IsReadableRange(base, u64(mid) * 4))
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}
}

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::rdna {

gpu::gcn::RecompiledCs
RecompileCompute(const u32*, u32, u32, u32, u32, u32, u32, bool) {
  return {};  // no SPIR-V backend: the caller skips the dispatch
}

}  // namespace gpu::rdna
#else

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/gcn/spirv/translator.h"
#include "gpu/guest_memory.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_emit.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include <base/logging.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kGpuSpirvNoopt, "DELTA_GPU_SPIRV_NOOPT", false);
DELTA_OPTION(bool, kGpuShtrace, "DELTA_GPU_SHTRACE", false);
DELTA_OPTION(bool, kAgcTrace, "DELTA_AGC_TRACE", false);
}  // namespace

namespace gpu::rdna {
namespace {

using gpu::gcn::CsResource;
using gpu::gcn::Enc;
using gpu::gcn::Id;
using gpu::gcn::Inst;
using gpu::gcn::kMaxCsResources;
using gpu::gcn::Program;
using gpu::gcn::RecompiledCs;
using gpu::gcn::StageContext;
using gpu::gcn::Translator;

// The decoded compute program, instruction by instruction, under the same knob
// the vertex and pixel stages use. A declined dispatch says which register it
// could not follow; this is what says what wrote it.
void DumpCsProgram(const Program& program) {
  if (!kGpuShtrace && !kAgcTrace)
    return;
  BASE_LOGI("gcnspv", "=== cs decode: {} insts ===", program.size());
  for (const Inst& in : program) {
    base::String line;
    base::FormatTo(line, "  pc={:04x} len={} enc={} op={:#05x}  {:08x}", in.pc,
                   in.size, static_cast<u32>(in.enc), in.opcode, in.raw[0]);
    if (in.size >= 2)
      base::FormatTo(line, " {:08x}", in.raw[1]);
    if (in.has_literal)
      base::FormatTo(line, " lit={:08x}", in.literal);
    BASE_LOGI("gcnspv", "{}", line.c_str());
  }
}

// Why a descriptor the replay could not follow was declined, as its own tag:
// the three have very different fixes, and one line per shader is all the
// visibility a skipped pass gets.
const char* LossTag(ScalarReplayPlan::Loss loss) {
  switch (loss) {
    case ScalarReplayPlan::kUnmodelled:
      return "cs.descriptor-unmodelled-write.rdna";
    case ScalarReplayPlan::kConditional:
      return "cs.descriptor-conditional-write.rdna";
    case ScalarReplayPlan::kLoopBody:
      return "cs.descriptor-loop-body-write.rdna";
    default:
      return "cs.descriptor-loop-carried.rdna";
  }
}

// MIMG forms the compute path serves: the shared emitter's own set, plus the
// ones EmitCsMemory lowers onto it (see EmitLoweredMimg).
bool CsMimgSupported(u32 op) {
  switch (op) {
    case 0x00:  // image_load
    case 0x01:  // image_load_mip
    case 0x08:  // image_store
    case 0x09:  // image_store_mip
    case 0x0f:  // image_atomic_swap .. image_atomic_xor
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1a:
    case 0x24:  // image_sample_l
    case 0x27:  // image_sample_lz
    case 0x2c:  // image_sample_c_l
    case 0x2f:  // image_sample_c_lz
    case 0x34:  // image_sample_l_o
    case 0x37:  // image_sample_lz_o
    case 0x44:  // image_gather4_l
    case 0x47:  // image_gather4_lz
      return true;
    default:
      return false;
  }
}

// One set-0 storage buffer per distinct descriptor, mirroring the GFX7 planner
// (gpu/gcn PlanCsResources) with RDNA2's SMEM decode.
//
// A descriptor either sits INLINE in the COMPUTE_USER_DATA window, where the
// dispatch path reads it straight out of base_sgpr, or an SRT chain produced it
// (the shader s_loads a table pointer out of user data and the real V# out of
// that table). The dispatch path recovers the second kind by replaying the
// shader's scalar ops (rdna::ResolveBuffers), keyed on use_pc. What the replay
// cannot follow faithfully is declined, not guessed: a compute dispatch writes
// guest memory.
bool PlanResources(const Program& program,
                   u32 lds_dwords,
                   u32 user_sgpr,
                   RecompiledCs& r,
                   std::unordered_map<u32, u32>& bind) {
  const ScalarReplayPlan replay = PlanScalarReplay(program);
  bool uses_gds = false;
  std::unordered_set<u32> image_candidates, staged_images;
  // COMPUTE_USER_DATA is 16 dwords wide; the dispatch path reads no further.
  const u32 ud_dwords = std::min(user_sgpr, 16u);
  bool scalar_written[136] = {};
  u32 version[136] = {};  // instruction index of the last write

  // Uses of the same registers with no write in between are the same
  // descriptor and share a binding. A reloaded quad gets its own, or the
  // second buffer would silently inherit the first one's range.
  struct BindingKey {
    u32 base_sgpr;
    u32 kind;
    u32 versions[8];
  };
  std::vector<BindingKey> keys;
  u32 index = 0;  // instruction index of the use being planned
  const auto resource = [&](u32 pc, u32 base_sgpr, u32 dwords,
                            u8 kind, bool written, u32 min_bytes) {
    if (base_sgpr + dwords > 136) {
      gpu::gcn::WarnUnsupported("cs.descriptor-range.rdna", base_sgpr);
      return false;
    }
    bool untouched = true;
    for (u32 k = 0; k < dwords; k++)
      untouched = untouched && !scalar_written[base_sgpr + k];
    // A descriptor the shader never wrote and that fits the COMPUTE_USER_DATA
    // window is that window's, verbatim. Anything else -- an SRT chain, or a
    // load over the shader's own user data -- reaches the dispatch through the
    // replay, which prefers its own result over the window for exactly this
    // reason, so it is only as good as the replay is.
    const bool inline_user_data = base_sgpr + dwords <= ud_dwords && untouched;
    bool runtime_address = false;
    if (!inline_user_data) {
      const ScalarReplayPlan::Loss loss =
          replay.LossAt(base_sgpr, dwords, index);
      if (loss != ScalarReplayPlan::kCovered) {
        // The replay would hand the dispatch a descriptor the wave never had.
        // A read only stages a copy of guest memory and can at worst come out
        // as wrong pixels; a write would land in the wrong place.
        if (written && kind == 1) {
          gpu::gcn::WarnUnsupported(LossTag(loss), base_sgpr);
          return false;
        }
        if (kind != 1)
          runtime_address = true;
        else
          gpu::gcn::NoteApproximated("cs.descriptor-unproven-read.rdna",
                                     base_sgpr);
      }
    }
    BindingKey key{.base_sgpr = base_sgpr, .kind = kind};
    for (u32 k = 0; k < dwords && k < 8; k++)
      key.versions[k] = version[base_sgpr + k];
    for (u32 i = 0; i < keys.size(); i++) {
      if (std::memcmp(&keys[i], &key, sizeof(key)) != 0)
        continue;
      CsResource& res = r.resources[i];
      res.written = res.written || written;
      res.runtime_address = res.runtime_address || runtime_address;
      res.min_bytes = std::max(res.min_bytes, min_bytes);
      bind[pc] = i;
      return true;
    }
    const u32 idx = static_cast<u32>(r.resources.size());
    if (idx >= kMaxCsResources) {
      if (kGpuShtrace) {
        for (const CsResource& res : r.resources)
          BASE_LOGI("rdnacs", "resource {} kind={} sgpr={} pc={:#x}",
                    res.binding, res.kind, res.base_sgpr, res.use_pc);
        BASE_LOGI("rdnacs", "overflow kind={} sgpr={} pc={:#x}", kind,
                  base_sgpr, pc);
      }
      gpu::gcn::WarnUnsupported("cs.resource-count", idx + 1);
      return false;
    }
    keys.push_back(key);
    bind[pc] = idx;
    r.resources.push_back({base_sgpr, pc, idx, kind, written, /*read=*/true,
                           min_bytes, runtime_address});
    return true;
  };

  for (const Inst& inst : program) {
    const u32 w = inst.raw[0], w1 = inst.raw[1];
    switch (inst.enc) {
      case Enc::kSmrd: {
        const Smem smem = DecodeSmem(inst);
        const u32 n = SmemLoadCount(smem.op);
        // A negative immediate reads below the descriptor's base, which the
        // staged range cannot cover.
        if (!n || smem.offset < 0) {
          gpu::gcn::WarnUnsupported("smem.cs.rdna", smem.op, w, w1);
          return false;
        }
        const bool buffer = smem.op >= 0x08;  // s_buffer_load_* takes a V#
        // The immediate is a BYTE offset on RDNA2 (a dword index on GFX7).
        if (!resource(inst.pc, smem.sbase, buffer ? 4 : 2, buffer ? 0 : 2,
                      false, static_cast<u32>(smem.offset) + n * 4))
          return false;
        break;
      }
      case Enc::kMubuf: {
        const u32 op = inst.opcode;
        const bool load = op <= 0x03 || (op >= 0x08 && op <= 0x0f);
        const bool store = (op >= 0x04 && op <= 0x07) || op == 0x18 ||
                           op == 0x1a || (op >= 0x1c && op <= 0x1f);
        // The atomics read-modify-write the same storage buffer a store lands
        // in; the emitter names the handful with no single SPIR-V op itself.
        const bool atomic =
            (op >= 0x30 && op <= 0x3f) || (op >= 0x50 && op <= 0x5f);
        // Anything else is a d16 form, and the shared emitter masks the opcode
        // to 7 bits, so the 0x80+ format variants would alias a plain load.
        // LDS and TFE add a destination it has no model for.
        if ((!load && !store && !atomic) || ((w >> 16) & 1) ||
            ((w1 >> 23) & 1)) {
          gpu::gcn::WarnUnsupported("mubuf.cs.rdna", op, w, w1);
          return false;
        }
        if (!resource(inst.pc, ((w1 >> 16) & 0x1F) * 4, 4, 0, store || atomic,
                      0))
          return false;
        break;
      }
      case Enc::kMtbuf: {
        const u32 op = inst.opcode;
        if (op > 0x07 || ((w1 >> 23) & 1)) {  // op[3] selects the d16 forms
          gpu::gcn::WarnUnsupported("mtbuf.cs.rdna", op, w, w1);
          return false;
        }
        if (!resource(inst.pc, ((w1 >> 16) & 0x1F) * 4, 4, 0, op >= 4, 0))
          return false;
        break;
      }
      case Enc::kDs: {
        // ds_swizzle is a cross-lane move rather than an LDS access, so it is
        // the one DS op that runs without an LDS allocation. The GDS bit picks
        // the global scratchpad instead of LDS; the counter pair that lives
        // there (append / consume) we do model, the rest we do not.
        const bool gds = ((w >> 17) & 1) != 0;
        const bool counter = inst.opcode == 0x3d || inst.opcode == 0x3e;
        if (gds && counter) {
          uses_gds = true;
          break;
        }
        if (gds || (!lds_dwords && inst.opcode != 0x35)) {
          gpu::gcn::WarnUnsupported("ds.cs.rdna", inst.opcode, w, w1);
          return false;
        }
        break;
      }
      case Enc::kMimg: {
        // The shared emitter reads the T# with the gfx10.3 field positions when
        // the translator is in RDNA mode, so the plan is the same as GCN's.
        const u32 op = inst.opcode;
        const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
        if (op == 0xe6 || op == 0xe7) {
          const u32 nsa = (w >> 1) & 3;
          const u32 addresses = ((w1 >> 30) & 1 ? 8u : 11u) + (op == 0xe7);
          // DIM=0, DMASK=15, UNRM=R128=1, TFE=LWE=D16=SSAMP=0.
          if ((w & 0x39f38u) != 0x9f00u || (w1 & 0xbfe00000u) ||
              (nsa && 1 + nsa * 4 < addresses) ||
              (!nsa && (w1 & 255) + addresses > 256) ||
              ((w1 >> 8) & 255) > 252) {
            gpu::gcn::WarnUnsupported("bvh.control.rdna", op, w, w1);
            return false;
          }
          if (!resource(inst.pc, srsrc, 4, 3, false, 0))
            return false;
          break;
        }
        if (op == 0x0e)
          break;  // get_resinfo reads only descriptor SGPRs
        const bool store =
            op == 0x08 || op == 0x09 || (op >= 0x0f && op <= 0x1a);
        if (!CsMimgSupported(op) || ((w >> 15) & 1) || srsrc + 7 >= 136) {
          gpu::gcn::WarnUnsupported("mimg.cs.rdna", inst.opcode, w, w1);
          return false;
        }
        if (!resource(inst.pc, srsrc, 8, 1, store, 0))
          return false;
        (op == 0 || op == 8 ? image_candidates : staged_images).insert(bind[inst.pc]);
        break;
      }
      case Enc::kFlat: {
        // GLOBAL uses a live 64-bit SGPR base plus an unsigned VGPR byte
        // offset, or a full VGPR pair when SADDR=NULL. The signed immediate
        // is added in 64 bits; a CPU-guessed SSBO window cannot model this.
        const u32 seg = (w >> 14) & 3;
        const u32 op = inst.opcode;
        const u32 saddr = (w1 >> 16) & 0x7f;
        const bool load = op >= 0x08 && op <= 0x0f;
        const bool store = op == 0x18 || op == 0x1a || (op >= 0x1c && op <= 0x1f);
        if (seg != 2 || (saddr >= 104 && saddr != 125) || (!load && !store)) {
          gpu::gcn::WarnUnsupported("flat.cs.rdna", op, w, w1);
          return false;
        }
        if (!resource(inst.pc, saddr == 125 ? 0 : saddr, 2, 2, store, 0))
          return false;
        r.resources[bind[inst.pc]].runtime_address = true;
        break;
      }
      default:
        break;
    }
    // Bookkeeping runs after the use: an s_load may take its pointer from the
    // same SGPRs it lands the descriptor in.
    for (const ScalarWrites::Range& range : PossibleScalarWrites(inst).range)
      for (u32 k = 0; k < range.count && range.first + k < 136; k++) {
        scalar_written[range.first + k] = true;
        version[range.first + k] = index + 1;
      }
    index++;
  }
  if (uses_gds)
    r.gds_binding = static_cast<int>(r.resources.size());
  // Reuse the guest-address capability only in shaders that already require
  // it. Ordinary image shaders retain their portable staged implementation.
  const bool has_runtime = std::any_of(r.resources.begin(), r.resources.end(),
      [](const CsResource& res) { return res.runtime_address; });
  if (has_runtime)
    for (u32 binding : image_candidates)
      r.resources[binding].runtime_image = !staged_images.count(binding);
  if (std::any_of(r.resources.begin(), r.resources.end(),
                  [](const CsResource& res) { return res.runtime_address || res.runtime_image; })) {
    // Traversal follows pointers through scalar tables before reaching a BVH.
    // All read-only raw resources in this shader use their live GPU address.
    const bool writes = std::any_of(r.resources.begin(), r.resources.end(),
        [](const CsResource& res) { return res.runtime_address && res.written; });
    for (CsResource& res : r.resources)
      if (res.kind != 1 && (!res.written || writes))
        res.runtime_address = true;
    for (const CsResource& res : r.resources)
      r.guest_memory_written |= (res.runtime_address || res.runtime_image) && res.written;
    r.guest_memory_binding = r.resources.size() + (uses_gds ? 1 : 0);
  }
  return true;
}

// s_load / s_buffer_load: scalar reads of guest memory, served from the storage
// buffer planned for this instruction.
void EmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const Smem smem = DecodeSmem(inst);
  const u32 n = SmemLoadCount(smem.op);
  const int b = gpu::gcn::CsBindingFor(sc, inst.pc);
  if (!n || b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  // Both offsets are in BYTES on RDNA2 (GFX7's immediate was a dword index);
  // soffset field 125 is NULL, meaning no register offset at all.
  const Id immediate = t.U32(static_cast<u32>(smem.offset));
  const Id byte_off = smem.soffset == 125
                          ? immediate
                          : t.Add(t.SrcRaw(smem.soffset, 0), immediate);
  const Id dword0 = t.Shr(byte_off, t.U32(2));
  // The destination may overwrite the pointer/descriptor pair itself.
  // Read every component using the original SGPRs before updating any of them.
  std::vector<Id> values;
  for (u32 k = 0; k < n; k++)
    values.push_back(gpu::gcn::CsSsboLoad(t, sc, static_cast<u32>(b),
                                         t.Add(dword0, t.U32(k))));
  for (u32 k = 0; k < n; k++)
    t.SetSdst(smem.sdst, k, values[k]);
}

// MIMG address components in VGPR order: NSA names each in its own register,
// the sequential form numbers them up from VADDR. Slots past the ones an
// instruction actually uses still have to hold a valid id, because the shared
// emitter indexes a fixed few of them whatever the image type turns out to be.
std::array<Id, 8> MimgAddress(Translator& t, const Inst& inst) {
  const u32 nsa = (inst.raw[0] >> 1) & 0x3;
  const u32 vaddr = inst.raw[1] & 0xFF;
  std::array<u32, 8> reg;
  for (u32 i = 0; i < reg.size(); i++)
    reg[i] = std::min(vaddr + i, 255u);
  for (u32 d = 0; d < nsa; d++)
    for (u32 c = 0; c < 4 && 1 + d * 4 + c < reg.size(); c++)
      reg[1 + d * 4 + c] = (inst.raw[2 + d] >> (c * 8)) & 0xFF;
  std::array<Id, 8> address;
  for (u32 i = 0; i < address.size(); i++)
    address[i] = t.Vg(reg[i]);
  return address;
}

// Width and height of the level a sample reads, derived from the gfx10.3 T#
// exactly as the shared emitter derives them: a lowering that steps by whole
// texels has to agree with it or it lands in a different one.
struct LevelExtent {
  Id width;
  Id height;
};

LevelExtent MimgLevelExtent(Translator& t, u32 srsrc, Id lod) {
  const auto field = [&](u32 dword, u32 shift, u32 mask) {
    return t.And(t.Shr(t.Sg(srsrc + dword), t.U32(shift)), t.U32(mask));
  };
  const Id base_width = t.Add(
      t.Or(field(1, 30, 0x3), t.Shl(field(2, 0, 0xFFF), t.U32(2))), t.U32(1));
  const Id base_height = t.Add(field(2, 14, 0x3FFF), t.U32(1));
  const Id base_mip = field(3, 12, 0xF);
  const Id last_mip = t.UMax(base_mip, field(3, 16, 0xF));
  const Id mip = t.Add(base_mip, t.UMin(lod, t.Sub(last_mip, base_mip)));
  return {t.UMax(t.Shr(base_width, mip), t.U32(1)),
          t.UMax(t.Shr(base_height, mip), t.U32(1))};
}

Id ToFloat(Translator& t, Id u) {
  return t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {u});
}

// Move a normalised coordinate by whole texels of the sampled level.
Id StepTexels(Translator& t, Id coord, Id texels, Id extent) {
  return t.m.Bitcast(t.t_u,
                     t.FAdd(t.m.Bitcast(t.t_f, coord),
                            t.FDiv(texels, ToFloat(t, extent))));
}

// The gfx10.3 sample forms the shared emitter has no model for, expressed with
// the two it does (explicit-LOD and LOD-zero fetches of the staged image):
//   _c       the leading address dword is the depth reference, so fetch the
//            texel and then compare against it the way the S# asks.
//   _o       the leading address dword packs a signed per-axis texel offset,
//            and one texel is 1/extent of the normalised coordinate.
//   gather4  returns one component of the 2x2 footprint: four fetches, each
//            stepped onto its own texel.
// Declining one of these drops the whole dispatch, and with it every render
// target the pass was meant to fill.
void EmitLoweredMimg(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  const bool explicit_lod = op == 0x2c || op == 0x34 || op == 0x44;
  const bool compare = op == 0x2c || op == 0x2f;
  const bool offset = op == 0x34 || op == 0x37;
  const bool gather = op == 0x44 || op == 0x47;
  const bool da = (w & 0x4000) != 0;
  const u32 dmask = (w >> 8) & 0xF;
  const u32 vdata = (w1 >> 8) & 0xFF;
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
  const u32 ssamp = ((w1 >> 21) & 0x1F) * 4;

  std::array<Id, 8> address = MimgAddress(t, inst);
  const Id lead = address[0];
  if (compare || offset)  // the modifier rides in the first address dword
    for (u32 i = 0; i + 1 < address.size(); i++)
      address[i] = address[i + 1];
  // The 1D forms park the LOD one component earlier; they do not reach a
  // compute sampler in practice, so the extents follow the 2D placement.
  const Id lod =
      explicit_lod
          ? t.m.Emit(spv::Op::OpConvertFToU, t.t_u,
                     {t.Ext2(GLSLstd450FMax,
                             t.m.Bitcast(t.t_f, address[da ? 3 : 2]),
                             t.F32(0.f))})
          : t.U32(0);

  Inst plain = inst;
  plain.opcode = explicit_lod ? 0x24u : 0x27u;
  plain.raw[0] = (w & ~((0x7Fu << 18) | 1u)) | (plain.opcode << 18);

  if (offset) {
    const LevelExtent extent = MimgLevelExtent(t, srsrc, lod);
    // Six signed bits per axis: [5:0] x, [13:8] y. A z offset would need the
    // volume's depth, which only a 3D sample carries.
    const auto texels = [&](u32 shift) {
      const Id raw = t.And(t.Shr(lead, t.U32(shift)), t.U32(0x3F));
      const Id value = ToFloat(t, raw);
      return t.SelectF(t.Uge(raw, t.U32(0x20)), t.FSub(value, t.F32(64.f)),
                       value);
    };
    address[0] = StepTexels(t, address[0], texels(0), extent.width);
    address[1] = StepTexels(t, address[1], texels(8), extent.height);
  }

  if (gather) {
    const LevelExtent extent = MimgLevelExtent(t, srsrc, lod);
    // One component only, so each tap lands in its own destination register.
    const u32 component = dmask & (~dmask + 1u);
    // gather4 reports the footprint counter-clockwise from its lower left:
    // (x0,y1) (x1,y1) (x1,y0) (x0,y0).
    static const float step[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    for (u32 tap = 0; tap < 4; tap++) {
      std::array<Id, 8> tapped = address;
      tapped[0] = StepTexels(t, tapped[0], t.F32(step[tap][0]), extent.width);
      tapped[1] = StepTexels(t, tapped[1], t.F32(step[tap][1]), extent.height);
      Inst one = plain;
      one.raw[0] = (plain.raw[0] & ~(0xFu << 8)) | (component << 8);
      one.raw[1] = (w1 & ~0xFF00u) | (((vdata + tap) & 0xFF) << 8);
      gpu::gcn::EmitCsMimg(t, one, sc, tapped.data());
    }
    return;
  }

  gpu::gcn::EmitCsMimg(t, plain, sc, address.data());
  if (!compare)
    return;
  // S# word 0 [15:13] names the comparison; the fetch left the depth in the
  // first destination register.
  const Id depth = t.m.Bitcast(t.t_f, t.Vg(vdata));
  const Id reference = t.m.Bitcast(t.t_f, lead);
  const Id func = t.And(t.Shr(t.Sg(ssamp), t.U32(13)), t.U32(0x7));
  static const spv::Op kCompare[8] = {
      spv::Op::OpFOrdEqual,        // 0 never, selected away below
      spv::Op::OpFOrdLessThan,     spv::Op::OpFOrdEqual,
      spv::Op::OpFOrdLessThanEqual, spv::Op::OpFOrdGreaterThan,
      spv::Op::OpFOrdNotEqual,     spv::Op::OpFOrdGreaterThanEqual,
      spv::Op::OpFOrdEqual,        // 7 always, selected away below
  };
  Id pass = t.m.ConstBool(false);
  for (u32 f = 1; f < 8; f++) {
    const Id value = f == 7 ? t.m.ConstBool(true)
                            : t.m.Emit(kCompare[f], t.t_bool,
                                       {reference, depth});
    pass = t.m.Emit(spv::Op::OpSelect, t.t_bool,
                    {t.Eq(func, t.U32(f)), value, pass});
  }
  t.SetVgF(vdata, t.SelectF(pass, t.F32(1.f), t.F32(0.f)));
}

bool NoOpt() {
  return kGpuSpirvNoopt;
}

// A declined dispatch leaves the buffers it was meant to fill untouched, which
// is otherwise invisible in the frame. One line per distinct shader.
void ReportDecline(const u32* cs_code, size_t insts) {
  static std::unordered_set<u64> reported;
  const u64 address = reinterpret_cast<uintptr_t>(cs_code);
  if (!reported.insert(address).second)
    return;
  BASE_LOGI("rdnacs", "declined CS @{:#x} ({} insts) on [{}]; dispatch skipped",
            static_cast<unsigned long>(address), insts,
            gpu::gcn::UnsupportedOps().c_str());
}

bool TranslateCs(const Program& program,
                 u32 num_thread_x,
                 u32 num_thread_y,
                 u32 num_thread_z,
                 u32 user_sgpr,
                 u32 tgid_enable,
                 u32 lds_dwords,
                 RecompiledCs& r,
                 Translator& t) {
  if (program.empty())
    return false;
  DumpCsProgram(program);
  StageContext sc;
  sc.is_cs = true;
  sc.lds_dwords = lds_dwords * 128;  // RSRC2.LDS_SIZE is in 128-dword granules
  if (!PlanResources(program, sc.lds_dwords, user_sgpr, r, sc.cs_bind) ||
      r.resources.empty())
    return false;

  // Cross-lane work needs this invocation's lane. ds_swizzle is not the only
  // way a program asks for it: a DPP modifier is a lane shuffle too, and a
  // compute shader that used one without a lane id was rejected outright --
  // Astro Bot's world map skips a dozen dispatches that way.
  bool uses_lane_id = false;
  for (const Inst& inst : program)
    if ((inst.enc == Enc::kDs && inst.opcode == 0x35) ||
        inst.extension == gpu::gcn::InstExtension::kDpp ||
        inst.extension == gpu::gcn::InstExtension::kDpp8 ||
        inst.extension == gpu::gcn::InstExtension::kDpp8Fi) {
      uses_lane_id = true;
      break;
    }

  t.rdna_sources = true;
  t.InitTypes();
  // Storage buffers: Buf { uint data[]; } at set 0, binding = resource index.
  const Id t_run = t.m.TypeRuntimeArray(t.t_u);
  t.m.Decorate(t_run, spv::Decoration::ArrayStride, {4});
  const Id t_buf = t.m.TypeStruct({t_run});
  t.m.Decorate(t_buf, spv::Decoration::Block);
  t.m.MemberDecorate(t_buf, 0, spv::Decoration::Offset, {0});
  const Id p_buf = t.m.TypePointer(spv::StorageClass::StorageBuffer, t_buf);
  sc.cs_ssbo.resize(r.resources.size());
  for (const CsResource& res : r.resources) {
    const Id v = t.m.Variable(p_buf, spv::StorageClass::StorageBuffer);
    t.m.Decorate(v, spv::Decoration::DescriptorSet, {0});
    t.m.Decorate(v, spv::Decoration::Binding, {res.binding});
    t.m.Name(v, "buf" + std::to_string(res.binding));
    sc.cs_ssbo[res.binding] = v;
    if (res.runtime_address)
      sc.cs_runtime_resources[res.binding] = {res.kind, res.base_sgpr};
    if (res.runtime_image)
      sc.cs_runtime_images.insert(res.binding);
  }

  // GDS: the append/consume counters, in a buffer of their own past the
  // resources. It is device memory the title never names, so it needs no range.
  if (r.gds_binding >= 0) {
    const Id v = t.m.Variable(p_buf, spv::StorageClass::StorageBuffer);
    t.m.Decorate(v, spv::Decoration::DescriptorSet, {0});
    t.m.Decorate(v, spv::Decoration::Binding,
                 {static_cast<u32>(r.gds_binding)});
    t.m.Name(v, "gds");
    sc.gds_var = v;
  }

  // LDS: a Workgroup-storage uint array sized by RSRC2.
  if (sc.lds_dwords) {
    const Id lds_arr = t.m.TypeArray(t.t_u, sc.lds_dwords);
    sc.lds_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Workgroup, lds_arr),
                     spv::StorageClass::Workgroup);
    t.m.Name(sc.lds_var, "lds");
  }

  // Push constant: the 16 COMPUTE_USER_DATA dwords seed s0.. (descriptors +
  // params), then one bound (in dwords) per SSBO binding; 0 = unbounded. The
  // bounds come from the range the dispatch actually mapped: an OOB access
  // from the shader runs into whatever comes next if the emitted reads kept
  // going.
  constexpr u32 kUdWords = 16;
  constexpr u32 kBoundWords = 48;
  const Id t_arr16 = t.m.TypeArray(t.t_u, kUdWords);
  t.m.Decorate(t_arr16, spv::Decoration::ArrayStride, {4});
  const Id t_bounds = t.m.TypeArray(t.t_u, kBoundWords);
  t.m.Decorate(t_bounds, spv::Decoration::ArrayStride, {4});
  const Id t_pc = t.m.TypeStruct({t_arr16, t_bounds});
  t.m.Decorate(t_pc, spv::Decoration::Block);
  t.m.MemberDecorate(t_pc, 0, spv::Decoration::Offset, {0});
  t.m.MemberDecorate(t_pc, 1, spv::Decoration::Offset, {4 * kUdWords});
  const Id pc_var =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, t_pc),
                   spv::StorageClass::PushConstant);
  t.m.Name(pc_var, "user_data");
  sc.cs_bounds_var = pc_var;

  // Builtins: gl_LocalInvocationID (-> v0..v2) and gl_WorkGroupID (-> the
  // SGPRs after the user data, per tgid_enable).
  const Id t_uv3 = t.m.TypeVec(t.t_u, 3);
  const Id p_in_v3 = t.m.TypePointer(spv::StorageClass::Input, t_uv3);
  const Id local_id = t.m.Variable(p_in_v3, spv::StorageClass::Input);
  t.m.Decorate(local_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::LocalInvocationId)});
  const Id group_id = t.m.Variable(p_in_v3, spv::StorageClass::Input);
  t.m.Decorate(group_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::WorkgroupId)});
  std::vector<Id> iface{local_id, group_id};
  if (uses_lane_id) {
    t.m.Capability(spv::Capability::GroupNonUniform);
    t.m.Capability(spv::Capability::GroupNonUniformShuffle);
    sc.subgroup_local_id =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                     spv::StorageClass::Input);
    t.m.Decorate(
        sc.subgroup_local_id, spv::Decoration::BuiltIn,
        {static_cast<u32>(spv::BuiltIn::SubgroupLocalInvocationId)});
    iface.push_back(sc.subgroup_local_id);
  }

  if (r.guest_memory_binding >= 0)
    gpu::gcn::DeclareGuestMemory(t, sc, r.guest_memory_binding);
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  const Id p_pc_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (u32 i = 0; i < 16; i++)  // user data -> s0..s15
    t.SetSg(i, t.m.Load(t.t_u,
                        t.m.AccessChain(p_pc_u, pc_var, {t.U32(0), t.U32(i)})));
  const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
  const auto group_comp = [&](u32 c) {
    return t.m.Load(t.t_u, t.m.AccessChain(p_in_u, group_id, {t.U32(c)}));
  };
  u32 sg = user_sgpr;
  if ((tgid_enable & 1) && sg < 106)
    t.SetSg(sg++, group_comp(0));
  if ((tgid_enable & 2) && sg < 106)
    t.SetSg(sg++, group_comp(1));
  if ((tgid_enable & 4) && sg < 106)
    t.SetSg(sg++, group_comp(2));
  for (u32 c = 0; c < 3; c++)  // local invocation id (tidig) -> v0..v2
    t.SetVg(c, t.m.Load(t.t_u, t.m.AccessChain(p_in_u, local_id, {t.U32(c)})));
  if (sc.subgroup_local_id)
    t.lane_id = t.m.Load(t.t_u, sc.subgroup_local_id);
  t.SeedExec();
  t.predicate_vector = true;
  EmitCfg(t, program, sc);
  if (sc.cs_unsupported)
    return false;
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::GLCompute, main_fn, "main", iface);
  t.m.ExecMode(
      main_fn, spv::ExecutionMode::LocalSize,
      {num_thread_x ? num_thread_x : 1, num_thread_y ? num_thread_y : 1,
       num_thread_z ? num_thread_z : 1});
  r.local_size[0] = num_thread_x ? num_thread_x : 1;
  r.local_size[1] = num_thread_y ? num_thread_y : 1;
  r.local_size[2] = num_thread_z ? num_thread_z : 1;
  return true;
}

}  // namespace

bool EmitCsMemory(Translator& t, const Inst& inst, StageContext& sc) {
  sc.cs_cur_pc = inst.pc;
  switch (inst.enc) {
    case Enc::kSmrd:
      EmitSmem(t, inst, sc);
      return true;
    case Enc::kFlat:
      gpu::gcn::EmitCsGlobal(t, inst, sc);
      return true;
    case Enc::kMubuf:
      gpu::gcn::EmitCsMubuf(t, inst, sc);
      return true;
    case Enc::kMtbuf:
      gpu::gcn::EmitCsMtbuf(t, inst, sc);
      return true;
    case Enc::kDs:
      // The GDS bit picks the global counters, not LDS.
      if (((inst.raw[0] >> 17) & 1) && sc.gds_var)
        gpu::gcn::EmitGdsCounter(t, inst, sc);
      else
        gpu::gcn::EmitDs(t, inst, sc);
      return true;
    case Enc::kMimg: {
      if (inst.opcode == 0xe6 || inst.opcode == 0xe7) {
        EmitBvh(t, inst, sc);
        return true;
      }
      // Same lowering as the graphics path: dim 3 (cube) and 5 (2D array)
      // become the shared emitter's DA bit, and NSA names each address
      // component in its own VGPR.
      const u32 w = inst.raw[0];
      const u32 dim = (w >> 3) & 0x7;
      const bool arrayed_dim = dim == 3 || dim == 5;
      Inst lowered = inst;
      lowered.raw[0] = (w & ~0x4000u) | (arrayed_dim ? 0x4000u : 0u);
      switch (inst.opcode) {  // the depth-compare, offset and gather forms
        case 0x2c:
        case 0x2f:
        case 0x34:
        case 0x37:
        case 0x44:
        case 0x47:
          EmitLoweredMimg(t, lowered, sc);
          return true;
        default:
          break;
      }
      const u32 nsa = (w >> 1) & 0x3;
      if (nsa) {
        std::array<gpu::gcn::Id, 13> address{};
        address[0] = t.Vg(inst.raw[1] & 0xFF);
        for (u32 d = 0; d < nsa; d++)
          for (u32 c = 0; c < 4; c++)
            address[1 + d * 4 + c] = t.Vg((inst.raw[2 + d] >> (c * 8)) & 0xFF);
        gpu::gcn::EmitCsMimg(t, lowered, sc, address.data());
        return true;
      }
      gpu::gcn::EmitCsMimg(t, lowered, sc);
      return true;
    }
    default:
      return false;
  }
}

gpu::gcn::RecompiledCs RecompileCompute(const u32* cs_code,
                                        u32 num_thread_x,
                                        u32 num_thread_y,
                                        u32 num_thread_z,
                                        u32 user_sgpr,
                                        u32 tgid_enable,
                                        u32 lds_dwords,
                                        bool trap_present) {
  RecompiledCs r;
  const u32 code_dwords = ComputeCodeDwords(cs_code);
  if (!code_dwords)
    return r;
  Program program = ReachableProgram(DecodeShader(cs_code, code_dwords));
  if (program.empty())
    return r;
  const auto& last = program.back();
  if (last.pc + last.size >= code_dwords &&
      !(last.enc == Enc::kSopp && (last.opcode == 1 || last.opcode == 0x1f))) {
    gpu::gcn::WarnUnsupported("cs.truncated.rdna", last.pc);
    return r;
  }
  // RDNA2 STATUS.TRAP_EN=0 makes S_TRAP a hardware NOP. A present trap
  // handler still requires trap emulation and follows the unsupported path.
  if (!trap_present)
    for (Inst& inst : program)
      if (inst.enc == Enc::kSopp && inst.opcode == 0x12) {
        inst.opcode = 0;
        inst.raw[0] = 0xbf800000;
      }

  Translator t;
  RecompiledCs tmp;  // build into a temp so a mid-emit failure leaves r intact
  gpu::gcn::ResetUnsupported();
  if (!TranslateCs(program, num_thread_x, num_thread_y, num_thread_z, user_sgpr,
                   tgid_enable, lds_dwords, tmp, t) ||
      gpu::gcn::HadUnsupported()) {
    ReportDecline(cs_code, program.size());
    return r;
  }

  const std::vector<u32> spv_bin = t.m.Assemble();
  // A module the translator emitted but the validator rejects is a translator
  // bug (wrong codegen, not a guest gap): always loud.
  std::string err;
  // Use the same content-addressed optimizer cache as graphics. Large native
  // decoder kernels otherwise repeat legalization on every game launch.
  const bool no_opt = NoOpt();
  const bool valid = no_opt ? gpu::gcn::spirv::Validate(spv_bin, &err)
                            : gpu::gcn::spirv::Finalize(spv_bin, &tmp.spirv, &err);
  if (!valid) {
    BASE_LOGI("rdnacs", "CS invalid @{:p}: {}",
              static_cast<const void*>(cs_code), err.c_str());
    return r;
  }
  if (no_opt)
    tmp.spirv = spv_bin;
  if (tmp.spirv.empty())
    return r;
  tmp.ok = true;
  return tmp;
}

}  // namespace gpu::rdna

#endif  // DELTA_HAVE_SPIRV_BACKEND
