/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/gcn/spirv/gcn_spirv.h"
#include "base/arch.h"
#include "guest/session.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::gcn {
bool RecompileSpirv(const u32*,
                    const u32*,
                    const u32*,
                    const u32*,
                    u32,
                    const u32*,
                    u32,
                    u32,
                    u32,
                    u32,
                    u32,
                    u32,
                    bool,
                    const GsPipeline*,
                    u32,
                    u32,
                    u32,
                    Recompiled&) {
  return false;
}
bool RecompileComputeSpirv(const u32*,
                           u32,
                           u32,
                           u32,
                           u32,
                           u32,
                           u32,
                           u64,
                           u64,
                           u64,
                           RecompiledCs&) {
  return false;
}
}  // namespace gpu::gcn
#else

#include <cstdio>
#include <cstdlib>

#include <cstring>
#include "base/logging.h"
#include "base/strings/to_string.h"

#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/hashing/hash.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/gcn_audit.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/gcn/spirv/translator.h"
#include "options/options.h"

namespace {
DELTA_OPTION(u32, kCfgMaxIter, "DELTA_GPU_CFG_MAXITER", 16384);
DELTA_OPTION(const char*, kShDisAddr, "DELTA_GPU_SHDIS_ADDR", nullptr);
DELTA_OPTION(bool, kGpuNokill, "DELTA_GPU_NOKILL", false);
DELTA_OPTION(const char*, kProbePs, "DELTA_GPU_PROBE_PS", nullptr);
DELTA_OPTION(bool, kGpuPswhite, "DELTA_GPU_PSWHITE", false);
DELTA_OPTION(bool, kProbeAlpha, "DELTA_GPU_PSPROBE_A", false);
DELTA_OPTION(int, kGpuPstex, "DELTA_GPU_PSTEX", 0);
DELTA_OPTION(float, kGpuPstexScale, "DELTA_GPU_PSTEXSCALE", 1.f);
DELTA_OPTION(int, kGpuPsattr, "DELTA_GPU_PSATTR", 0);
DELTA_OPTION(bool, kGpuPsreach, "DELTA_GPU_PSREACH", false);
DELTA_OPTION(int, kGpuPsvgpr, "DELTA_GPU_PSVGPR", 0);
DELTA_OPTION(u32, kGpuPsvgprAt, "DELTA_GPU_PSVGPR_AT", 0);
DELTA_OPTION(bool, kGpuShdis, "DELTA_GPU_SHDIS", false);
DELTA_OPTION(u64, kGcnCsVgprCs, "DELTA_GPU_GCN_CSVGPR_CS", 0);
DELTA_OPTION(u32, kGcnCsVgprPc, "DELTA_GPU_GCN_CSVGPR_PC", 0);
DELTA_OPTION(u32, kGcnCsVgpr, "DELTA_GPU_GCN_CSVGPR", 0);
DELTA_OPTION(bool, kGpuShtrace, "DELTA_GPU_SHTRACE", false);
DELTA_OPTION(bool, kGpuSpirv, "DELTA_GPU_SPIRV", false);
DELTA_OPTION(u64, kSpvDumpAddr, "DELTA_GPU_SPVDUMP_ADDR", 0);
DELTA_OPTION(bool, kGpuSpirvCfg, "DELTA_GPU_SPIRV_CFG", false);
DELTA_OPTION(bool, kGpuSpirvNoopt, "DELTA_GPU_SPIRV_NOOPT", false);
DELTA_OPTION(bool, kGpuVsflipz, "DELTA_GPU_VSFLIPZ", false);
DELTA_OPTION(bool, kGpuNoZRemap, "DELTA_GPU_NOZREMAP", false);
DELTA_OPTION(float, kGpuVsForceZ, "DELTA_GPU_VSFORCEZ", 0.0f);
DELTA_OPTION(float, kGpuVsProbeW, "DELTA_GPU_VSPROBEW", 0.0f);
DELTA_OPTION(bool, kGpuVsfull, "DELTA_GPU_VSFULL", false);
DELTA_OPTION(bool, kGpuVsNoPred, "DELTA_GPU_VSNOPRED", false);
}  // namespace

namespace gpu::gcn {

namespace {
bool CodeMatches(const char* spec, const void* code) {
  const u64 wanted = std::strtoull(spec, nullptr, 16);
  if (reinterpret_cast<u64>(code) == wanted)
    return true;
  const auto* dwords = static_cast<const u32*>(code);
  u64 h = 1469598103934665603ull;
  for (u32 i = 0; i < 8192 && code; i++) {
    h = (h ^ dwords[i]) * 1099511628211ull;
    if (dwords[i] == 0xBF810000u)
      break;
  }
  return h == wanted;
}

bool ProbeThisPs(const Translator& t) {
  return !kProbePs ||
         CodeMatches(kProbePs, reinterpret_cast<const void*>(t.program_base));
}

thread_local bool g_had_unsupported = false;
thread_local base::String g_unsupported_ops;

bool IsVopc64(u32 op) {
  return (op >= 0x20 && op <= 0x3f) || (op >= 0x60 && op <= 0x7f) ||
         (op >= 0xa0 && op <= 0xbf) || op >= 0xe0;
}
}  // namespace

void ResetUnsupported() {
  g_had_unsupported = false;
  g_unsupported_ops.clear();
}

bool HadUnsupported() {
  return g_had_unsupported;
}

const base::String& UnsupportedOps() {
  return g_unsupported_ops;
}

bool TraceEnabled() {
  return kGpuShtrace;
}

void NoteApproximated(const char* enc, u32 op) {
  AuditNote(enc, op);
  static base::HashSet<u64> seen;
  static const guest::SessionReset reset_seen(
      [] { guest::ResetResource(seen); });
  const u64 key =
      base::HashBytes(enc, std::strlen(enc)) ^ (static_cast<u64>(op) << 40);
  if (seen.size() > 512 || !seen.insert(key).second)
    return;
  BASE_LOGI("gcnspv", "APPROXIMATED {} op={:#x}", enc, op);
}

void WarnUnsupported(const char* enc, u32 op, u32 w0, u32 w1) {
  g_had_unsupported = true;
  {
    char one[64];
    std::snprintf(one, sizeof(one), "%s:%#x", enc, op);
    if (g_unsupported_ops.find(one) == base::String::npos) {
      if (!g_unsupported_ops.empty())
        g_unsupported_ops += ' ';
      g_unsupported_ops += one;
    }
  }
  AuditNote(enc, op);
  static base::HashSet<u64> seen;
  static const guest::SessionReset reset_seen(
      [] { guest::ResetResource(seen); });
  const u64 key =
      base::HashBytes(enc, std::strlen(enc)) ^ (static_cast<u64>(op) << 40);
  if (seen.size() > 512 || !seen.insert(key).second)
    return;
  BASE_LOGI("gcnspv", "UNSUPPORTED {} op={:#x} (w0={:#x} w1={:#x}) -> rejected",
            enc, op, w0, w1);
}

u32 PsAttrLocation(const StageContext& sc, u32 attr) {
  if (!sc.ps_in_cntl || attr >= sc.ps_num_interp)
    return attr;
  const u32 off = sc.ps_in_cntl[attr & 31] & 0x1F;
  if (sc.vs_exported_params && off < sc.vs_exported_params->size())
    return (*sc.vs_exported_params)[off];
  return off;
}

Id PsInputVar(Translator& t, StageContext& sc, u32 attr) {
  const u32 loc0 = PsAttrLocation(sc, attr);
  auto it = sc.in_vars.find(loc0);
  if (it != sc.in_vars.end())
    return it->second;
  const u32 loc = loc0;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                            spv::StorageClass::Input);
  t.m.Decorate(v, spv::Decoration::Location, {loc});
  if (sc.flat_attrs && sc.flat_attrs->count(attr))
    t.m.Decorate(v, spv::Decoration::Flat);
  t.m.Name(v, "in_attr" + base::ToString(attr));
  sc.iface->push_back(v);
  sc.in_vars[loc] = v;
  return v;
}

base::HashSet<u32> PlanPerVertexAttrs(const Program& program,
                                      const u8* reachable) {
  base::HashSet<u32> out;
  for (size_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (inst.enc != Enc::kVintrp || (reachable && !reachable[i]))
      continue;
    const u32 w = inst.raw[0];
    if (((w >> 16) & 3) != 2)
      continue;
    const u32 vsrc = w & 0xFF;
    if (vsrc == 0 || vsrc == 1)
      out.insert((w >> 10) & 0x3F);
  }
  return out;
}

struct VgprWrite {
  bool known = true;
  u32 dst = 0;
  u32 count = 0;
};

VgprWrite VgprDestOf(const Inst& inst) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSop1:
    case Enc::kSop2:
    case Enc::kSopk:
    case Enc::kSopc:
    case Enc::kSopp:
    case Enc::kSmrd:
    case Enc::kVopc:
    case Enc::kExp:
      return {};
    case Enc::kVop1:
      return inst.opcode == 0x02 ? VgprWrite{}
                                 : VgprWrite{true, (w >> 17) & 0xFF, 1};
    case Enc::kVop2:
      return inst.opcode == 0x01 ? VgprWrite{}
                                 : VgprWrite{true, (w >> 17) & 0xFF, 1};
    case Enc::kVop3:
      if (inst.opcode < 0x100 || inst.opcode == 0x101)
        return {};
      return {true, w & 0xFF, 2};
    case Enc::kVop3p:
      return {true, w & 0xFF, 1};
    case Enc::kVintrp:
      return {true, (w >> 18) & 0xFF, 1};
    case Enc::kDs:
      return {true, (w1 >> 24) & 0xFF, 2};
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kMimg:
      return {true, (w1 >> 8) & 0xFF, 4};
    default:
      return {false, 0, 0};
  }
}

bool CollectLabels(const Program& program,
                   const u8* reachable,
                   base::HashSet<u32>& labels) {
  for (size_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (reachable && !reachable[i])
      continue;
    if (inst.enc == Enc::kSop1 && (inst.opcode == 0x20 || inst.opcode == 0x21))
      return false;
    if (inst.enc != Enc::kSopp)
      continue;
    const bool branch =
        inst.opcode == 0x02 || (inst.opcode >= 0x04 && inst.opcode <= 0x0b);
    if (!branch)
      continue;
    const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
    labels.insert(
        static_cast<u32>(static_cast<i64>(inst.pc) + inst.size + simm));
  }
  return true;
}

int NearestVgprDef(const Program& program,
                   const u8* reachable,
                   const base::HashSet<u32>& labels,
                   size_t use,
                   u32 reg) {
  for (size_t i = use; i-- > 0;) {
    const Inst& inst = program[i];
    if (reachable && !reachable[i])
      continue;
    if (labels.count(inst.pc))
      return -2;
    const VgprWrite d = VgprDestOf(inst);
    if (!d.known)
      return -2;
    if (d.count && reg >= d.dst && reg < d.dst + d.count)
      return static_cast<int>(i);
  }
  return -1;
}

bool IsLaneChainStep(const Inst& inst,
                     u32 want_op,
                     u32 want_src0,
                     int want_src1,
                     u32* src1_reg) {
  const bool vop2 = inst.enc == Enc::kVop2 && inst.opcode == want_op;
  const bool vop3 = inst.enc == Enc::kVop3 && inst.opcode == 0x100 + want_op;
  if (!vop2 && !vop3)
    return false;
  const u32 src0 = vop3 ? (inst.raw[1] & 0x1FF) : (inst.raw[0] & 0x1FF);
  const u32 src1 =
      vop3 ? ((inst.raw[1] >> 9) & 0x1FF) : (256 + ((inst.raw[0] >> 9) & 0xFF));
  if (src0 != want_src0)
    return false;
  if (want_src1 >= 0)
    return src1 == static_cast<u32>(want_src1);
  if (src1 < 256)
    return false;
  if (src1_reg)
    *src1_reg = src1 - 256;
  return true;
}

base::HashSet<u32> PlanDsOwnLane(const Program& program, const u8* reachable) {
  base::HashSet<u32> out;
  base::HashSet<u32> labels;
  if (!CollectLabels(program, reachable, labels))
    return out;
  for (size_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (inst.enc != Enc::kDs || (reachable && !reachable[i]))
      continue;
    if (inst.opcode == 176 || inst.opcode == 177) {
      out.insert(inst.pc);
      continue;
    }
    u32 reg = inst.raw[1] & 0xFF;
    static constexpr u32 kOps[3] = {0x1a, 0x23, 0x24};
    static constexpr u32 kSrc0[3] = {130, 193, 193};
    static constexpr int kSrc1[3] = {-1, -1, 128};
    size_t at = i;
    bool ok = true;
    for (u32 step = 0; step < 3 && ok; step++) {
      const int def = NearestVgprDef(program, reachable, labels, at, reg);
      if (def < 0 || !IsLaneChainStep(program[def], kOps[step], kSrc0[step],
                                      kSrc1[step], &reg)) {
        ok = false;
        break;
      }
      at = static_cast<size_t>(def);
    }
    if (ok)
      out.insert(inst.pc);
  }
  return out;
}

Id PsPerVertexVar(Translator& t, StageContext& sc, u32 attr) {
  const u32 loc0 = PsAttrLocation(sc, attr);
  auto it = sc.pervertex_vars.find(loc0);
  if (it != sc.pervertex_vars.end())
    return it->second;
  t.m.Capability(spv::Capability::FragmentBarycentricKHR);
  t.m.Extension("SPV_KHR_fragment_shader_barycentric");
  const Id arr = t.m.TypeArray(t.t_v4, 3);
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, arr),
                            spv::StorageClass::Input);
  const u32 loc = loc0;
  t.m.Decorate(v, spv::Decoration::Location, {loc});
  t.m.Decorate(v, spv::Decoration::PerVertexKHR);
  if (sc.flat_attrs && sc.flat_attrs->count(attr))
    t.m.Decorate(v, spv::Decoration::Flat);
  t.m.Name(v, "in_attr" + base::ToString(attr) + "_pv");
  sc.iface->push_back(v);
  sc.pervertex_vars[loc] = v;
  return v;
}

Id PsBaryCoord(Translator& t, StageContext& sc) {
  if (sc.bary_var)
    return sc.bary_var;
  t.m.Capability(spv::Capability::FragmentBarycentricKHR);
  t.m.Extension("SPV_KHR_fragment_shader_barycentric");
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v3),
                            spv::StorageClass::Input);
  t.m.Decorate(v, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::BaryCoordKHR)});
  t.m.Name(v, "bary");
  sc.iface->push_back(v);
  sc.bary_var = v;
  return v;
}

Id PsBaryCoordNoPersp(Translator& t, StageContext& sc) {
  if (sc.bary_nopersp_var)
    return sc.bary_nopersp_var;
  t.m.Capability(spv::Capability::FragmentBarycentricKHR);
  t.m.Extension("SPV_KHR_fragment_shader_barycentric");
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v3),
                            spv::StorageClass::Input);
  t.m.Decorate(v, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::BaryCoordNoPerspKHR)});
  t.m.Name(v, "bary_np");
  sc.iface->push_back(v);
  sc.bary_nopersp_var = v;
  return v;
}

void SeedPsBarycentrics(Translator& t, u32 ena, StageContext& sc) {
  if (!sc.is_ps || !(ena & 0x7Fu))
    return;
  static constexpr u8 kWidth[16] = {2, 2, 2, 3, 2, 2, 2, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1};
  u32 vg[16] = {}, next = 0;
  for (u32 bit = 0; bit < 16; bit++)
    if (ena & (1u << bit)) {
      vg[bit] = next;
      next += kWidth[bit];
    }
  const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
  const auto seed = [&](u32 bit, Id bary) {
    if (!(ena & (1u << bit)))
      return;
    for (u32 c = 0; c < 2; c++)
      t.SetVgF(vg[bit] + c,
               t.m.Load(t.t_f, t.m.AccessChain(p_in_f, bary, {t.U32(c + 1)})));
  };
  if (ena & 0x7u) {
    const Id b = PsBaryCoord(t, sc);
    for (u32 bit = 0; bit < 3; bit++)
      seed(bit, b);
  }
  if (ena & 0x70u) {
    const Id b = PsBaryCoordNoPersp(t, sc);
    for (u32 bit = 4; bit < 7; bit++)
      seed(bit, b);
  }
}

bool ReadsPerVertex(StageContext& sc, u32 attr) {
  if (sc.pervertex_attrs.count(attr))
    return true;
  const u32 loc = PsAttrLocation(sc, attr);
  for (u32 a : sc.pervertex_attrs)
    if (PsAttrLocation(sc, a) == loc)
      return true;
  return false;
}

void EmitVintrp(Translator& t, u32 w, StageContext& sc) {
  if (!sc.is_ps)
    return;
  const u32 chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F;
  const u32 op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
  if (ReadsPerVertex(sc, attr)) {
    const Id var = PsPerVertexVar(t, sc, attr);
    const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
    const auto vert = [&](u32 i) {
      return t.m.Load(t.t_f,
                      t.m.AccessChain(p_in_f, var, {t.U32(i), t.U32(chan)}));
    };
    if (op == 0)
      return;
    if (op == 1) {
      const Id b = PsBaryCoord(t, sc);
      const Id p_bary_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
      const auto weight = [&](u32 i) {
        return t.m.Load(t.t_f, t.m.AccessChain(p_bary_f, b, {t.U32(i)}));
      };
      t.SetVgF(vdst, t.FAdd(t.FAdd(t.FMul(vert(0), weight(0)),
                                   t.FMul(vert(1), weight(1))),
                            t.FMul(vert(2), weight(2))));
      return;
    }
    const u32 vsrc = w & 0xFF;
    const bool passthrough = sc.ps_in_cntl && attr < sc.ps_num_interp &&
                             (sc.ps_in_cntl[attr] & 0x420u) == 0x420u;
    if (vsrc == 2)
      t.SetVgF(vdst, vert(0));
    else if (vsrc == 0)
      t.SetVgF(vdst, passthrough ? vert(1) : t.FSub(vert(1), vert(0)));
    else if (vsrc == 1)
      t.SetVgF(vdst, passthrough ? vert(2) : t.FSub(vert(2), vert(0)));
    return;
  }
  if (op == 1 || (op == 2 && (w & 0xFF) == 2)) {
    const Id v = PsInputVar(t, sc, attr);
    const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
    t.SetVgF(vdst, t.m.Load(t.t_f, t.m.AccessChain(p_in_f, v, {t.U32(chan)})));
  }
}

Id VsParamOut(Translator& t, StageContext& sc, u32 p) {
  auto it = sc.param_outs.find(p);
  if (it != sc.param_outs.end())
    return it->second;
  const auto storage =
      sc.is_mesh ? spv::StorageClass::Private : spv::StorageClass::Output;
  const Id v = t.m.Variable(t.m.TypePointer(storage, t.t_v4), storage);
  if (!sc.is_mesh)
    t.m.Decorate(v, spv::Decoration::Location, {p});
  if (!sc.is_mesh && sc.flat_attrs && sc.flat_attrs->count(p))
    t.m.Decorate(v, spv::Decoration::Flat);
  t.m.Name(v, "out_param" + base::ToString(p));
  if (!sc.is_mesh)
    sc.iface->push_back(v);
  sc.param_outs[p] = v;
  return v;
}

Id PsColorOut(Translator& t, StageContext& sc, u32 target) {
  if (sc.color_outs[target])
    return sc.color_outs[target];
  const bool integer = (sc.mrt_uint_mask >> target) & 1u;
  const Id type = integer ? t.m.TypeVec(t.t_u, 4) : t.t_v4;
  const Id init =
      integer
          ? t.m.ConstComposite(type, {t.U32(0), t.U32(0), t.U32(0), t.U32(1)})
          : t.m.ConstComposite(
                type, {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(1.f)});
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, type),
                            spv::StorageClass::Output, init);
  t.m.Decorate(v, spv::Decoration::Location, {target});
  t.m.Name(v, "mrt" + base::ToString(target));
  sc.iface->push_back(v);
  sc.color_outs[target] = v;
  sc.r->ps_mrt_mask |= static_cast<u8>(1u << target);
  return v;
}

void ExportColor(Translator& t,
                 StageContext& sc,
                 u32 target,
                 u32 en,
                 u32 compr,
                 const u32 v[4],
                 bool int_target,
                 Id c[4],
                 bool live[4]) {
  u32 fmt = sc.col_format == kColFormatUnknown
                ? (compr ? 4u : 9u)
                : (sc.col_format >> (4 * target)) & 0xFu;
  const bool packed = fmt >= 4 && fmt <= 8;
  if (fmt == 0 || packed != (compr != 0))
    fmt = compr ? 4u : 9u;
  for (u32 i = 0; i < 4; i++) {
    c[i] = int_target ? t.U32(i == 3 ? 1u : 0u) : t.F32(i == 3 ? 1.f : 0.f);
    live[i] = false;
  }
  const auto dword = [&](u32 i) {
    c[i] = int_target ? t.Vg(v[i]) : t.VgF(v[i]);
    live[i] = (en >> i) & 1u;
  };
  switch (fmt) {
    case 1:
      dword(0);
      return;
    case 2:
      dword(0);
      dword(1);
      return;
    case 3:
      dword(0);
      dword(3);
      return;
    case 9:
      for (u32 i = 0; i < 4; i++)
        dword(i);
      return;
    default:
      break;
  }
  for (u32 p = 0; p < 2; p++) {
    if (!(en & (0x3u << (2 * p))))
      continue;
    const Id word = t.Vg(v[p]);
    live[2 * p] = live[2 * p + 1] = true;
    if (fmt == 7 || fmt == 8) {
      const spv::Op ext =
          fmt == 8 ? spv::Op::OpBitFieldSExtract : spv::Op::OpBitFieldUExtract;
      for (u32 h = 0; h < 2; h++) {
        const Id x = t.m.Emit(ext, t.t_u, {word, t.U32(16 * h), t.U32(16)});
        c[2 * p + h] =
            int_target
                ? x
                : t.m.Emit(fmt == 8 ? spv::Op::OpConvertSToF
                                    : spv::Op::OpConvertUToF,
                           t.t_f, {fmt == 8 ? t.m.Bitcast(t.t_i, x) : x});
      }
      continue;
    }
    if (int_target) {
      for (u32 h = 0; h < 2; h++)
        c[2 * p + h] = t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u,
                                {word, t.U32(16 * h), t.U32(16)});
      continue;
    }
    const u32 unpack = fmt == 5   ? GLSLstd450UnpackUnorm2x16
                       : fmt == 6 ? GLSLstd450UnpackSnorm2x16
                                  : GLSLstd450UnpackHalf2x16;
    const Id pair = t.m.ExtInst(t.t_v2, unpack, {word});
    c[2 * p] = t.m.CompositeExtract(t.t_f, pair, 0);
    c[2 * p + 1] = t.m.CompositeExtract(t.t_f, pair, 1);
  }
}

Id PsDepthOut(Translator& t, StageContext& sc) {
  if (sc.depth_out)
    return sc.depth_out;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_f),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::FragDepth)});
  sc.iface->push_back(v);
  sc.depth_out = v;
  if (sc.main_fn)
    t.m.ExecMode(sc.main_fn, spv::ExecutionMode::DepthReplacing);
  return v;
}

namespace {

struct FetchAttr {
  u32 semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
  bool direct_fetch = false;
  u32 pc = ~0u;
  u32 dfmt = 0, nfmt = 0;
  u32 index_vgpr = 0;
};

base::Vector<FetchAttr> ParseFetch(u64 fetch_addr) {
  base::Vector<FetchAttr> out;
  if (!InGuest(fetch_addr))
    return out;
  const auto* code = reinterpret_cast<const u32*>(fetch_addr);
  const Program program = Decode(code, 256);
  struct Load {
    u32 table_sgpr, dword_off;
  };
  base::HashMap<u32, Load> loads;
  u32 semantic = 0;
  for (const Inst& inst : program) {
    const u32 w = inst.raw[0];
    if (inst.enc == Enc::kSop1 && (inst.opcode == 0x20 || inst.opcode == 0x21))
      break;
    if (inst.enc == Enc::kSopp && inst.opcode == 1)
      break;
    if (inst.enc == Enc::kSmrd && inst.opcode == 0x02) {
      const u32 sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
      loads[sdst] = {sbase * 2u, w & 0xFF};
    } else if (inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf) {
      const u32 w1 = inst.raw[1];
      const u32 vdata = (w1 >> 8) & 0xFF;
      const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
      const u32 nc = (inst.opcode & 3) + 1;
      const bool typed = inst.enc == Enc::kMtbuf;
      const auto it = loads.find(srsrc);
      const u32 tbl = it != loads.end() ? it->second.table_sgpr : 0;
      const u32 off = it != loads.end() ? it->second.dword_off : 0;
      out.push_back({semantic, nc, vdata, tbl, off, false, ~0u,
                     typed ? (w >> 19) & 0xF : 0, typed ? (w >> 23) & 0x7 : 0,
                     (w >> 13) & 1 ? w1 & 0xFF : 0});
      semantic++;
    }
  }
  if (out.empty() && kGpuShdis) {
    static int n = 0;
    if (n++ < 8) {
      BASE_LOGI("gcnspv", "fetch shader parsed empty @{:#x}", fetch_addr);
      DisassembleAt(fetch_addr, "fetch");
    }
  }
  return out;
}

void EmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  if (sc.direct &&
      (inst.enc == Enc::kSmrd || inst.enc == Enc::kMubuf ||
       inst.enc == Enc::kMtbuf) &&
      EmitDirectLoad(t, inst, sc))
    return;
  if (inst.extension == InstExtension::kSdwa ||
      inst.extension == InstExtension::kDpp) {
    WarnUnsupported(inst.extension == InstExtension::kSdwa ? "sdwa" : "dpp",
                    inst.opcode, w, w1);
    return;
  }
  const bool uses_lds_direct =
      ((inst.enc == Enc::kVop1 || inst.enc == Enc::kVop2 ||
        inst.enc == Enc::kVopc) &&
       (w & 0x1ff) == 254) ||
      ((inst.enc == Enc::kVop3 || inst.enc == Enc::kVop3p) &&
       ((w1 & 0x1ff) == 254 || ((w1 >> 9) & 0x1ff) == 254 ||
        ((w1 >> 18) & 0x1ff) == 254));
  if (uses_lds_direct) {
    WarnUnsupported("lds_direct", inst.opcode, w, w1);
    return;
  }
  const bool uses_neo_inline =
      ((inst.enc == Enc::kVop1 || inst.enc == Enc::kVop2 ||
        inst.enc == Enc::kVopc) &&
       (w & 0x1ff) == 248) ||
      ((inst.enc == Enc::kVop3 || inst.enc == Enc::kVop3p) &&
       ((w1 & 0x1ff) == 248 || ((w1 >> 9) & 0x1ff) == 248 ||
        ((w1 >> 18) & 0x1ff) == 248)) ||
      ((inst.enc == Enc::kSop2 || inst.enc == Enc::kSopc) &&
       ((w & 0xff) == 248 || ((w >> 8) & 0xff) == 248)) ||
      (inst.enc == Enc::kSop1 && (w & 0xff) == 248) ||
      (inst.enc == Enc::kSmrd && ((w >> 8) & 1) == 0 && (w & 0xff) == 248);
  if (inst.isa == IsaMode::kBase && uses_neo_inline) {
    WarnUnsupported("source.inv_2pi.neo", inst.opcode, w, w1);
    return;
  }
  if (inst.isa == IsaMode::kBase && inst.enc == Enc::kVop3 &&
      ((w1 & 0x1ff) == 255 || ((w1 >> 9) & 0x1ff) == 255 ||
       ((w1 >> 18) & 0x1ff) == 255)) {
    WarnUnsupported("vop3.literal.neo", inst.opcode, w, w1);
    return;
  }
  switch (inst.enc) {
    case Enc::kSop1:
      EmitSop1(t, inst);
      break;
    case Enc::kSop2:
      EmitSop2(t, inst);
      break;
    case Enc::kSopc:
      EmitSopc(t, inst);
      break;
    case Enc::kSopk:
      EmitSopk(t, inst);
      break;
    case Enc::kSopp:
      if (inst.opcode == 0x0a && sc.is_cs) {
        t.m.EmitVoid(spv::Op::OpControlBarrier,
                     {t.U32(2), t.U32(2), t.U32(0x108)});
      } else if (inst.opcode == 0x10 && sc.gs_input_verts) {
        EmitGsMessage(t, inst, sc);
      } else if (inst.opcode >= 0x16 && inst.opcode <= 0x19 &&
                 static_cast<i16>(w & 0xffff) == 0) {
      } else if (inst.opcode == 0x12) {
        // s_trap: no trap handler is installed, so it never stops the wave.
      } else if (inst.opcode != 0x00 && inst.opcode != 0x01 &&
                 inst.opcode != 0x02 &&
                 !(inst.opcode >= 0x04 && inst.opcode <= 0x0a) &&
                 inst.opcode != 0x0c) {
        WarnUnsupported("sopp", inst.opcode, w, w1);
      }
      break;
    case Enc::kSmrd:
      if (sc.is_cs)
        EmitCsSmrd(t, inst, sc);
      else
        EmitCbufSmrd(t, inst, sc.cbuf_bind);
      break;
    case Enc::kVop1: {
      if (inst.isa == IsaMode::kNeo && EmitNeoVop1(t, inst))
        break;
      const u32 op = inst.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
      if (const auto known = t.movrel_m0.find(inst.pc);
          known != t.movrel_m0.end() && op >= 0x42 && op <= 0x44 &&
          (op == 0x42 || src0 >= 256)) {
        const auto& values = known->second;
        const Id v = op == 0x42 ? t.SrcRaw(src0, inst.literal)
                                : t.VgIndexed(src0 - 256, values);
        if (op == 0x43)
          t.SetVg(vdst, v);
        else
          t.SetVgIndexed(vdst, values, v);
        break;
      }
      if (op == 0x42) {
        t.SetVgRel(vdst, t.SrcRaw(src0, inst.literal));
        break;
      }
      if ((op == 0x43 || op == 0x44) && src0 >= 256) {
        const Id v = t.m.Load(t.t_u, t.VgRelPtr(src0 - 256));
        if (op == 0x43)
          t.SetVg(vdst, v);
        else
          t.SetVgRel(vdst, v);
        break;
      }
      EmitVop1(t, op, vdst, t.SrcF(src0, inst.literal));
      break;
    }
    case Enc::kVop2: {
      if (inst.isa == IsaMode::kNeo && EmitNeoVop2(t, inst))
        break;
      const u32 op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const u32 vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      const u32 src1 = op == 0x01 || op == 0x02 ? vsrc1 : 256 + vsrc1;
      if (EmitLaneSpill(t, op, vdst, src0, src1, inst.literal) ||
          EmitMbcntOfExec(t, op, vdst, src0, src1, inst.literal))
        break;
      EmitVop2(t, op, vdst, t.SrcF(src0, inst.literal),
               t.SrcF(src1, inst.literal), inst.literal);
      break;
    }
    case Enc::kVop3: {
      if (inst.isa == IsaMode::kNeo && EmitNeoVop3(t, inst))
        break;
      const u32 op = inst.opcode, vdst = w & 0xFF;
      const bool vop3b = IsVop3b(op);
      const u32 sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const u32 abs = vop3b ? 0 : ((w >> 8) & 7);
      const bool clamp = !vop3b && ((w >> 11) & 1);
      const u32 s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const u32 s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      const u32 omod = (w1 >> 27) & 3;
      if ((op == 0x101 || op == 0x102) &&
          EmitLaneSpill(t, op - 0x100, vdst, s0, s1, inst.literal))
        break;
      if (op >= 0x100 && op < 0x140 && !neg && !abs &&
          EmitMbcntOfExec(t, op - 0x100, vdst, s0, s1, inst.literal))
        break;
      Id source0 = t.SrcF(s0, inst.literal, neg & 1, abs & 1);
      if (op == 0x18b) {
        source0 =
            t.m.CompositeExtract(t.t_f,
                                 t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                                             {t.SrcRaw(s0, inst.literal)}),
                                 0);
        if (abs & 1)
          source0 = t.Ext1(GLSLstd450FAbs, source0);
        if (neg & 1)
          source0 = t.FNeg(source0);
      }
      if (op < 0x100 && IsVopc64(op) &&
          (s0 >= 128 || s1 >= 128 || ((neg | abs) & 3)))
        WarnUnsupported("vopc64.vop3-operand", op, w, w1);
      EmitVop3(t, op, vdst, source0, t.SrcRawHi(s0, inst.literal, op == 0x163),
               t.SrcF(s1, inst.literal, neg & 2, abs & 2),
               t.SrcF(s2, inst.literal, neg & 4, abs & 4),
               t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp, omod,
               t.SrcRawHi(s1, inst.literal, false));
      break;
    }
    case Enc::kVop3p:
      if (!EmitNeoVop3p(t, inst))
        WarnUnsupported("vop3p", inst.opcode, w, w1);
      break;
    case Enc::kVopc: {
      const u32 op = inst.opcode;
      const u32 vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      if (inst.isa == IsaMode::kNeo &&
          EmitNeoVopc(t, op, 106, src0, 256 + vsrc1, inst.literal))
        break;
      if (IsVopc64(op) && src0 >= 128)
        WarnUnsupported("vopc64.inline-or-literal-operand", op, w,
                        inst.literal);
      EmitVopc(t, op, t.SrcF(src0, inst.literal),
               t.SrcF(256 + vsrc1, inst.literal), t.SrcRaw(src0, inst.literal),
               t.SrcRaw(256 + vsrc1, inst.literal), 106,
               t.SrcRawHi(src0, inst.literal, false),
               t.SrcRawHi(256 + vsrc1, inst.literal, false));
      break;
    }
    case Enc::kVintrp:
      EmitVintrp(t, w, sc);
      break;
    case Enc::kMubuf:
      if (sc.gs_input_verts) {
        EmitGsRingAccess(t, inst, sc);
      } else if (sc.es_ring && ((w >> 18) & 0x7F) >= 0x18 &&
                 ((w >> 18) & 0x7F) <= 0x1f) {
        EmitEsRingStore(t, inst, sc);
      } else if (sc.is_cs) {
        EmitCsMubuf(t, inst, sc);
      } else if (!sc.is_ps && sc.direct_vfetch.count(inst.pc)) {
        AuditInstTag("vertex-input");
      } else {
        EmitGfxMubuf(t, inst, sc);
      }
      break;
    case Enc::kMtbuf:
      if (sc.is_cs)
        EmitCsMtbuf(t, inst, sc);
      else if (!sc.is_ps && sc.direct_vfetch.count(inst.pc))
        AuditInstTag("vertex-input");
      else
        EmitGfxMtbuf(t, inst, sc);
      break;
    case Enc::kDs:
      if (sc.is_cs && sc.gds_var && ((w >> 17) & 1)) {
        EmitGdsCounter(t, inst, sc);
      } else if (sc.is_cs || inst.opcode == 0x35) {
        EmitDs(t, inst, sc);
      } else if (sc.lds_var && DsGraphicsSupported(inst.opcode)) {
        if (!sc.ds_own_lane.count(inst.pc))
          NoteApproximated("ds.private", inst.opcode);
        EmitDs(t, inst, sc);
      } else {
        WarnUnsupported("ds.graphics", inst.opcode, w, w1);
      }
      break;
    case Enc::kMimg:
      if (sc.is_cs)
        EmitCsMimg(t, inst, sc);
      else if (sc.is_ps)
        EmitMimg(t, inst, sc);
      else if (sc.mimg_plan && MimgNamesItsLod(inst.opcode))
        EmitMimg(t, inst, sc);
      else
        WarnUnsupported("mimg.vs", inst.opcode, w, w1);
      break;
    case Enc::kExp: {
      if (sc.is_cs) {
        WarnUnsupported("exp.cs", (w >> 4) & 0x3F, w, w1);
        sc.cs_unsupported = true;
        break;
      }
      const u32 en = w & 0xF, target = (w >> 4) & 0x3F;
      const u32 compr = (w >> 10) & 1;
      const u32 v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF,
                        (w1 >> 24) & 0xFF};
      if (sc.is_ps) {
        const Id export_block = t.m.NewBlock(), after_export = t.m.NewBlock();
        const Id active = t.LaneActive(t.Exec());
        if ((w & (1u << 12)) && sc.color_written_var)
          t.m.Store(sc.color_written_var,
                    t.SelectB(active, t.U32(1), t.U32(0)));
        t.m.SelectionMerge(after_export);
        t.m.BranchConditional(active, export_block, after_export);
        t.m.OpenBlock(export_block);
        if (target <= 7 && en && !((sc.mrt_bound_mask >> target) & 1u)) {
          sc.wrote_color = true;
          if (sc.color_written_var)
            t.m.Store(sc.color_written_var, t.U32(1));
        } else if (target <= 7 && en) {
          sc.wrote_color = true;
          const bool int_target = ((sc.mrt_uint_mask >> target) & 1u) != 0;
          Id c[4];
          bool live[4];
          ExportColor(t, sc, target, en, compr, v, int_target, c, live);
          const Id col = t.m.CompositeConstruct(
              int_target ? t.m.TypeVec(t.t_u, 4) : t.t_v4,
              {c[0], c[1], c[2], c[3]});
          const Id out_var = PsColorOut(t, sc, target);
          const Id comp_ty = int_target ? t.t_u : t.t_f;
          const Id comp_ptr_ty =
              t.m.TypePointer(spv::StorageClass::Output, comp_ty);
          const bool all_channels = live[0] && live[1] && live[2] && live[3];
          Id col2 = col;
          if (kProbeAlpha && !int_target) {
            const Id a = t.m.CompositeExtract(t.t_f, col, 3);
            col2 = t.m.CompositeConstruct(
                t.t_v4, {a, a, a, t.m.CompositeExtract(t.t_f, col, 3)});
          }
          const Id col_out = col2;
          if (all_channels) {
            t.m.Store(out_var, col_out);
          } else {
            for (u32 i = 0; i < 4; i++) {
              if (!live[i])
                continue;
              t.m.Store(t.m.AccessChain(comp_ptr_ty, out_var, {t.U32(i)}),
                        t.m.CompositeExtract(comp_ty, col_out, i));
            }
          }
          if (sc.color_written_var)
            t.m.Store(sc.color_written_var, t.U32(1));
        } else if (target == 8 && (en & 1)) {
          const Id depth =
              compr ? t.m.CompositeExtract(
                          t.t_f,
                          t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                                      {t.Vg(v[0])}),
                          0)
                    : t.VgF(v[0]);
          t.m.Store(PsDepthOut(t, sc), depth);
        }
        t.m.Branch(after_export);
        t.m.OpenBlock(after_export);
      } else {
        if (target == 12) {
          Id c[4];
          for (int i = 0; i < 4; i++)
            c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
          t.m.Store(sc.pos_out,
                    t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
        } else if (target >= 32 && target <= 63) {
          const u32 p = target - 32;
          if (p + 1 > sc.max_param)
            sc.max_param = p + 1;
          const Id out_var = VsParamOut(t, sc, p);
          Id c[4];
          for (int i = 0; i < 4; i++)
            c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(0.f);
          t.m.Store(out_var,
                    t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
        }
      }
      break;
    }
    case Enc::kFlat:
      WarnUnsupported("flat", inst.opcode, w, w1);
      break;
    case Enc::kUnknown:
      WarnUnsupported("encoding.unknown", 0, w, w1);
      break;
    default:
      break;
  }
}

void CapturePsVgprs(Translator& t, const Inst& inst) {
  if (!t.probe_var || inst.pc != kGpuPsvgprAt.get())
    return;
  const u32 first = static_cast<u32>(kGpuPsvgpr - 1);
  t.m.Store(t.probe_var,
            t.m.CompositeConstruct(t.t_v4, {t.VgF(first), t.VgF(first + 1),
                                            t.VgF(first + 2), t.F32(1.f)}));
}

bool SortedContains(const base::Vector<u32>& sorted, u32 value) {
  const u32* at = base::LowerBound(sorted.begin(), sorted.end(), value);
  return at != sorted.end() && *at == value;
}

bool NeedsExecGuard(const Translator& t,
                    const Inst& inst,
                    const StageContext& sc) {
  if (!sc.is_cs || t.rdna_sources)
    return false;
  switch (inst.enc) {
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kMimg:
      return true;
    case Enc::kDs:
      return !((inst.raw[0] >> 17) & 1) && inst.opcode != 0x35 &&
             inst.opcode != 178 && inst.opcode != 179;
    default:
      return false;
  }
}

void EmitInstGuarded(Translator& t, const Inst& inst, StageContext& sc) {
  if (!NeedsExecGuard(t, inst, sc)) {
    EmitInst(t, inst, sc);
    return;
  }
  const Id active = t.LaneActive(t.Exec());
  const Id body = t.m.NewBlock(), merge = t.m.NewBlock();
  t.m.SelectionMerge(merge);
  t.m.BranchConditional(active, body, merge);
  t.m.OpenBlock(body);
  EmitInst(t, inst, sc);
  t.m.Branch(merge);
  t.m.OpenBlock(merge);
}

// DELTA_GPU_GCN_CSVGPR_*: in the compute shader at _CS, remember the four
// registers starting at DELTA_GPU_GCN_CSVGPR (v<n>, or s<n - 1000>) after the
// instruction at _PC in every lane that has EXEC set, and make every later
// image_store write those values instead of its data. The output image then
// holds the register at the lane's own store address, which
// tools/gcn_interp.py --probe reproduces on the CPU for a lane-exact diff.
void ProbeCsStore(Translator& t, const Inst& inst, StageContext& sc) {
  if (!sc.cs_probe_var || inst.enc != Enc::kMimg || inst.opcode != 0x08)
    return;
  const u32 vdata = (inst.raw[1] >> 8) & 0xFF;
  const u32 n = __builtin_popcount((inst.raw[0] >> 8) & 0xF);
  for (u32 i = 0; i < n; i++)
    t.SetVg(vdata + i,
            t.m.Load(t.t_u, t.m.AccessChain(t.p_priv_u, sc.cs_probe_var,
                                            {t.U32(i)})));
}

void ProbeCsCapture(Translator& t, const Inst& inst, StageContext& sc) {
  if (!sc.cs_probe_var || inst.pc != kGcnCsVgprPc)
    return;
  const Id active = t.LaneActive(t.Exec());
  for (u32 i = 0; i < 4; i++) {
    const Id slot = t.m.AccessChain(t.p_priv_u, sc.cs_probe_var, {t.U32(i)});
    const Id value = kGcnCsVgpr >= 1000 ? t.Sg(kGcnCsVgpr - 1000 + i)
                                        : t.Vg(kGcnCsVgpr + i);
    t.StorePrivate(slot, t.SelectB(active, value, t.m.Load(t.t_u, slot)));
  }
}

void EmitInstAudited(Translator& t,
                     const Inst& inst,
                     u32 index,
                     StageContext& sc) {
  t.uniform_here = index < sc.uniform_points.size() && sc.uniform_points[index];
  t.readfirstlane_uniform = index < sc.uniform_readfirstlane.size() &&
                            sc.uniform_readfirstlane[index];
  if (!sc.lds_barrier_at.empty() && SortedContains(sc.lds_barrier_at, index))
    t.Barrier();
  ProbeCsStore(t, inst, sc);
  if (!ShaderDebugEnabled()) {
    EmitInstGuarded(t, inst, sc);
    CapturePsVgprs(t, inst);
    ProbeCsCapture(t, inst, sc);
    return;
  }
  AuditInstBegin(index, inst.pc);
  if (ShaderDumpEnabled()) {
    if (!t.dbg_file)
      t.dbg_file = t.m.String("gcn");
    t.m.Line(t.dbg_file, inst.pc);
  }
  const size_t before = t.m.BodyWords();
  EmitInstGuarded(t, inst, sc);
  CapturePsVgprs(t, inst);
  ProbeCsCapture(t, inst, sc);
  AuditInstEnd(index, static_cast<u32>(t.m.BodyWords() - before));
}

int BranchKind(const Inst& inst) {
  if (inst.enc != Enc::kSopp)
    return 0;
  switch (inst.opcode) {
    case 0x01:
      return 8;
    case 0x02:
      return 1;
    case 0x04:
      return 2;
    case 0x05:
      return 3;
    case 0x06:
      return 4;
    case 0x07:
      return 5;
    case 0x08:
      return 6;
    case 0x09:
      return 7;
    default:
      return 0;
  }
}

bool HasControlFlow(const Program& program) {
  return base::AnyOf(program.begin(), program.end(), [](const Inst& inst) {
    const int k = BranchKind(inst);
    return k >= 1 && k <= 7;
  });
}

Id BranchTaken(Translator& t, int kind) {
  switch (kind) {
    case 2:
      return t.IsZero(t.Scc());
    case 3:
      return t.IsNonZero(t.Scc());
    case 4:
      return t.IsZero(t.Sg(106));
    case 5:
      return t.IsNonZero(t.Sg(106));
    case 6:
      return t.IsZero(t.Exec());
    case 7:
      return t.IsNonZero(t.Exec());
    default:
      return t.m.ConstBool(false);
  }
}

Id BranchRaw(Translator& t, int kind) {
  switch (kind) {
    case 2:
    case 3:
      return t.SelectB(t.IsNonZero(t.Scc()), t.U32(1), t.U32(0));
    case 4:
    case 5:
      return t.SelectB(t.IsNonZero(t.Sg(106)), t.U32(1), t.U32(0));
    case 6:
    case 7:
      return t.SelectB(t.IsNonZero(t.Exec()), t.U32(1), t.U32(0));
    default:
      return t.U32(0);
  }
}

u32 BranchInverts(int kind) {
  return (kind == 2 || kind == 4 || kind == 6) ? 1u : 0u;
}

base::Vector<u32> BlockStarts(const Program& program, u32 max_pc) {
  base::Vector<u32> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0)
      continue;
    leaders.push_back(inst.pc + inst.size);
    if (k >= 1 && k <= 7) {
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<u32>(static_cast<i32>(inst.pc) +
                                         static_cast<i32>(inst.size) + simm));
    }
  }
  base::Sort(leaders.begin(), leaders.end());
  leaders.erase(base::Unique(leaders.begin(), leaders.end()), leaders.end());
  base::Vector<u32> starts;
  for (u32 l : leaders)
    if (l < max_pc)
      starts.push_back(l);
  return starts;
}

bool IsLdsReturningAtomic(u32 op) {
  return (op >= 0x20 && op <= 0x34) || (op >= 0x60 && op <= 0x73);
}

bool IsLdsRead(const Inst& inst) {
  if (inst.enc != Enc::kDs)
    return false;
  const u32 op = inst.opcode;
  return IsLdsReturningAtomic(op) || (op >= 0x36 && op <= 0x3c) ||
         (op >= 0x76 && op <= 0x78);
}

bool IsLdsWrite(const Inst& inst) {
  if (inst.enc != Enc::kDs)
    return false;
  const u32 op = inst.opcode;
  return IsLdsReturningAtomic(op) || op <= 0x13 || op == 0x1e || op == 0x1f ||
         (op >= 0x40 && op <= 0x53);
}

enum class VccState : u8 {
  kVaries,
  kUniform,
  kExecOrZero,
};

bool SourceUniform(u32 field, const bool* vgpr_uniform) {
  return field < 256 || vgpr_uniform[field - 256];
}

void VgprWrites(const Inst& in, u32& first, u32& count) {
  const u32 w = in.raw[0], w1 = in.raw[1];
  first = 0;
  count = 0;
  switch (in.enc) {
    case Enc::kVintrp:
      first = (w >> 18) & 0xFF;
      count = 1;
      break;
    case Enc::kDs: {
      const u32 op = in.opcode;
      first = (w1 >> 24) & 0xFF;
      if (op == 54 || op == 0x35 || op == 0x20)
        count = 1;
      else if (op == 55 || op == 56 || op == 118)
        count = 2;
      else if (op == 119)
        count = 4;
      break;
    }
    case Enc::kMubuf: {
      const u32 op = (w >> 18) & 0x7F;
      first = (w1 >> 8) & 0xFF;
      if (op <= 0x03)
        count = op + 1;
      else if (op >= 0x08 && op <= 0x0c)
        count = 1;
      else if (op == 0x0d)
        count = 2;
      else if (op == 0x0e)
        count = 4;
      else if (op == 0x0f)
        count = 3;
      else if (((op >= 0x30 && op <= 0x3f) || (op >= 0x50 && op <= 0x5f)) &&
               ((w >> 14) & 1))
        count = 1;
      break;
    }
    case Enc::kMtbuf: {
      const u32 op = (w >> 16) & 0x7;
      first = (w1 >> 8) & 0xFF;
      if (op <= 0x03)
        count = op + 1;
      break;
    }
    case Enc::kMimg: {
      const u32 op = (w >> 18) & 0x7F;
      first = (w1 >> 8) & 0xFF;
      if (op == 0x08 || op == 0x09)
        break;
      const u32 dmask = (w >> 8) & 0xF;
      for (u32 b = 0; b < 4; b++)
        count += (dmask >> b) & 1;
      break;
    }
    case Enc::kFlat: {
      const u32 op = (w >> 18) & 0x7F;
      first = (w1 >> 24) & 0xFF;
      if (op >= 0x10 && op <= 0x17)
        count = op == 0x15 ? 2 : op == 0x16 ? 4 : op == 0x17 ? 3 : 1;
      break;
    }
    default:
      break;
  }
}

base::Vector<u8> ProvenUniformReadFirstLane(const Program& program,
                                            const u8* reachable) {
  base::Vector<u8> proven(program.size(), 0);
  bool uni[256];
  for (u32 i = 0; i < 256; i++)
    uni[i] = true;
  uni[0] = uni[1] = uni[2] = false;

  for (bool changed = true; changed;) {
    changed = false;
    VccState vcc = VccState::kVaries;
    u32 block_pc = 0;
    for (u32 i = 0; i < program.size(); i++) {
      const Inst& in = program[i];
      if (reachable && !reachable[i])
        continue;
      if (BranchKind(in) != 0) {
        vcc = VccState::kVaries;
        block_pc = in.pc;
        continue;
      }
      (void)block_pc;
      const u32 w = in.raw[0];
      const auto clear = [&](u32 vdst) {
        if (vdst < 256 && uni[vdst]) {
          uni[vdst] = false;
          changed = true;
        }
      };
      const auto set_from = [&](u32 vdst, bool u) {
        if (!u)
          clear(vdst);
      };

      switch (in.enc) {
        case Enc::kSop1: {
          const u32 sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
          if (in.opcode == 0x04 && sdst == 106 && ssrc0 == 126)
            vcc = VccState::kExecOrZero;
          else if (sdst == 106 || sdst == 107)
            vcc = VccState::kVaries;
          break;
        }
        case Enc::kSop2: {
          const u32 sdst = (w >> 16) & 0x7F;
          const u32 a = w & 0xFF, b = (w >> 8) & 0xFF;
          if (in.opcode == 0x0b && sdst == 106 && a == 126 && b == 128)
            vcc = VccState::kExecOrZero;
          else if (sdst == 106 || sdst == 107)
            vcc = VccState::kVaries;
          break;
        }
        case Enc::kVop1: {
          const u32 vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
          if (in.opcode == 0x02)
            break;
          set_from(vdst, SourceUniform(src0, uni));
          break;
        }
        case Enc::kVop2: {
          const u32 op = in.opcode, vdst = (w >> 17) & 0xFF;
          const u32 src0 = w & 0x1FF, vsrc1 = 256 + ((w >> 9) & 0xFF);
          if (op == 0x01) {
            break;
          }
          const bool src_u =
              SourceUniform(src0, uni) && SourceUniform(vsrc1, uni);
          switch (op) {
            case 0x00:
              set_from(vdst, src_u && vcc != VccState::kVaries);
              break;
            case 0x02:
              clear(vdst);
              break;
            case 0x23:
            case 0x24:
              clear(vdst);
              break;
            case 0x28:
            case 0x29:
            case 0x2a:
              set_from(vdst, src_u && vcc == VccState::kUniform);
              vcc = src_u && vcc == VccState::kUniform ? VccState::kUniform
                                                       : VccState::kVaries;
              break;
            case 0x25:
            case 0x26:
            case 0x27:
              set_from(vdst, src_u);
              vcc = src_u ? VccState::kUniform : VccState::kVaries;
              break;
            default:
              set_from(vdst, src_u);
              break;
          }
          break;
        }
        case Enc::kVopc: {
          const u32 src0 = w & 0x1FF, vsrc1 = 256 + ((w >> 9) & 0xFF);
          vcc = SourceUniform(src0, uni) && SourceUniform(vsrc1, uni)
                    ? VccState::kUniform
                    : VccState::kVaries;
          break;
        }
        case Enc::kVop3: {
          const u32 vdst = w & 0xFF, w1 = in.raw[1];
          const u32 s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF,
                    s2 = (w1 >> 18) & 0x1FF;
          const bool src_u = SourceUniform(s0, uni) && SourceUniform(s1, uni) &&
                             SourceUniform(s2, uni);
          set_from(vdst, src_u);
          break;
        }
        case Enc::kVintrp:
        case Enc::kDs:
        case Enc::kMubuf:
        case Enc::kMtbuf:
        case Enc::kMimg:
        case Enc::kFlat: {
          u32 first = 0, count = 0;
          VgprWrites(in, first, count);
          for (u32 d = 0; d < count; d++)
            clear(first + d);
          break;
        }
        default:
          break;
      }
    }
  }

  VccState vcc = VccState::kVaries;
  for (u32 i = 0; i < program.size(); i++) {
    const Inst& in = program[i];
    if (reachable && !reachable[i])
      continue;
    if (BranchKind(in) != 0) {
      vcc = VccState::kVaries;
      continue;
    }
    if (in.enc == Enc::kVop1 && in.opcode == 0x02) {
      const u32 src0 = in.raw[0] & 0x1FF;
      proven[i] = SourceUniform(src0, uni) ? 1 : 0;
    }
  }
  return proven;
}

}  // namespace

base::Vector<u8> UniformPoints(const Program& program) {
  if (!HasControlFlow(program))
    return base::Vector<u8>(program.size(), 1);
  const u32 max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const base::Vector<u32> starts = BlockStarts(program, max_pc);
  const u32 first_end = starts.size() > 1 ? starts[1] : max_pc;
  base::Vector<u8> at(program.size(), 0);
  for (u32 i = 0; i < program.size(); i++)
    at[i] = program[i].pc < first_end;
  return at;
}

namespace {

struct WaveAccess {
  bool read = false, write = false;
};

WaveAccess LdsAccess(const Inst& inst) {
  return {IsLdsRead(inst), IsLdsWrite(inst)};
}

WaveAccess MemoryAccess(const Inst& inst) {
  const u32 w = inst.raw[0];
  switch (inst.enc) {
    case Enc::kMubuf: {
      const u32 op = (w >> 18) & 0x7F;
      const bool load = op <= 0x03 || (op >= 0x08 && op <= 0x0f);
      const bool store =
          (op >= 0x04 && op <= 0x07) || (op >= 0x18 && op <= 0x1f);
      return {!store, !load};
    }
    case Enc::kMtbuf: {
      const u32 op = (w >> 16) & 0x7;
      return {op < 4, op >= 4};
    }
    case Enc::kMimg: {
      const u32 op = (w >> 18) & 0x7F;
      if (op == 0x0e)
        return {};
      const bool store = op == 0x08 || op == 0x09;
      return {!store, store};
    }
    default:
      return {};
  }
}

}  // namespace

LdsBarrierPlan PlanLdsBarriers(const Program& program,
                               const u8* reachable,
                               u32 threads_per_group,
                               const base::HashMap<u32, u32>* bindings,
                               u64 shared_mask) {
  LdsBarrierPlan plan;
  if (!WaveSplitsAcrossSubgroups() || threads_per_group > kGcnWave ||
      threads_per_group <= HostSubgroupSize())
    return plan;
  const auto live = [&](u32 i) { return !reachable || reachable[i]; };
  for (u32 i = 0; i < program.size(); i++)
    if (live(i) && program[i].enc == Enc::kSopp && program[i].opcode == 0x0a)
      return plan;

  const auto shared = [&](const Inst& inst) {
    if (!bindings)
      return WaveAccess{};
    const auto it = bindings->find(inst.pc);
    if (it == bindings->end() || it->second >= 64 ||
        !((shared_mask >> it->second) & 1))
      return WaveAccess{};
    return MemoryAccess(inst);
  };
  const bool branchy = HasControlFlow(program);
  base::Vector<u8> at(program.size(), 0);
  for (u32 domain = 0; domain < 2; domain++) {
    const auto access = [&](const Inst& inst) {
      return domain == 0 ? LdsAccess(inst) : shared(inst);
    };
    bool any_read = false, any_write = false;
    for (u32 i = 0; i < program.size(); i++) {
      if (!live(i))
        continue;
      const WaveAccess a = access(program[i]);
      any_read |= a.read;
      any_write |= a.write;
    }
    if (!any_read || !any_write)
      continue;
    u32 last = 0;
    for (u32 i = 0; i < program.size(); i++) {
      if (!live(i))
        continue;
      const WaveAccess a = access(program[i]);
      if (!a.read && !a.write)
        continue;
      if ((a.read && (last & 2)) || (a.write && (last & 1))) {
        plan.lockstep = branchy;
        plan.orders_memory |= domain == 1;
        at[i] = 1;
      }
      last = (a.read ? 1 : 0) | (a.write ? 2 : 0);
    }
  }
  for (u32 i = 0; i < program.size(); i++)
    if (at[i])
      plan.at.push_back(i);
  return plan;
}

namespace {

DELTA_OPTION(bool, kStructuredCfg, "DELTA_GPU_STRUCTURED_CFG", true);

// Lowers a branchy program to structured SPIR-V: if/else at each conditional
// branch, merging at its immediate post-dominator, and natural loops with
// break and continue. Drivers compile this several times faster than the
// dispatch loop, which carries every register around one loop. Plan() walks
// exactly what Emit() will, so a graph that does not fit (irreducible flow, a
// loop with more than one exit or an exit to the end) is refused before
// anything is emitted. A block two paths reach before their merge is emitted
// once per path, within a size budget.
class Structurizer {
 public:
  Structurizer(Translator& t,
               const Program& program,
               StageContext& sc,
               const u8* reachable)
      : t_(t), program_(program), sc_(sc), reachable_(reachable) {}

  bool Plan();
  void Emit();

 private:
  static constexpr u32 kNone = ~0u;
  struct Block {
    u32 lo = 0, hi = 0;   // pc range
    int kind = 0;         // BranchKind of the terminator, 0 when it falls
    u32 succ[2] = {kNone, kNone};  // taken, fallthrough
    u32 nsucc = 0;
    u32 loop = kNone;     // innermost loop header
    bool header = false;
    u32 exit = kNone;     // loop exit, for headers
    u32 size = 0;         // instructions
  };
  struct Region {
    base::Vector<base::Vector<u64>> pd;  // post-dominator sets per node
  };
  struct Loop {
    u32 header;
    Id cont, merge;
  };

  u32 Exit() const { return n_; }
  u32 Cont() const { return n_ + 1; }
  u32 Brk() const { return n_ + 2; }
  u32 Sink() const { return n_ + 3; }
  bool InBody(u32 header, u32 b) const;
  bool RegionSuccs(u32 b, u32 g, u32* out, u32* count) const;
  bool BuildRegion(u32 g);
  u32 IPostDom(u32 b, u32 g);
  bool Walk(u32 b, u32 stop, u32 g, bool header_first);
  bool Seq(u32 b, u32 stop, u32 g, bool header_first);
  bool Select(u32 b, u32 g);
  void EmitLoop(u32 h);
  void EmitCode(u32 b);
  Id Taken(u32 b);
  void Open(Id label) {
    t_.m.OpenBlock(label);
    open_ = true;
  }
  void Jump(Id label) {
    t_.m.Branch(label);
    open_ = false;
  }

  Translator& t_;
  const Program& program_;
  StageContext& sc_;
  const u8* reachable_;
  u32 n_ = 0;
  base::Vector<Block> blocks_;
  base::Vector<u32> idom_;
  base::Vector<base::Vector<u64>> body_;  // per header, bitset of blocks
  base::HashMap<u32, Region> regions_;
  base::Vector<u8> visited_;
  base::Vector<Loop> loops_;  // emission stack
  bool emit_ = false, lockstep_ = false, fresh_barrier_ = false;
  bool open_ = true;  // the current SPIR-V block still needs a terminator
  u64 emitted_ = 0;
  Id iter_var_ = 0;
};

bool Structurizer::InBody(u32 h, u32 b) const {
  return b < n_ && (body_[h][b >> 6] >> (b & 63)) & 1;
}

bool Structurizer::Plan() {
  const u32 max_pc =
      program_.empty() ? 0 : program_.back().pc + program_.back().size;
  const base::Vector<u32> starts = BlockStarts(program_, max_pc);
  n_ = static_cast<u32>(starts.size());
  if (n_ == 0 || n_ > 4096)
    return false;
  const auto block_of = [&](u32 pc) -> u32 {
    if (pc >= max_pc)
      return n_;
    u32 b = 0;
    for (u32 i = 0; i < n_; i++)
      if (starts[i] <= pc)
        b = i;
      else
        break;
    return b;
  };
  blocks_.resize(n_);
  for (u32 b = 0; b < n_; b++) {
    Block& blk = blocks_[b];
    blk.lo = starts[b];
    blk.hi = b + 1 < n_ ? starts[b + 1] : max_pc;
    const u32 fall = b + 1 < n_ ? b + 1 : n_;
    u32 idx = 0;
    for (const Inst& inst : program_) {
      const u32 i = idx++;
      if (inst.pc < blk.lo || inst.pc >= blk.hi ||
          (reachable_ && !reachable_[i]))
        continue;
      blk.size++;
      const int k = BranchKind(inst);
      if (!k)
        continue;
      blk.kind = k;
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      const u32 target = block_of(static_cast<u32>(
          static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm));
      if (k == 8) {
        blk.succ[0] = n_;
      } else if (k == 1) {
        blk.succ[0] = target;
      } else {
        blk.succ[0] = target;
        blk.succ[1] = fall;
      }
      break;
    }
    if (!blk.kind)
      blk.succ[0] = fall;
    blk.nsucc = blk.succ[1] == kNone || blk.succ[1] == blk.succ[0] ? 1 : 2;
  }

  // Dominators over the blocks reachable from the entry.
  base::Vector<u32> order, rpo(n_ + 1, kNone), stack, cursor(n_ + 1, 0);
  base::Vector<u8> seen(n_ + 1, 0);
  stack.push_back(0);
  seen[0] = 1;
  while (!stack.empty()) {
    const u32 b = stack.back();
    if (b < n_ && cursor[b] < blocks_[b].nsucc) {
      const u32 s = blocks_[b].succ[cursor[b]++];
      if (!seen[s]) {
        seen[s] = 1;
        stack.push_back(s);
      }
      continue;
    }
    order.push_back(b);
    stack.pop_back();
  }
  for (u32 i = 0; i < order.size(); i++)
    rpo[order[i]] = static_cast<u32>(order.size() - 1 - i);
  base::Vector<base::Vector<u32>> preds(n_ + 1);
  for (u32 b = 0; b < n_; b++)
    if (seen[b])
      for (u32 k = 0; k < blocks_[b].nsucc; k++)
        preds[blocks_[b].succ[k]].push_back(b);
  idom_.assign(n_ + 1, kNone);
  idom_[0] = 0;
  for (bool changed = true; changed;) {
    changed = false;
    for (u32 i = order.size(); i-- > 0;) {
      const u32 b = order[i];
      if (b == 0)
        continue;
      u32 d = kNone;
      for (const u32 p : preds[b]) {
        if (idom_[p] == kNone)
          continue;
        if (d == kNone) {
          d = p;
          continue;
        }
        u32 x = p, y = d;
        while (x != y) {
          while (rpo[x] > rpo[y])
            x = idom_[x];
          while (rpo[y] > rpo[x])
            y = idom_[y];
        }
        d = x;
      }
      if (idom_[b] != d) {
        idom_[b] = d;
        changed = true;
      }
    }
  }
  const auto dominates = [&](u32 a, u32 b) {
    for (;;) {
      if (a == b)
        return true;
      if (b == 0 || idom_[b] == kNone)
        return false;
      b = idom_[b];
    }
  };

  // Natural loops; any other edge against the reverse postorder is
  // irreducible.
  const u32 words = (n_ + 63) / 64;
  body_.resize(n_);
  for (u32 b = 0; b < n_; b++) {
    if (!seen[b])
      continue;
    for (u32 k = 0; k < blocks_[b].nsucc; k++) {
      const u32 h = blocks_[b].succ[k];
      if (h == n_ || rpo[h] > rpo[b])
        continue;
      if (!dominates(h, b))
        return false;
      if (body_[h].empty())
        body_[h].assign(words, 0);
      blocks_[h].header = true;
      stack.clear();
      stack.push_back(b);
      body_[h][h >> 6] |= 1ull << (h & 63);
      while (!stack.empty()) {
        const u32 x = stack.back();
        stack.pop_back();
        if (InBody(h, x))
          continue;
        body_[h][x >> 6] |= 1ull << (x & 63);
        for (const u32 p : preds[x])
          stack.push_back(p);
      }
    }
  }
  // Innermost loop per block: the header with the smallest body.
  base::Vector<u32> size(n_, 0);
  for (u32 h = 0; h < n_; h++)
    if (blocks_[h].header)
      for (u32 b = 0; b < n_; b++)
        size[h] += InBody(h, b);
  for (u32 b = 0; b < n_; b++)
    for (u32 h = 0; h < n_; h++)
      if (blocks_[h].header && InBody(h, b) &&
          (blocks_[b].loop == kNone || size[h] < size[blocks_[b].loop]))
        blocks_[b].loop = h;
  for (u32 h = 0; h < n_; h++) {
    if (!blocks_[h].header)
      continue;
    for (u32 b = 0; b < n_; b++) {
      if (!InBody(h, b))
        continue;
      for (u32 k = 0; k < blocks_[b].nsucc; k++) {
        const u32 s = blocks_[b].succ[k];
        if (InBody(h, s))
          continue;
        if (s == n_ || (blocks_[h].exit != kNone && blocks_[h].exit != s))
          return false;
        blocks_[h].exit = s;
      }
    }
    if (blocks_[h].exit == kNone)
      return false;
  }

  visited_.assign(n_, 0);
  emit_ = false;
  if (!Seq(0, Exit(), kNone, false))
    return false;
  for (u32 b = 0; b < n_; b++)
    if (seen[b] && !visited_[b])
      return false;
  return true;
}

// The successors of b in loop g's region (kNone: the top level), with inner
// loops collapsed to their exit and edges leaving g mapped to continue/break.
bool Structurizer::RegionSuccs(u32 b, u32 g, u32* out, u32* count) const {
  u32 raw[2];
  u32 n = 0;
  if (b < n_ && blocks_[b].header && b != g) {
    raw[n++] = blocks_[b].exit;
  } else if (b < n_) {
    for (u32 k = 0; k < blocks_[b].nsucc; k++)
      raw[n++] = blocks_[b].succ[k];
  }
  *count = 0;
  for (u32 k = 0; k < n; k++) {
    u32 s = raw[k];
    if (g != kNone) {
      if (s == g)
        s = Cont();
      else if (s == n_ || !InBody(g, s)) {
        if (s != blocks_[g].exit)
          return false;
        s = Brk();
      }
    }
    out[(*count)++] = s;
  }
  return true;
}

bool Structurizer::BuildRegion(u32 g) {
  if (regions_.find(g) != regions_.end())
    return true;
  const u32 m = n_ + 4, words = (m + 63) / 64;
  Region region;
  region.pd.resize(m);
  // Nodes reachable from the region entry.
  base::Vector<u32> nodes, stack{g == kNone ? 0u : g};
  base::Vector<u8> seen(m, 0);
  seen[stack[0]] = 1;
  while (!stack.empty()) {
    const u32 x = stack.back();
    stack.pop_back();
    nodes.push_back(x);
    u32 succ[2], count = 0;
    if (x < n_) {
      if (!RegionSuccs(x, g, succ, &count))
        return false;
    } else if (x != Sink()) {
      succ[0] = Sink();
      count = 1;
    }
    for (u32 k = 0; k < count; k++)
      if (!seen[succ[k]]) {
        seen[succ[k]] = 1;
        stack.push_back(succ[k]);
      }
  }
  if (!seen[Sink()])
    return false;
  for (const u32 x : nodes) {
    region.pd[x].assign(words, ~0ull);
  }
  region.pd[Sink()].assign(words, 0);
  region.pd[Sink()][Sink() >> 6] |= 1ull << (Sink() & 63);
  for (bool changed = true; changed;) {
    changed = false;
    for (u32 i = nodes.size(); i-- > 0;) {
      const u32 x = nodes[i];
      if (x == Sink())
        continue;
      u32 succ[2], count = 0;
      if (x < n_)
        RegionSuccs(x, g, succ, &count);
      else {
        succ[0] = Sink();
        count = 1;
      }
      base::Vector<u64> next(words, ~0ull);
      for (u32 k = 0; k < count; k++)
        for (u32 w = 0; w < words; w++)
          next[w] &= region.pd[succ[k]][w];
      next[x >> 6] |= 1ull << (x & 63);
      bool same = true;
      for (u32 w = 0; w < words; w++)
        same = same && next[w] == region.pd[x][w];
      if (!same) {
        region.pd[x] = base::move(next);
        changed = true;
      }
    }
  }
  regions_.emplace(g, base::move(region));
  return true;
}

u32 Structurizer::IPostDom(u32 b, u32 g) {
  const Region& r = regions_.find(g)->second;
  const base::Vector<u64>& set = r.pd[b];
  u32 best = kNone, best_n = 0;
  for (u32 w = 0; w < set.size(); w++)
    for (u64 bits = set[w]; bits; bits &= bits - 1) {
      const u32 c = w * 64 + static_cast<u32>(__builtin_ctzll(bits));
      if (c == b)
        continue;
      u32 n = 0;
      for (const u64 x : r.pd[c])
        n += static_cast<u32>(__builtin_popcountll(x));
      if (best == kNone || n > best_n) {
        best = c;
        best_n = n;
      }
    }
  return best;
}

Id Structurizer::Taken(u32 b) {
  const int kind = blocks_[b].kind;
  if (!lockstep_)
    return BranchTaken(t_, kind);
  // Lock-step: the whole group takes the branch when any lane does.
  const Id raw = BranchRaw(t_, kind);
  const Id vote = t_.m.Emit(spv::Op::OpGroupNonUniformAny, t_.t_bool,
                            {t_.U32(3), t_.IsNonZero(raw)});
  const Id elected =
      t_.m.Emit(spv::Op::OpGroupNonUniformElect, t_.t_bool, {t_.U32(3)});
  const Id publish = t_.m.NewBlock(), published = t_.m.NewBlock();
  t_.m.SelectionMerge(published);
  t_.m.BranchConditional(elected, publish, published);
  Open(publish);
  t_.m.Store(t_.XchgAt(t_.m.Load(t_.t_u, sc_.subgroup_id)),
             t_.SelectB(vote, t_.U32(1), t_.U32(0)));
  Jump(published);
  Open(published);
  t_.Barrier();
  Id bits = t_.U32(0);
  const u32 subgroups =
      (t_.xchg_lanes + HostSubgroupSize() - 1) / HostSubgroupSize();
  for (u32 group = 0; group < subgroups; group++)
    bits = t_.Or(bits, t_.m.Load(t_.t_u, t_.XchgAt(t_.U32(group))));
  t_.Barrier();
  fresh_barrier_ = true;
  return t_.m.Emit(spv::Op::OpLogicalNotEqual, t_.t_bool,
                   {t_.IsNonZero(bits), t_.m.ConstBool(BranchInverts(kind))});
}

void Structurizer::EmitCode(u32 b) {
  // Lock-step keeps the dispatch loop's guarantee that LDS accesses in
  // different blocks have a barrier between them.
  if (lockstep_ && !fresh_barrier_ && (b != 0 || blocks_[0].header))
    t_.Barrier();
  fresh_barrier_ = false;
  const Block& blk = blocks_[b];
  u32 idx = 0;
  for (const Inst& inst : program_) {
    const u32 i = idx++;
    if (inst.pc < blk.lo || inst.pc >= blk.hi ||
        (reachable_ && !reachable_[i]))
      continue;
    if (BranchKind(inst))
      break;
    EmitInstAudited(t_, inst, i, sc_);
  }
}

bool Structurizer::Select(u32 b, u32 g) {
  if (!BuildRegion(g))
    return false;
  const u32 m = IPostDom(b, g);
  if (m == kNone)
    return false;
  const Block& blk = blocks_[b];
  u32 arms[2], count = 0;
  if (!RegionSuccs(b, g, arms, &count) || count != 2)
    return false;
  Id merge = 0, labels[2] = {0, 0};
  if (emit_) {
    const Id cond = Taken(b);
    merge = t_.m.NewBlock();
    for (u32 k = 0; k < 2; k++)
      labels[k] = arms[k] == m ? merge : t_.m.NewBlock();
    t_.m.SelectionMerge(merge);
    t_.m.BranchConditional(cond, labels[0], labels[1]);
  }
  for (u32 k = 0; k < 2; k++) {
    if (arms[k] == m)
      continue;
    if (emit_)
      Open(labels[k]);
    if (!Seq(blk.succ[k], m, g, false))
      return false;
    if (emit_ && open_)
      Jump(merge);
  }
  if (emit_) {
    Open(merge);
    fresh_barrier_ = false;
    if (m == Cont()) {
      Jump(loops_.back().cont);
    } else if (m == Brk()) {
      Jump(loops_.back().merge);
    } else if (m == Sink()) {
      t_.m.Unreachable();
      open_ = false;
    }
  }
  return true;
}

// Emits from block b until control reaches `stop`; with header_first, b is
// loop g's header and is emitted rather than taken as a continue. False when
// the graph does not fit; during Emit() the result is always true.
bool Structurizer::Seq(u32 b, u32 stop, u32 g, bool header_first) {
  for (bool first = header_first;; first = false) {
    if (b == stop)
      return true;
    if (g != kNone && b == g && !first) {
      if (emit_)
        Jump(loops_.back().cont);
      return true;
    }
    if (g != kNone && (b == n_ || !InBody(g, b))) {
      if (b != blocks_[g].exit)
        return false;
      if (emit_)
        Jump(loops_.back().merge);
      return true;
    }
    if (b == n_)
      return stop == Exit() || stop == Sink();
    if (blocks_[b].header && !(b == g && first)) {
      if (!emit_) {
        if (visited_[b])
          return false;
        if (!Seq(b, Sink(), b, true))
          return false;
      } else {
        EmitLoop(b);
      }
      b = blocks_[b].exit;
      continue;
    }
    if (!emit_) {
      visited_[b] = 1;
      emitted_ += blocks_[b].size;
      if (emitted_ > 2 * program_.size() + 64)
        return false;
    } else {
      EmitCode(b);
    }
    if (blocks_[b].nsucc == 1) {
      b = blocks_[b].succ[0];
      continue;
    }
    if (!Select(b, g))
      return false;
    const u32 m = IPostDom(b, g);
    if (m >= n_ && m != Exit())
      return true;  // merge already branched to continue/break
    b = m;
  }
}

void Structurizer::EmitLoop(u32 h) {
  const Id header = t_.m.NewBlock(), body = t_.m.NewBlock();
  const Id cont = t_.m.NewBlock(), merge = t_.m.NewBlock();
  Jump(header);
  Open(header);
  t_.m.LoopMerge(merge, cont);
  Jump(body);
  Open(body);
  fresh_barrier_ = false;
  if (iter_var_) {
    // The dispatch loop's runaway guard, per loop iteration here.
    const Id it = t_.m.Load(t_.t_u, iter_var_);
    t_.m.Store(iter_var_, t_.m.Emit(spv::Op::OpIAdd, t_.t_u, {it, t_.U32(1)}));
    const Id over =
        t_.m.Emit(spv::Op::OpUGreaterThan, t_.t_bool, {it, t_.U32(kCfgMaxIter)});
    const Id stop = t_.m.NewBlock(), go = t_.m.NewBlock();
    t_.m.SelectionMerge(go);
    t_.m.BranchConditional(over, stop, go);
    Open(stop);
    Jump(merge);
    Open(go);
  }
  loops_.push_back({h, cont, merge});
  Seq(h, Sink(), h, true);
  if (open_)
    Jump(cont);
  loops_.pop_back();
  Open(cont);
  Jump(header);
  Open(merge);
  fresh_barrier_ = false;
}

void Structurizer::Emit() {
  emit_ = true;
  lockstep_ = sc_.lockstep_loop && t_.xchg_var;
  if (kCfgMaxIter)
    iter_var_ = t_.m.Variable(t_.p_priv_u, spv::StorageClass::Private,
                              t_.m.ConstNull(t_.t_u));
  if (lockstep_)
    t_.RequireSubgroup(spv::Capability::GroupNonUniformVote);
  Seq(0, Exit(), kNone, false);
}

void EmitCfg(Translator& t,
             const Program& program,
             StageContext& sc,
             const u8* reachable = nullptr) {
  if (kStructuredCfg) {
    Structurizer structured(t, program, sc, reachable);
    if (structured.Plan()) {
      structured.Emit();
      return;
    }
  }
  const u32 max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const base::Vector<u32> starts = BlockStarts(program, max_pc);
  const u32 num_blocks = static_cast<u32>(starts.size());
  const u32 kExit = num_blocks;
  const auto block_of = [&](u32 pc) -> u32 {
    if (pc >= max_pc)
      return kExit;
    u32 b = 0;
    for (u32 i = 0; i < num_blocks; i++)
      if (starts[i] <= pc)
        b = i;
      else
        break;
    return b;
  };

  const Id header = t.m.NewBlock(), dispatch = t.m.NewBlock();
  const Id merge_sel = t.m.NewBlock();
  const Id cont = t.m.NewBlock(), merge = t.m.NewBlock();
  const Id exit_blk = t.m.NewBlock();
  const Id lockstep_blk = t.m.NewBlock();
  base::Vector<Id> case_labels(num_blocks);
  for (Id& l : case_labels)
    l = t.m.NewBlock();

  const Id iter_var = kCfgMaxIter
                          ? t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                         t.m.ConstNull(t.t_u))
                          : 0;

  const bool lockstep = sc.lockstep_loop && t.xchg_var;
  const auto priv = [&]() {
    return t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                        t.m.ConstNull(t.t_u));
  };
  const Id br_raw = lockstep ? priv() : 0;
  const Id br_invert = lockstep ? priv() : 0;
  const Id br_taken = lockstep ? priv() : 0;
  const Id br_fall = lockstep ? priv() : 0;
  const auto set_next = [&](u32 taken_state, u32 fall_state, Id raw,
                            u32 invert) {
    if (!lockstep) {
      if (taken_state == fall_state) {
        t.SetState(taken_state);
      } else {
        const Id cond = invert ? t.IsZero(raw) : t.IsNonZero(raw);
        t.SetStateId(t.SelectB(cond, t.U32(taken_state), t.U32(fall_state)));
      }
      return;
    }
    t.m.Store(br_raw, raw ? raw : t.U32(0));
    t.m.Store(br_invert, t.U32(invert));
    t.m.Store(br_taken, t.U32(taken_state));
    t.m.Store(br_fall, t.U32(fall_state));
  };

  t.SetState(0);
  t.m.Branch(header);
  t.m.OpenBlock(header);
  t.m.LoopMerge(merge, cont);
  if (lockstep) {
    t.m.Branch(lockstep_blk);
    t.m.OpenBlock(lockstep_blk);
    t.RequireSubgroup(spv::Capability::GroupNonUniformVote);
    const Id vote = t.m.Emit(spv::Op::OpGroupNonUniformAny, t.t_bool,
                             {t.U32(3), t.IsNonZero(t.m.Load(t.t_u, br_raw))});
    const Id elected =
        t.m.Emit(spv::Op::OpGroupNonUniformElect, t.t_bool, {t.U32(3)});
    const Id publish = t.m.NewBlock(), published = t.m.NewBlock();
    t.m.SelectionMerge(published);
    t.m.BranchConditional(elected, publish, published);
    t.m.OpenBlock(publish);
    t.m.Store(t.XchgAt(t.m.Load(t.t_u, sc.subgroup_id)),
              t.SelectB(vote, t.U32(1), t.U32(0)));
    t.m.Branch(published);
    t.m.OpenBlock(published);
    t.Barrier();
    Id bits = t.U32(0);
    const u32 subgroups =
        (t.xchg_lanes + HostSubgroupSize() - 1) / HostSubgroupSize();
    for (u32 group = 0; group < subgroups; group++)
      bits = t.Or(bits, t.m.Load(t.t_u, t.XchgAt(t.U32(group))));
    const Id any = t.IsNonZero(bits);
    t.Barrier();
    const Id inv = t.IsNonZero(t.m.Load(t.t_u, br_invert));
    const Id taken = t.m.Emit(spv::Op::OpLogicalNotEqual, t.t_bool, {any, inv});
    t.SetStateId(
        t.SelectB(taken, t.m.Load(t.t_u, br_taken), t.m.Load(t.t_u, br_fall)));
    t.m.BranchConditional(
        t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {t.State(), t.U32(kExit)}),
        dispatch, merge);
  } else {
    t.m.Branch(dispatch);
  }
  t.m.OpenBlock(dispatch);
  Id state = t.State();
  if (kCfgMaxIter) {
    const Id it = t.m.Load(t.t_u, iter_var);
    t.m.Store(iter_var, t.m.Emit(spv::Op::OpIAdd, t.t_u, {it, t.U32(1)}));
    const Id over =
        t.m.Emit(spv::Op::OpUGreaterThan, t.t_bool, {it, t.U32(kCfgMaxIter)});
    state = t.SelectB(over, t.U32(kExit), state);
  }
  t.m.SelectionMerge(merge_sel);
  base::Vector<base::Pair<u32, Id> > cases;
  for (u32 i = 0; i < num_blocks; i++)
    cases.push_back({i, case_labels[i]});
  t.m.Switch(state, exit_blk, cases);

  for (u32 bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    const u32 blk_start = starts[bi];
    const u32 blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    u32 idx = 0;
    for (const Inst& inst : program) {
      const u32 inst_idx = idx++;
      if (inst.pc < blk_start || inst.pc >= blk_end)
        continue;
      if (reachable && !reachable[inst_idx])
        continue;
      const int k = BranchKind(inst);
      if (k == 0) {
        EmitInstAudited(t, inst, inst_idx, sc);
        continue;
      }
      const u32 fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (k == 8) {
        set_next(kExit, kExit, 0, 0);
      } else if (k == 1) {
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        const u32 target = block_of(static_cast<u32>(
            static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm));
        set_next(target, target, 0, 0);
      } else {
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        const u32 target = block_of(static_cast<u32>(
            static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm));
        set_next(target, fall, BranchRaw(t, k), BranchInverts(k));
      }
      terminated = true;
      break;
    }
    if (!terminated)
      set_next((bi + 1 < num_blocks) ? bi + 1 : kExit,
               (bi + 1 < num_blocks) ? bi + 1 : kExit, 0, 0);
    t.m.Branch(merge_sel);
  }
  t.m.OpenBlock(exit_blk);
  t.m.Branch(merge);
  t.m.OpenBlock(merge_sel);
  t.m.Branch(cont);
  t.m.OpenBlock(cont);
  t.m.Branch(header);
  t.m.OpenBlock(merge);
}

bool ForceCfg() {
  return kGpuSpirvCfg;
}

void EmitBody(Translator& t,
              const Program& program,
              StageContext& sc,
              const u8* reachable) {
  t.SeedExec();
  t.predicate_vector = !(kGpuVsNoPred && !sc.is_ps && !sc.is_cs);
  if (ForceCfg() || HasControlFlow(program)) {
    EmitCfg(t, program, sc, reachable);
    return;
  }
  u32 index = 0;
  for (const Inst& inst : program) {
    const u32 inst_idx = index++;
    if (reachable && !reachable[inst_idx])
      continue;
    if (inst.enc == Enc::kSopp && inst.opcode == 1)
      break;
    EmitInstAudited(t, inst, inst_idx, sc);
  }
}

}  // namespace

bool UsesDsSwizzle(const Program& program, const u8* reachable) {
  for (u32 i = 0; i < program.size(); i++)
    if ((!reachable || reachable[i]) && program[i].enc == Enc::kDs &&
        program[i].opcode == 0x35)
      return true;
  return false;
}

void EnableDsSwizzle(Translator& t, StageContext& sc, base::Vector<Id>& iface) {
  t.m.Capability(spv::Capability::GroupNonUniform);
  t.m.Capability(spv::Capability::GroupNonUniformShuffle);
  sc.subgroup_local_id =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                   spv::StorageClass::Input);
  t.m.Decorate(sc.subgroup_local_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::SubgroupLocalInvocationId)});
  if (sc.is_ps)
    t.m.Decorate(sc.subgroup_local_id, spv::Decoration::Flat);
  iface.push_back(sc.subgroup_local_id);
}

namespace {

constexpr u32 kPsUserDataOffset = 64;
constexpr u32 kPcBaseOffset = 128;

u32 CbufBindingLimit(const Translator& t) {
  return t.indirect_cbufs ? kIndirectCbufBindings : kMaxCbufBindings;
}

bool NeedsIndirectCbufs(const std::initializer_list<const Program*> programs) {
  u32 total = 0;
  for (const Program* program : programs) {
    if (!program || program->empty())
      continue;
    const base::Vector<u8> reachable = ComputeReachability(*program);
    base::Vector<ShaderCbuf> cbufs;
    base::HashMap<u32, u32> bindings;
    PlanCbufs(*program, 0, cbufs, bindings, reachable.data(),
              kIndirectCbufBindings);
    total += static_cast<u32>(cbufs.size());
  }
  return total > kMaxCbufBindings;
}

Id DeclareUserData(Translator& t, u32 byte_offset) {
  const Id words = t.m.TypeArray(t.t_u, 16);
  t.m.Decorate(words, spv::Decoration::ArrayStride, {4});
  Id block;
  if (PushCodeBase()) {
    block = t.m.TypeStruct({words, t.t_u, t.t_u});
    const u32 base = kPcBaseOffset + (byte_offset ? 8u : 0u);
    t.m.MemberDecorate(block, 1, spv::Decoration::Offset, {base});
    t.m.MemberDecorate(block, 2, spv::Decoration::Offset, {base + 4});
  } else {
    block = t.m.TypeStruct({words});
  }
  t.m.Decorate(block, spv::Decoration::Block);
  t.m.MemberDecorate(block, 0, spv::Decoration::Offset, {byte_offset});
  const Id v =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, block),
                   spv::StorageClass::PushConstant);
  t.m.Name(v, "user_data");
  if (PushCodeBase()) {
    t.pc_base_var = v;
    t.pc_base_member = 1;
  }
  return v;
}

void SeedUserData(Translator& t, Id user_data) {
  const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (u32 i = 0; i < 16; i++)
    t.SetSg(i, t.m.Load(t.t_u,
                        t.m.AccessChain(p_u, user_data, {t.U32(0), t.U32(i)})));
}

bool TranslateVs(const Program& program,
                 const u32* vs_user_data,
                 const base::HashSet<u32>& flat_attrs,
                 u32 tex_binding_base,
                 u32 tex_3d_mask,
                 u32 tex_1d_mask,
                 bool gl_clip_space,
                 const GsPipeline* es,
                 u32 int_attr_mask,
                 Recompiled& r,
                 Translator& t) {
  const u64 fetch =
      (static_cast<u64>(vs_user_data[1] & 0xFFFF) << 32) | vs_user_data[0];
  const base::Vector<u8> reachable = ComputeReachability(program);
  t.spill_vgprs = PlanLaneSpills(program, reachable.data());
  base::Vector<FetchAttr> direct_attrs;
  for (u32 i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (!reachable[i] || (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf) ||
        inst.opcode > 0x03)
      continue;
    const u32 w = inst.raw[0], w1 = inst.raw[1];
    const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
    const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
    const u32 soffset = (w1 >> 24) & 0xFF;
    if (!idxen || offen || soffset != 128)
      continue;
    const bool typed = inst.enc == Enc::kMtbuf;
    direct_attrs.push_back({static_cast<u32>(direct_attrs.size()),
                            inst.opcode + 1, (w1 >> 8) & 0xFF, srsrc, 0, true,
                            inst.pc, typed ? (w >> 19) & 0xF : 0,
                            typed ? (w >> 23) & 0x7 : 0});
  }
  base::Vector<FetchAttr> attrs;
  if (CallsFetchShader(program))
    attrs = ParseFetch(fetch);
  if (attrs.empty() && !r.direct)
    attrs = base::move(direct_attrs);
  r.direct_vs = r.direct && attrs.empty();
  t.InitTypes(&program);

  base::Vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);
  Id point_out = 0;
  if (!es) {
    point_out = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_f),
                             spv::StorageClass::Output);
    t.m.Decorate(point_out, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::PointSize)});
    iface.push_back(point_out);
  }

  StageContext sc;
  sc.gcn_tsharp = true;
  sc.r = &r;
  sc.direct = r.direct;
  sc.iface = &iface;
  sc.pos_out = pos_out;
  sc.flat_attrs = &flat_attrs;
  const MimgBindingPlan vs_mimg_plan =
      PlanMimgBindings(program, reachable.data());
  if (!vs_mimg_plan.binding_srsrc.empty()) {
    if (vs_mimg_plan.binding_srsrc.size() + tex_binding_base >
        StageContext::kMaxPsSamplers) {
      WarnUnsupported("mimg.vs-binding-count",
                      static_cast<u32>(vs_mimg_plan.binding_srsrc.size()));
      return false;
    }
    sc.mimg_plan = &vs_mimg_plan;
    sc.tex_binding_base = tex_binding_base;
    sc.tex_3d_mask = tex_3d_mask;
    sc.tex_1d_mask = tex_1d_mask;
    for (u32 i = 0; i < vs_mimg_plan.binding_srsrc.size(); i++)
      r.vs_texs.push_back({i + tex_binding_base, vs_mimg_plan.binding_srsrc[i],
                           vs_mimg_plan.binding_storage[i],
                           ((tex_3d_mask >> i) & 1u) != 0,
                           ((tex_1d_mask >> i) & 1u) != 0, false});
  }
  for (const FetchAttr& attr : attrs)
    if (attr.direct_fetch)
      sc.direct_vfetch.insert(attr.pc);
  if (UsesDsSwizzle(program, reachable.data()))
    EnableDsSwizzle(t, sc, iface);
  if (es) {
    sc.es_ring_vec4s = base::Max(1u, (es->esgs_dwords + 3) / 4);
    const Id ring = t.m.TypeArray(t.TypeV4u(), sc.es_ring_vec4s);
    sc.es_ring = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, ring),
                              spv::StorageClass::Output);
    t.m.Decorate(sc.es_ring, spv::Decoration::Location, {0});
    t.m.Name(sc.es_ring, "esgs_ring");
    iface.push_back(sc.es_ring);
  }
  const Id user_data = DeclareUserData(t, 0);
  if (sc.direct)
    DeclareDirectMemory(t, sc);

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  SeedUserData(t, user_data);
  if (point_out)
    t.m.Store(point_out, t.F32(1.f));

  Id debug_vertex_index = 0;
  if (kGpuVsfull) {
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    debug_vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(debug_vertex_index, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::VertexIndex)});
    iface.push_back(debug_vertex_index);
  }

  {
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    Id vertex_index = debug_vertex_index;
    if (!vertex_index) {
      vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
      t.m.Decorate(vertex_index, spv::Decoration::BuiltIn,
                   {static_cast<u32>(spv::BuiltIn::VertexIndex)});
      iface.push_back(vertex_index);
    }
    const Id instance_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(instance_index, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::InstanceIndex)});
    iface.push_back(instance_index);
    t.SetVg(0, t.m.Load(t.t_u, vertex_index));
    const Id instance = t.m.Load(t.t_u, instance_index);
    t.SetVg(1, instance);
    t.SetVg(3, instance);
  }

  for (const FetchAttr& a : attrs) {
    const Id comp_ty = a.num_comps == 1   ? t.t_f
                       : a.num_comps == 2 ? t.t_v2
                       : a.num_comps == 3 ? t.t_v3
                                          : t.t_v4;
    const Id in_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, comp_ty),
                     spv::StorageClass::Input);
    t.m.Decorate(in_var, spv::Decoration::Location, {a.semantic});
    t.m.Name(in_var, "v_attr" + base::ToString(a.semantic));
    iface.push_back(in_var);
    const Id val = t.m.Load(comp_ty, in_var);
    const bool uint_attr = (int_attr_mask >> a.semantic) & 1;
    const bool sint_attr = (int_attr_mask >> (16 + a.semantic)) & 1;
    for (u32 c = 0; c < a.num_comps; c++) {
      Id comp = a.num_comps == 1 ? val : t.m.CompositeExtract(t.t_f, val, c);
      if (kGpuVsflipz && a.semantic == 0 && c == 2 && a.num_comps >= 3)
        comp = t.FNeg(comp);
      if (uint_attr)
        t.SetVg(a.dest_vgpr + c,
                t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {comp}));
      else if (sint_attr)
        t.SetVg(a.dest_vgpr + c,
                t.m.Bitcast(t.t_u,
                            t.m.Emit(spv::Op::OpConvertFToS, t.t_i, {comp})));
      else
        t.SetVgF(a.dest_vgpr + c, comp);
    }
    r.attrs.push_back({a.semantic, a.num_comps, a.table_sgpr, a.dword_off,
                       a.direct_fetch, 0, a.pc, 0, a.dfmt, a.nfmt,
                       a.index_vgpr >= 1 && a.index_vgpr <= 3});
  }

  if (!sc.direct && !PlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind,
                               reachable.data(), CbufBindingLimit(t)))
    return false;
  if (!sc.direct)
    PlanGfxBuffers(program, 0, &sc.direct_vfetch, r.vs_bufs, sc.gfx_buf_bind,
                   reachable.data());

  if (!kGpuVsfull)
    EmitBody(t, program, sc, reachable.data());
  r.num_params = sc.max_param;

  if (debug_vertex_index) {
    const Id vertex = t.m.Load(t.t_u, debug_vertex_index);
    const Id x = t.SelectF(t.IsNonZero(t.And(vertex, t.U32(1))), t.F32(1.f),
                           t.F32(-1.f));
    const Id y = t.SelectF(t.IsNonZero(t.And(vertex, t.U32(2))), t.F32(-1.f),
                           t.F32(1.f));
    t.m.Store(pos_out,
              t.m.CompositeConstruct(t.t_v4, {x, y, t.F32(0.f), t.F32(1.f)}));
  }

  if (gl_clip_space && !kGpuNoZRemap && !es) {
    const Id p_out_f = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id z_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(2)});
    const Id w_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(3)});
    const Id z = t.m.Load(t.t_f, z_ptr), wv = t.m.Load(t.t_f, w_ptr);
    t.m.Store(z_ptr, t.FMul(t.FAdd(z, wv), t.F32(0.5f)));
  }
  if (kGpuVsProbeW > 0.0f) {
    const Id p_out_f3 = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id zp = t.m.AccessChain(p_out_f3, pos_out, {t.U32(2)});
    const Id wp = t.m.AccessChain(p_out_f3, pos_out, {t.U32(3)});
    const Id wv2 = t.m.Load(t.t_f, wp);
    t.m.Store(zp, t.FMul(t.FMul(wv2, wv2), t.F32(kGpuVsProbeW)));
  }
  if (kGpuVsForceZ > 0.0f) {
    const Id p_out_f2 = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id zp = t.m.AccessChain(p_out_f2, pos_out, {t.U32(2)});
    const Id wp = t.m.AccessChain(p_out_f2, pos_out, {t.U32(3)});
    t.m.Store(zp, t.FMul(t.m.Load(t.t_f, wp), t.F32(kGpuVsForceZ)));
  }

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Vertex, main_fn, "main", iface);
  return true;
}

base::Pair<u32, spv::ExecutionMode> GsInputOf(u32 prim) {
  switch (prim) {
    case 0x1:
      return {1, spv::ExecutionMode::InputPoints};
    case 0x2:
    case 0x3:
      return {2, spv::ExecutionMode::InputLines};
    case 0xa:
    case 0xb:
      return {4, spv::ExecutionMode::InputLinesAdjacency};
    case 0xc:
    case 0xd:
      return {6, spv::ExecutionMode::InputTrianglesAdjacency};
    default:
      return {3, spv::ExecutionMode::Triangles};
  }
}

bool TranslateGs(const Program& program,
                 const GsPipeline& gs,
                 const base::Vector<GsCopyExport>& exports,
                 const base::HashSet<u32>& flat_params,
                 bool gl_clip_space,
                 Recompiled& r,
                 Translator& t) {
  const base::Vector<u8> reachable = ComputeReachability(program);
  t.spill_vgprs = PlanLaneSpills(program, reachable.data());
  t.InitTypes(&program);
  t.m.Capability(spv::Capability::Geometry);
  const auto [verts, input_mode] = GsInputOf(gs.input_prim);

  base::Vector<Id> iface;
  StageContext sc;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_params;
  sc.gl_clip = gl_clip_space && !kGpuNoZRemap;
  sc.pos_out = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                            spv::StorageClass::Output);
  t.m.Decorate(sc.pos_out, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::Position)});
  iface.push_back(sc.pos_out);

  sc.gs_input_verts = verts;
  if (!PlanCbufs(program, r.vs_cbufs.size(), r.gs_cbufs, sc.cbuf_bind,
                 reachable.data(), CbufBindingLimit(t)))
    return false;
  sc.es_ring_vec4s = base::Max(1u, (gs.esgs_dwords + 3) / 4);
  const Id ring =
      t.m.TypeArray(t.m.TypeArray(t.TypeV4u(), sc.es_ring_vec4s), verts);
  sc.es_ring = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, ring),
                            spv::StorageClass::Input);
  t.m.Decorate(sc.es_ring, spv::Decoration::Location, {0});
  t.m.Name(sc.es_ring, "esgs_ring");
  iface.push_back(sc.es_ring);

  sc.gsvs_dwords = base::Max(1u, gs.gsvs_dwords);
  const Id gsvs = t.m.TypeArray(t.t_u, sc.gsvs_dwords);
  sc.gsvs = t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, gsvs),
                         spv::StorageClass::Private, t.m.ConstNull(gsvs));
  t.m.Name(sc.gsvs, "gsvs_ring");
  sc.gs_emitted = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                               t.m.ConstNull(t.t_u));
  sc.gs_max_vert_out = base::Max(1u, gs.max_vert_out);
  sc.gs_exports = &exports;

  const Id p_in_i = t.m.TypePointer(spv::StorageClass::Input, t.t_i);
  const Id primitive_id = t.m.Variable(p_in_i, spv::StorageClass::Input);
  t.m.Decorate(primitive_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::PrimitiveId)});
  iface.push_back(primitive_id);
  const Id invocation_id = t.m.Variable(p_in_i, spv::StorageClass::Input);
  t.m.Decorate(invocation_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::InvocationId)});
  iface.push_back(invocation_id);

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  static constexpr u32 kVertexVgpr[6] = {0, 1, 3, 4, 5, 6};
  for (u32 i = 0; i < 6; i++)
    t.SetVg(kVertexVgpr[i], t.U32(i));
  t.SetVg(2, t.m.Bitcast(t.t_u, t.m.Load(t.t_i, primitive_id)));
  t.SetVg(7, t.m.Bitcast(t.t_u, t.m.Load(t.t_i, invocation_id)));
  EmitBody(t, program, sc, reachable.data());
  r.num_params = sc.max_param;

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Geometry, main_fn, "main", iface);
  t.m.ExecMode(main_fn, spv::ExecutionMode::Invocations,
               {base::Max(1u, gs.instances)});
  t.m.ExecMode(main_fn, input_mode);
  t.m.ExecMode(main_fn, gs.out_prim == 0 ? spv::ExecutionMode::OutputPoints
                        : gs.out_prim == 1
                            ? spv::ExecutionMode::OutputLineStrip
                            : spv::ExecutionMode::OutputTriangleStrip);
  t.m.ExecMode(main_fn, spv::ExecutionMode::OutputVertices,
               {sc.gs_max_vert_out});
  return true;
}

bool TranslatePs(const Program& program,
                 const base::HashSet<u32>& flat_attrs,
                 u32 ps_input_ena,
                 const u32* ps_in_cntl,
                 u32 ps_num_interp,
                 const base::Vector<u32>* vs_exported_params,
                 u32 tex_3d_mask,
                 u32 tex_1d_mask,
                 u32 tex_uint_mask,
                 u32 mrt_uint_mask,
                 u32 mrt_bound_mask,
                 u32 col_format,
                 Recompiled& r,
                 Translator& t) {
  base::Vector<Id> iface;
  const base::Vector<u8> reachable = ComputeReachability(program);
  t.spill_vgprs = PlanLaneSpills(program, reachable.data());
  StageContext sc;
  sc.is_ps = true;
  sc.gcn_tsharp = true;
  sc.r = &r;
  sc.direct = r.direct;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  sc.pervertex_attrs = PlanPerVertexAttrs(program, reachable.data());
  if (!sc.direct &&
      !PlanCbufs(program, r.vs_cbufs.size() + r.gs_cbufs.size(), r.ps_cbufs,
                 sc.cbuf_bind, reachable.data(), CbufBindingLimit(t)))
    return false;
  if (!sc.direct)
    PlanGfxBuffers(program, r.vs_bufs.size(), nullptr, r.ps_bufs,
                   sc.gfx_buf_bind, reachable.data());

  sc.ds_own_lane = PlanDsOwnLane(program, reachable.data());
  if (const u32 lds_dwords = GraphicsLdsDwords(program, reachable.data())) {
    const Id lds_arr = t.m.TypeArray(t.t_u, lds_dwords);
    sc.lds_storage = spv::StorageClass::Private;
    sc.lds_dwords = lds_dwords;
    sc.lds_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, lds_arr),
                     spv::StorageClass::Private, t.m.ConstNull(lds_arr));
    t.m.Name(sc.lds_var, "lds");
  }

  const MimgBindingPlan mimg_plan = PlanMimgBindings(program, reachable.data());
  if (mimg_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers) {
    WarnUnsupported("mimg.binding-count",
                    static_cast<u32>(mimg_plan.binding_srsrc.size()));
    return false;
  }
  sc.mimg_plan = &mimg_plan;
  sc.bindless = r.bindless = g_bindless && !mimg_plan.binding_srsrc.empty() &&
                             r.vs_texs.empty() &&
                             base::NoneOf(mimg_plan.binding_storage.begin(),
                                          mimg_plan.binding_storage.end(),
                                          [](bool storage) { return storage; });
  sc.tex_3d_mask = tex_3d_mask;
  sc.tex_1d_mask = tex_1d_mask;
  sc.tex_uint_mask = tex_uint_mask;
  sc.mrt_uint_mask = mrt_uint_mask;
  sc.mrt_bound_mask = mrt_bound_mask;
  sc.col_format = col_format;
  for (u32 i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back(
        {i, mimg_plan.binding_srsrc[i], mimg_plan.binding_storage[i],
         ((tex_3d_mask >> i) & 1u) != 0, ((tex_1d_mask >> i) & 1u) != 0,
         ((tex_uint_mask >> i) & 1u) != 0});

  if (UsesDsSwizzle(program, reachable.data()))
    EnableDsSwizzle(t, sc, iface);
  const Id user_data = DeclareUserData(t, kPsUserDataOffset);
  if (sc.direct)
    DeclareDirectMemory(t, sc);
  if (sc.bindless) {
    BindlessLookup(t, sc, 0);
    BindlessLookup(t, sc, 1);
  }
  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data);
  sc.ps_in_cntl = ps_in_cntl;
  sc.ps_num_interp = ps_num_interp;
  sc.vs_exported_params = vs_exported_params;
  SeedPsInputVgprs(t, ps_input_ena, iface);
  SeedPsBarycentrics(t, ps_input_ena, sc);

  bool has_color_export = false;
  for (u32 i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (reachable[i] && inst.enc == Enc::kExp &&
        ((inst.raw[0] >> 4) & 0x3F) <= 7 && (inst.raw[0] & 0xF)) {
      has_color_export = true;
      break;
    }
  }

  if (has_color_export) {
    if (sc.mrt_bound_mask & 1u) {
      const bool mrt0_int = (sc.mrt_uint_mask & 1u) != 0;
      t.m.Store(
          PsColorOut(t, sc, 0),
          mrt0_int
              ? t.m.ConstComposite(t.m.TypeVec(t.t_u, 4),
                                   {t.U32(0), t.U32(0), t.U32(0), t.U32(0)})
              : t.m.ConstComposite(
                    t.t_v4, {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(0.f)}));
    }
    sc.color_written_var = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                        t.m.ConstNull(t.t_u));
  }
  const bool probe = ProbeThisPs(t);
  if (kGpuPsvgpr > 0 && kGpuPsvgprAt && probe)
    t.probe_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, t.t_v4),
                     spv::StorageClass::Private, t.m.ConstNull(t.t_v4));
  if (kGpuPstex != 0 && probe)
    t.last_texel_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, t.t_v4),
                     spv::StorageClass::Private, t.m.ConstNull(t.t_v4));
  if (!(kGpuPswhite && probe))
    EmitBody(t, program, sc, reachable.data());

  if (sc.wrote_color && sc.color_written_var) {
    if (!kGpuNokill && !(kGpuPsreach && probe)) {
      const Id wrote = t.IsNonZero(t.m.Load(t.t_u, sc.color_written_var));
      const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
      t.m.SelectionMerge(after_kill);
      t.m.BranchConditional(wrote, after_kill, kill_blk);
      t.m.OpenBlock(kill_blk);
      t.m.Kill();
      t.m.OpenBlock(after_kill);
    }
  }

  const Id pstex =
      t.last_texel_var && t.last_texel ? t.m.Load(t.t_v4, t.last_texel_var) : 0;
  if (kGpuPstex != 0 && probe && has_color_export && (sc.mrt_bound_mask & 1u) &&
      pstex && !(sc.mrt_uint_mask & 1u))
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.CompositeConstruct(
                  t.t_v4, {t.FMul(t.m.CompositeExtract(t.t_f, pstex, 0),
                                  t.F32(kGpuPstexScale)),
                           t.FMul(t.m.CompositeExtract(t.t_f, pstex, 1),
                                  t.F32(kGpuPstexScale)),
                           t.FMul(t.m.CompositeExtract(t.t_f, pstex, 2),
                                  t.F32(kGpuPstexScale)),
                           t.F32(1.f)}));

  if (kGpuPsattr > 0 && probe && has_color_export && (sc.mrt_bound_mask & 1u) &&
      !(sc.mrt_uint_mask & 1u)) {
    const Id in = PsInputVar(t, sc, static_cast<u32>(kGpuPsattr - 1));
    t.m.Store(PsColorOut(t, sc, 0), t.m.Load(t.t_v4, in));
  }

  if (kGpuPsvgpr > 0 && probe && has_color_export && (sc.mrt_bound_mask & 1u) &&
      !(sc.mrt_uint_mask & 1u)) {
    const u32 first = static_cast<u32>(kGpuPsvgpr - 1);
    const Id v = t.probe_var ? t.m.Load(t.t_v4, t.probe_var)
                             : t.m.CompositeConstruct(
                                   t.t_v4, {t.VgF(first), t.VgF(first + 1),
                                            t.VgF(first + 2), t.F32(1.f)});
    t.m.Store(
        PsColorOut(t, sc, 0),
        t.m.CompositeConstruct(
            t.t_v4,
            {t.FMul(t.m.CompositeExtract(t.t_f, v, 0), t.F32(kGpuPstexScale)),
             t.FMul(t.m.CompositeExtract(t.t_f, v, 1), t.F32(kGpuPstexScale)),
             t.FMul(t.m.CompositeExtract(t.t_f, v, 2), t.F32(kGpuPstexScale)),
             t.F32(1.f)}));
  }

  if (kGpuPsreach && probe && has_color_export && (sc.mrt_bound_mask & 1u) &&
      !(sc.mrt_uint_mask & 1u)) {
    const Id missed = sc.color_written_var
                          ? t.IsZero(t.m.Load(t.t_u, sc.color_written_var))
                          : t.m.ConstBool(false);
    const Id v = t.SelectF(missed, t.F32(1.f), t.F32(0.f));
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.CompositeConstruct(t.t_v4, {v, v, v, t.F32(1.f)}));
  }

  if (kGpuPswhite && probe && has_color_export && (sc.mrt_bound_mask & 1u) &&
      !(sc.mrt_uint_mask & 1u))
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(1.f), t.F32(1.f), t.F32(1.f), t.F32(1.f)}));

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, sc.main_fn, "main", iface);
  t.m.ExecMode(sc.main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

bool TranslateDepthOnlyPs(Translator& t) {
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, main_fn, "main", {});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

namespace {
bool NativeMimgForm(u32 op) {
  switch (op) {
    case 0x00:
    case 0x01:
    case 0x24:
    case 0x27:
    case 0x2c:
    case 0x2f:
    case 0x34:
    case 0x37:
      return true;
    default:
      return false;
  }
}

u64 NativeImageCandidates(const Program& program,
                          const RecompiledCs& r,
                          const base::HashMap<u32, u32>& bind) {
  u64 ok = 0, bad = 0;
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kMimg)
      continue;
    const auto it = bind.find(inst.pc);
    if (it == bind.end() || it->second >= 64)
      continue;
    const u32 w = inst.raw[0];
    const u32 op = (w >> 18) & 0x7F;
    const bool plain = !((w >> 14) & 0xF);
    (NativeMimgForm(op) && plain ? ok : bad) |= 1ull << it->second;
  }
  u64 out = ok & ~bad;
  for (const CsResource& res : r.resources)
    if (res.binding < 64 && (res.kind != 1 || res.written ||
                             res.runtime_image || res.runtime_address))
      out &= ~(1ull << res.binding);
  return out;
}
}  // namespace

bool TranslateCs(const Program& program,
                 u32 num_thread_x,
                 u32 num_thread_y,
                 u32 num_thread_z,
                 u32 user_sgpr,
                 u32 tgid_enable,
                 u32 lds_dwords,
                 u64 native_mask,
                 u64 uint_mask,
                 u64 shared_mask,
                 RecompiledCs& r,
                 Translator& t) {
  if (program.empty())
    return false;
  StageContext sc;
  sc.is_cs = true;
  sc.lds_dwords = lds_dwords * 128;
  const base::Vector<u8> reachable = ComputeReachability(program);
  t.spill_vgprs = PlanLaneSpills(program, reachable.data());
  if (!PlanCsResources(program, reachable.data(), sc.lds_dwords, r,
                       sc.cs_bind) ||
      r.resources.empty())
    return false;
  r.native_candidates = NativeImageCandidates(program, r, sc.cs_bind);
  for (const CsResource& res : r.resources)
    if (res.written)
      sc.cs_written.insert(res.binding);
  const u64 native = native_mask & r.native_candidates;
  MimgBindingPlan image_plan;
  if (native) {
    image_plan = PlanMimgBindings(program, reachable.data());
    if (image_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers)
      return false;
    sc.mimg_plan = &image_plan;
    sc.tex_binding_base = kCsTexBindingBase;
    sc.tex_uint_mask = uint_mask;
    r.textures.assign(image_plan.binding_srsrc.size(), -1);
    for (const Inst& inst : program) {
      const auto it = sc.cs_bind.find(inst.pc);
      if (inst.enc != Enc::kMimg || it == sc.cs_bind.end() ||
          !((native >> it->second) & 1))
        continue;
      sc.cs_native.insert(it->second);
      const u32 b = image_plan.binding_by_pc[inst.pc];
      r.textures[b] = static_cast<int>(kCsTexBindingBase + b);
    }
  }
  const u32 threads_per_group = (num_thread_x ? num_thread_x : 1) *
                                (num_thread_y ? num_thread_y : 1) *
                                (num_thread_z ? num_thread_z : 1);
  const LdsBarrierPlan lds_bar = PlanLdsBarriers(
      program, reachable.data(), threads_per_group, &sc.cs_bind, shared_mask);
  sc.lds_barrier_at = lds_bar.at;
  sc.lockstep_loop = lds_bar.lockstep;
  if (lds_bar.orders_memory)
    t.barrier_semantics = Translator::kBarrierWithMemory;
  r.orders_shared_memory =
      PlanLdsBarriers(program, reachable.data(), threads_per_group,
                      &sc.cs_bind, ~0ull)
          .orders_memory;

  bool uses_ds_swizzle = false, uses_cross_lane = false;
  for (u32 i = 0; i < program.size(); i++) {
    if (!reachable[i])
      continue;
    const Inst& in = program[i];
    if (in.enc == Enc::kDs && in.opcode == 0x35)
      uses_ds_swizzle = true;
    if (in.enc == Enc::kVop2 && (in.opcode == 0x01 || in.opcode == 0x02 ||
                                 in.opcode == 0x23 || in.opcode == 0x24))
      uses_cross_lane = true;
    if (in.enc == Enc::kVop1 && in.opcode == 0x02)
      uses_cross_lane = true;
  }

  t.InitTypes(&program);
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
    t.m.Name(v, "buf" + base::ToString(res.binding));
    sc.cs_ssbo[res.binding] = v;
  }
  if (r.gds_binding >= 0) {
    const Id v = t.m.Variable(p_buf, spv::StorageClass::StorageBuffer);
    t.m.Decorate(v, spv::Decoration::DescriptorSet, {0});
    t.m.Decorate(v, spv::Decoration::Binding,
                 {static_cast<u32>(r.gds_binding)});
    t.m.Name(v, "gds");
    sc.gds_var = v;
  }

  if (sc.lds_dwords) {
    const Id lds_arr = t.m.TypeArray(t.t_u, sc.lds_dwords);
    sc.lds_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Workgroup, lds_arr),
                     spv::StorageClass::Workgroup);
    t.m.Name(sc.lds_var, "lds");
  }

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

  const Id t_uv3 = t.m.TypeVec(t.t_u, 3);
  const Id p_in_v3 = t.m.TypePointer(spv::StorageClass::Input, t_uv3);
  const Id local_id = t.m.Variable(p_in_v3, spv::StorageClass::Input);
  t.m.Decorate(local_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::LocalInvocationId)});
  const Id group_id = t.m.Variable(p_in_v3, spv::StorageClass::Input);
  t.m.Decorate(group_id, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::WorkgroupId)});
  base::Vector<Id> iface{local_id, group_id};
  const u32 threads = threads_per_group;
  if (uses_cross_lane && HasControlFlow(program) && threads <= kGcnWave &&
      threads > HostSubgroupSize())
    sc.lockstep_loop = true;
  if (sc.lockstep_loop) {
    t.m.Capability(spv::Capability::GroupNonUniform);
    sc.subgroup_id =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                     spv::StorageClass::Input);
    t.m.Decorate(sc.subgroup_id, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::SubgroupId)});
    iface.push_back(sc.subgroup_id);
  }
  if ((uses_cross_lane || sc.lockstep_loop) && threads) {
    const Id lii =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                     spv::StorageClass::Input);
    t.m.Decorate(lii, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::LocalInvocationIndex)});
    iface.push_back(lii);
    const Id xchg_arr = t.m.TypeArray(t.t_u, threads * 2);
    t.xchg_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Workgroup, xchg_arr),
                     spv::StorageClass::Workgroup);
    t.m.Name(t.xchg_var, "wave_xchg");
    t.xchg_lanes = threads;
    t.xchg_index = lii;
  }
  if (uses_ds_swizzle) {
    t.m.Capability(spv::Capability::GroupNonUniform);
    t.m.Capability(spv::Capability::GroupNonUniformShuffle);
    sc.subgroup_local_id =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                     spv::StorageClass::Input);
    t.m.Decorate(sc.subgroup_local_id, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::SubgroupLocalInvocationId)});
    iface.push_back(sc.subgroup_local_id);
  }

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  const Id p_pc_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (u32 i = 0; i < 16; i++)
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
  for (u32 c = 0; c < 3; c++)
    t.SetVg(c, t.m.Load(t.t_u, t.m.AccessChain(p_in_u, local_id, {t.U32(c)})));
  if (t.xchg_var) {
    const Id index = t.m.Load(t.t_u, t.xchg_index);
    t.xchg_index = index;
    t.lane_id = t.And(index, t.U32(63));
    t.wave_base = t.And(index, t.U32(~63u));
  }
  sc.uniform_points = sc.lockstep_loop ? base::Vector<u8>(program.size(), 1)
                                       : UniformPoints(program);
  sc.uniform_readfirstlane =
      ProvenUniformReadFirstLane(program, reachable.data());
  if (kGcnCsVgprCs && t.program_base == kGcnCsVgprCs) {
    const Id arr = t.m.TypeArray(t.t_u, 4);
    sc.cs_probe_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, arr),
                     spv::StorageClass::Private, t.m.ConstNull(arr));
  }
  t.SeedExec();
  t.predicate_vector = true;
  EmitCfg(t, program, sc, reachable.data());
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

void DumpProgram(const char* tag, const Program& program) {
  BASE_LOGI("shdis", "{}, {} insts:", tag, program.size());
  for (const Inst& inst : program)
    BASE_LOGI("shdis", "  {}", DisasmLine(inst).c_str());
}

void MaybeDumpBranchy(const char* tag, const Program& program) {
  if (!kGpuShdis)
    return;
  static int dumped = 0;
  if (!HasControlFlow(program) || dumped >= 2)
    return;
  dumped++;
  BASE_LOGI("shdis", "(branchy)");
  DumpProgram(tag, program);
}

void MaybeDumpByAddr(const char* tag,
                     const void* code,
                     const Program& program) {
  static bool dumped = false;
  if (!kShDisAddr || dumped || !CodeMatches(kShDisAddr, code))
    return;
  dumped = true;
  BASE_LOGI("shdis", "@{}:", code);
  DumpProgram(tag, program);
}

bool NoOpt() {
  return kGpuSpirvNoopt;
}

base::String PlanSummaryGfx(const Recompiled& r, bool ps) {
  base::String s = ps ? "ps plan:" : "vs plan:";
  if (!ps) {
    s += " attrs=" + base::ToString(r.attrs.size());
  }
  const auto& cbufs = ps ? r.ps_cbufs : r.vs_cbufs;
  const auto& bufs = ps ? r.ps_bufs : r.vs_bufs;
  s += " cbufs:";
  for (const ShaderCbuf& c : cbufs)
    s += " [b" + base::ToString(c.binding) +
         " ud=" + base::ToString(c.ud_sgpr) + (c.pointer ? " ptr" : "") + " " +
         base::ToString(c.num_dwords) + "dw]";
  s += " bufs:";
  for (const ShaderBuffer& b : bufs)
    s += " [b" + base::ToString(b.binding) + " srsrc=s" +
         base::ToString(b.srsrc_sgpr) + " pc=" + base::ToString(b.use_pc) + "]";
  if (ps) {
    s += " texs:";
    for (const ShaderTex& tex : r.ps_texs)
      s += " [t" + base::ToString(tex.binding) +
           " ud=" + base::ToString(tex.ud_sgpr) +
           (tex.storage ? " storage" : "") + (tex.is_3d ? " 3d" : "") +
           (tex.is_1d ? " 1d" : "") + "]";
  }
  return s;
}

base::String PlanSummaryCs(const RecompiledCs& r) {
  base::String s = "cs plan: tg=" + base::ToString(r.local_size[0]) + "x" +
                   base::ToString(r.local_size[1]) + "x" +
                   base::ToString(r.local_size[2]);
  for (const CsResource& res : r.resources)
    s += " [b" + base::ToString(res.binding) + " s" +
         base::ToString(res.base_sgpr) + " kind=" + base::ToString(res.kind) +
         (res.written ? " w" : "") + " min=" + base::ToString(res.min_bytes) +
         "]";
  return s;
}

}  // namespace

base::Vector<u32> EmitRectListGeometry(u32 num_params,
                                       const base::HashSet<u32>& flat_attrs) {
  spirv::Module m;
  m.Capability(spv::Capability::Geometry);
  const Id t_void = m.TypeVoid(), t_f = m.TypeFloat(32),
           t_v4 = m.TypeVec(t_f, 4);
  const Id t_fn = m.TypeFunction(t_void);
  const Id t_in_v4 = m.TypeArray(t_v4, 3);
  const Id p_in_v4 = m.TypePointer(spv::StorageClass::Input, t_v4);
  const Id p_out_v4 = m.TypePointer(spv::StorageClass::Output, t_v4);

  base::Vector<Id> iface;
  const Id in_pos = m.Variable(m.TypePointer(spv::StorageClass::Input, t_in_v4),
                               spv::StorageClass::Input);
  const Id out_pos = m.Variable(p_out_v4, spv::StorageClass::Output);
  m.Decorate(in_pos, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::Position)});
  m.Decorate(out_pos, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::Position)});
  iface.push_back(in_pos);
  iface.push_back(out_pos);

  base::Vector<Id> inputs(num_params), outputs(num_params);
  for (u32 p = 0; p < num_params; p++) {
    inputs[p] = m.Variable(m.TypePointer(spv::StorageClass::Input, t_in_v4),
                           spv::StorageClass::Input);
    outputs[p] = m.Variable(p_out_v4, spv::StorageClass::Output);
    m.Decorate(inputs[p], spv::Decoration::Location, {p});
    m.Decorate(outputs[p], spv::Decoration::Location, {p});
    if (flat_attrs.count(p)) {
      m.Decorate(inputs[p], spv::Decoration::Flat);
      m.Decorate(outputs[p], spv::Decoration::Flat);
    }
    iface.push_back(inputs[p]);
    iface.push_back(outputs[p]);
  }

  const Id main_fn = m.BeginFunction(t_void, t_fn);
  const auto load_input = [&](Id input, u32 vertex) {
    return m.Load(t_v4, m.AccessChain(p_in_v4, input, {m.ConstU32(vertex)}));
  };
  const auto fourth_corner = [&](Id input) {
    const Id v0 = load_input(input, 0), v1 = load_input(input, 1);
    const Id v2 = load_input(input, 2);
    return m.Emit(spv::Op::OpFSub, t_v4,
                  {m.Emit(spv::Op::OpFAdd, t_v4, {v1, v2}), v0});
  };
  const Id pos3 = fourth_corner(in_pos);
  base::Vector<Id> param3(num_params);
  for (u32 p = 0; p < num_params; p++)
    if (!flat_attrs.count(p))
      param3[p] = fourth_corner(inputs[p]);

  for (u32 vertex = 0; vertex < 4; vertex++) {
    m.Store(out_pos, vertex < 3 ? load_input(in_pos, vertex) : pos3);
    for (u32 p = 0; p < num_params; p++) {
      const Id value = flat_attrs.count(p) ? load_input(inputs[p], 0)
                       : vertex < 3        ? load_input(inputs[p], vertex)
                                           : param3[p];
      m.Store(outputs[p], value);
    }
    m.EmitVoid(spv::Op::OpEmitVertex, {});
  }
  m.EmitVoid(spv::Op::OpEndPrimitive, {});
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::Geometry, main_fn, "main", iface);
  m.ExecMode(main_fn, spv::ExecutionMode::Invocations, {1});
  m.ExecMode(main_fn, spv::ExecutionMode::Triangles);
  m.ExecMode(main_fn, spv::ExecutionMode::OutputTriangleStrip);
  m.ExecMode(main_fn, spv::ExecutionMode::OutputVertices, {4});
  return m.Assemble();
}

bool RecompileSpirv(const u32* vs_code,
                    const u32* ps_code,
                    const u32* vs_user_data,
                    const u32* ps_user_data,
                    u32 ps_input_ena,
                    const u32* ps_in_cntl,
                    u32 ps_num_interp,
                    u32 tex_3d_mask,
                    u32 tex_1d_mask,
                    u32 tex_uint_mask,
                    u32 mrt_uint_mask,
                    u32 mrt_bound_mask,
                    bool gl_clip_space,
                    const GsPipeline* gs,
                    u32 int_attr_mask,
                    u32 col_format,
                    u32 tex_cube_mask,
                    Recompiled& r) {
  if (!vs_code || !vs_user_data || !ps_user_data)
    return false;
  if (gs && (!gs->es_code || !gs->gs_code || !gs->es_user_data))
    return false;

  const Program vs_program = DecodeShader(vs_code, 4096);
  const Program ps_program = ps_code ? DecodeShader(ps_code, 4096) : Program{};
  MaybeDumpBranchy("VS", vs_program);
  if (!ps_program.empty())
    MaybeDumpBranchy("PS", ps_program);
  MaybeDumpByAddr("VS", vs_code, vs_program);
  if (!ps_program.empty())
    MaybeDumpByAddr("PS", ps_code, ps_program);

  base::HashSet<u32> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 &&
        (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  base::Vector<u32> vs_exported_params;
  {
    for (const Inst& inst : vs_program)
      if (inst.enc == Enc::kExp) {
        const u32 tgt = (inst.raw[0] >> 4) & 0x3F;
        if (tgt >= 32 && tgt <= 63)
          vs_exported_params.push_back(tgt - 32);
      }
    base::Sort(vs_exported_params.begin(), vs_exported_params.end());
    vs_exported_params.erase(
        base::Unique(vs_exported_params.begin(), vs_exported_params.end()),
        vs_exported_params.end());
  }

  base::HashSet<u32> flat_params;
  for (u32 slot : flat_attrs)
    flat_params.insert((ps_in_cntl && slot < ps_num_interp)
                           ? (ps_in_cntl[slot & 31] & 0x1F)
                           : slot);

  u32 vs_tex_base = 0;
  if (!ps_program.empty()) {
    const base::Vector<u8> ps_reachable = ComputeReachability(ps_program);
    vs_tex_base = static_cast<u32>(
        PlanMimgBindings(ps_program, ps_reachable.data()).binding_srsrc.size());
  }

  const Program es_program = gs ? DecodeShader(gs->es_code, 4096) : Program{};
  base::Vector<GsCopyExport> copy_exports;
  if (gs && !ParseCopyShader(vs_program, gs->max_vert_out, copy_exports)) {
    BASE_LOGI("gcnspv", "copy shader not understood @{}",
              static_cast<const void*>(vs_code));
    return false;
  }
  const u32* vertex_code = gs ? gs->es_code : vs_code;
  const Program& vertex_program = gs ? es_program : vs_program;

  const bool dbg = ShaderDebugEnabled();
  const Program gs_program = gs ? DecodeShader(gs->gs_code, 4096) : Program{};
  r.direct = DirectGcnEnabled() && g_direct_table && !gs;
  r.indirect_cbufs =
      !r.direct &&
      NeedsIndirectCbufs({&vertex_program, &gs_program, &ps_program});
  Translator tv;
  tv.program_base = reinterpret_cast<u64>(vertex_code);
  tv.indirect_cbufs = r.indirect_cbufs;
  tv.tex_cube_mask = tex_cube_mask >> vs_tex_base;
  ResetUnsupported();
  if (dbg)
    AuditBegin(gs ? "es" : "vs", vertex_code, vertex_program);
  const bool vs_ok =
      TranslateVs(vertex_program, gs ? gs->es_user_data : vs_user_data,
                  flat_params, vs_tex_base, tex_3d_mask >> vs_tex_base,
                  tex_1d_mask >> vs_tex_base, gl_clip_space, gs, int_attr_mask,
                  r, tv) &&
      !HadUnsupported();
  base::Vector<u32> vs;
  if (vs_ok)
    vs = tv.m.Assemble();
  if (dbg) {
    if (vs_ok)
      AuditPlan(PlanSummaryGfx(r, false));
    else
      AuditDecline("vs translation rejected");
    AuditEnd(vs_ok ? &vs : nullptr);
  }
  if (!vs_ok) {
    if (TraceEnabled())
      BASE_LOGI("gcnspv", "VS translation rejected @{}",
                static_cast<const void*>(vertex_code));
    return false;
  }

  base::Vector<u32> gs_words;
  if (gs) {
    Translator tg;
    tg.program_base = reinterpret_cast<u64>(gs->gs_code);
    tg.indirect_cbufs = r.indirect_cbufs;
    ResetUnsupported();
    if (dbg)
      AuditBegin("gs", gs->gs_code, gs_program);
    const bool gs_ok = TranslateGs(gs_program, *gs, copy_exports, flat_params,
                                   gl_clip_space, r, tg) &&
                       !HadUnsupported();
    if (gs_ok)
      gs_words = tg.m.Assemble();
    if (dbg) {
      if (!gs_ok)
        AuditDecline("gs translation rejected");
      AuditEnd(gs_ok ? &gs_words : nullptr);
    }
    if (!gs_ok) {
      BASE_LOGI("gcnspv", "GS translation rejected @{}",
                static_cast<const void*>(gs->gs_code));
      return false;
    }
    r.guest_gs = true;
  }

  Translator tp;
  tp.program_base = reinterpret_cast<u64>(ps_code);
  tp.indirect_cbufs = r.indirect_cbufs;
  tp.tex_cube_mask = tex_cube_mask;
  tp.InitTypes(&ps_program);
  ResetUnsupported();
  if (dbg && ps_code)
    AuditBegin("ps", ps_code, ps_program);
  const bool ps_ok =
      (ps_code ? TranslatePs(ps_program, flat_attrs, ps_input_ena, ps_in_cntl,
                             ps_num_interp, &vs_exported_params, tex_3d_mask,
                             tex_1d_mask, tex_uint_mask, mrt_uint_mask,
                             mrt_bound_mask, col_format, r, tp)
               : TranslateDepthOnlyPs(tp)) &&
      !HadUnsupported();
  base::Vector<u32> ps;
  if (ps_ok)
    ps = tp.m.Assemble();
  if (dbg && ps_code) {
    if (ps_ok)
      AuditPlan(PlanSummaryGfx(r, true));
    else
      AuditDecline("ps translation rejected");
    AuditEnd(ps_ok ? &ps : nullptr);
  }
  if (!ps_ok) {
    if (TraceEnabled())
      BASE_LOGI("gcnspv", "PS translation rejected @{}",
                static_cast<const void*>(ps_code));
    return false;
  }

  if (!gs)
    gs_words = EmitRectListGeometry(r.num_params, flat_params);
  base::String err;
  if (NoOpt()) {
    if (!spirv::Validate(vs, &err)) {
      BASE_LOGI("gcnspv", "VS invalid @{}: {}",
                static_cast<const void*>(vs_code), err.c_str());
      return false;
    }
    if (!spirv::Validate(ps, &err)) {
      BASE_LOGI("gcnspv", "PS invalid @{}: {}",
                static_cast<const void*>(ps_code), err.c_str());
      return false;
    }
    if (!spirv::Validate(gs_words, &err)) {
      BASE_LOGI("gcnspv", "GS invalid: {}", err.c_str());
      return false;
    }
    r.vs_spirv = vs;
    r.gs_spirv = gs_words;
    r.fs_spirv = ps;
  } else {
    spirv::Prefetch(ps);
    spirv::Prefetch(gs_words);
    if (!spirv::Finalize(vs, &r.vs_spirv, &err)) {
      BASE_LOGI("gcnspv", "VS invalid @{}: {}",
                static_cast<const void*>(vs_code), err.c_str());
      return false;
    }
    if (!spirv::Finalize(ps, &r.fs_spirv, &err)) {
      BASE_LOGI("gcnspv", "PS invalid @{}: {}",
                static_cast<const void*>(ps_code), err.c_str());
      return false;
    }
    if (!spirv::Finalize(gs_words, &r.gs_spirv, &err)) {
      BASE_LOGI("gcnspv", "GS invalid: {}", err.c_str());
      return false;
    }
  }
  r.ok = !r.vs_spirv.empty() && !r.gs_spirv.empty() && !r.fs_spirv.empty();

  if (kSpvDumpAddr && ps_code &&
      reinterpret_cast<u64>(ps_code) == kSpvDumpAddr.get() &&
      !r.fs_spirv.empty()) {
    static bool once = false;
    if (!once) {
      once = true;
      char path[128];
      std::snprintf(path, sizeof(path), "/tmp/delta_ps_%lx.spv",
                    (unsigned long)kSpvDumpAddr.get());
      if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(r.fs_spirv.data(), 4, r.fs_spirv.size(), f);
        std::fclose(f);
        BASE_LOGI("spvdump", "ps {:#x} -> {} ({} words)", kSpvDumpAddr.get(),
                  path, r.fs_spirv.size());
      }
    }
  }
  if (kGpuSpirv) {
    static int ok_count = 0, cfg_count = 0, logged = 0;
    if (r.ok)
      ok_count++;
    if (HasControlFlow(vs_program) || HasControlFlow(ps_program))
      cfg_count++;
    if (logged < 12) {
      logged++;
      BASE_LOGI("gcnspv", "recompiled ok={} (cfg-shaders={}) this={}", ok_count,
                cfg_count, r.ok ? "spirv" : "FALLBACK");
    }
  }
  return r.ok;
}

bool RecompileComputeSpirv(const u32* cs_code,
                           u32 num_thread_x,
                           u32 num_thread_y,
                           u32 num_thread_z,
                           u32 user_sgpr,
                           u32 tgid_enable,
                           u32 lds_dwords,
                           u64 native_mask,
                           u64 uint_mask,
                           u64 shared_mask,
                           RecompiledCs& r) {
  if (!cs_code)
    return false;
  const Program program = DecodeShader(cs_code, 2048);
  MaybeDumpByAddr("CS", cs_code, program);
  const bool dbg = ShaderDebugEnabled();
  Translator t;
  t.program_base = reinterpret_cast<u64>(cs_code);
  RecompiledCs tmp;
  ResetUnsupported();
  if (dbg)
    AuditBegin("cs", cs_code, program);
  const bool cs_ok =
      TranslateCs(program, num_thread_x, num_thread_y, num_thread_z, user_sgpr,
                  tgid_enable, lds_dwords, native_mask, uint_mask, shared_mask,
                  tmp, t) &&
      !HadUnsupported();
  base::Vector<u32> spv_bin;
  if (cs_ok)
    spv_bin = t.m.Assemble();
  if (dbg) {
    if (cs_ok)
      AuditPlan(PlanSummaryCs(tmp));
    else
      AuditDecline("cs translation rejected (dispatch will be skipped)");
    AuditEnd(cs_ok ? &spv_bin : nullptr);
  }
  if (!cs_ok) {
    static base::HashSet<u64> listed;
    static const guest::SessionReset reset_listed(
        [] { guest::ResetResource(listed); });
    const u64 addr = reinterpret_cast<u64>(cs_code);
    if (listed.size() < 16 && listed.insert(addr).second) {
      BASE_LOGW("gcnspv", "CS rejected @{:#x}: {}", addr,
                UnsupportedOps().empty() ? "no resources or untranslatable"
                                         : UnsupportedOps().c_str());
      Disassemble(cs_code, 2048, "rejected CS");
    }
    return false;
  }
  base::String err;
  if (NoOpt()) {
    if (!spirv::Validate(spv_bin, &err)) {
      BASE_LOGI("gcnspv", "CS invalid @{}: {}",
                static_cast<const void*>(cs_code), err.c_str());
      return false;
    }
    tmp.spirv = spv_bin;
  } else if (!spirv::Finalize(spv_bin, &tmp.spirv, &err)) {
    BASE_LOGI("gcnspv", "CS invalid @{}: {}", static_cast<const void*>(cs_code),
              err.c_str());
    return false;
  }
  if (tmp.spirv.empty())
    return false;
  tmp.ok = true;
  r = base::move(tmp);
  return true;
}

}  // namespace gpu::gcn

#endif
