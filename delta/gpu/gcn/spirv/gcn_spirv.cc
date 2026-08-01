/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> SPIR-V translator: per-instruction dispatch, control-flow lowering,
 * and the per-stage (VS/PS/CS) drivers. The ALU and memory emitters live in
 * translate_alu.cc / translate_mem.cc; the shared context in translator.h.
 */

#include "gpu/gcn/spirv/gcn_spirv.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
// Backend disabled at build time (SPIRV-Tools/Headers unavailable). There is
// no other recompiler: every recompile declines and the affected
// draws/dispatches are skipped.
namespace gpu::gcn {
bool RecompileSpirv(const uint32_t*,
                     const uint32_t*,
                     const uint32_t*,
                     const uint32_t*,
                     uint32_t,
                     uint32_t,
                     Recompiled&) {
  return false;
}
bool RecompileComputeSpirv(const uint32_t*,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           RecompiledCs&) {
  return false;
}
}  // namespace gpu::gcn
#else

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <string>
#include <unordered_set>

#include "gpu/gcn/gcn_audit.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/gcn/spirv/translator.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(uint32_t, kCfgMaxIter, "DELTA_GPU_CFG_MAXITER", 16384);
DELTA_OPTION(uint64_t, kShDisAddr, "DELTA_GPU_SHDIS_ADDR", 0);
DELTA_OPTION(bool, kGpuNokill, "DELTA_GPU_NOKILL", false);
DELTA_OPTION(bool, kGpuPswhite, "DELTA_GPU_PSWHITE", false);
DELTA_OPTION(int, kGpuPstex, "DELTA_GPU_PSTEX", 0);
DELTA_OPTION(float, kGpuPstexScale, "DELTA_GPU_PSTEXSCALE", 1.f);
DELTA_OPTION(bool, kGpuShdis, "DELTA_GPU_SHDIS", false);
DELTA_OPTION(bool, kGpuShtrace, "DELTA_GPU_SHTRACE", false);
DELTA_OPTION(bool, kGpuSpirv, "DELTA_GPU_SPIRV", false);
DELTA_OPTION(bool, kGpuSpirvCfg, "DELTA_GPU_SPIRV_CFG", false);
DELTA_OPTION(bool, kGpuSpirvNoopt, "DELTA_GPU_SPIRV_NOOPT", false);
DELTA_OPTION(bool, kGpuVsflipz, "DELTA_GPU_VSFLIPZ", false);
DELTA_OPTION(bool, kGpuVsfull, "DELTA_GPU_VSFULL", false);
}  // namespace

namespace gpu::gcn {

namespace {
thread_local bool g_had_unsupported = false;
thread_local std::string g_unsupported_ops;
}

void ResetUnsupported() {
  g_had_unsupported = false;
  g_unsupported_ops.clear();
}

bool HadUnsupported() {
  return g_had_unsupported;
}

// Which ops made the shader currently being translated unsupported. The
// warn-once dedup hides everything after the first shader that hit a given op,
// so a later shader's rejection is otherwise unattributable.
const std::string& UnsupportedOps() {
  return g_unsupported_ops;
}

bool TraceEnabled() {
  return kGpuShtrace;
}

void WarnUnsupported(const char* enc, uint32_t op, uint32_t w0, uint32_t w1) {
  g_had_unsupported = true;
  {
    char one[64];
    std::snprintf(one, sizeof(one), "%s:%#x", enc, op);
    if (g_unsupported_ops.find(one) == std::string::npos) {
      if (!g_unsupported_ops.empty())
        g_unsupported_ops += ' ';
      g_unsupported_ops += one;
    }
  }
  // Every event feeds the audit (per-shader, per-pc attribution); the
  // warn-once dedup below only limits the stderr flood.
  AuditNote(enc, op);
  static std::unordered_set<uint64_t> seen;
  const uint64_t key =
      static_cast<uint64_t>(std::hash<std::string_view>{}(enc)) ^
      (static_cast<uint64_t>(op) << 40);
  if (seen.size() > 512 || !seen.insert(key).second)
    return;
  std::fprintf(stderr,
               "[gcnspv] UNSUPPORTED %s op=%#x (w0=%#x w1=%#x) -> rejected\n",
               enc, op, w0, w1);
}

// ---- stage-io helpers -------------------------------------------------------
Id PsInputVar(Translator& t, StageContext& sc, uint32_t attr) {
  auto it = sc.in_vars.find(attr);
  if (it != sc.in_vars.end())
    return it->second;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                            spv::StorageClass::Input);
  t.m.Decorate(v, spv::Decoration::Location, {attr});
  if (sc.flat_attrs && sc.flat_attrs->count(attr))
    t.m.Decorate(v, spv::Decoration::Flat);
  t.m.Name(v, "in_attr" + std::to_string(attr));
  sc.iface->push_back(v);
  sc.in_vars[attr] = v;
  return v;
}

Id VsParamOut(Translator& t, StageContext& sc, uint32_t p) {
  auto it = sc.param_outs.find(p);
  if (it != sc.param_outs.end())
    return it->second;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::Location, {p});
  if (sc.flat_attrs && sc.flat_attrs->count(p))
    t.m.Decorate(v, spv::Decoration::Flat);
  t.m.Name(v, "out_param" + std::to_string(p));
  sc.iface->push_back(v);
  sc.param_outs[p] = v;
  return v;
}

// Lazily declare the PS color output for an MRT target (location == target)
// and record it in ps_mrt_mask so the renderer masks unwritten attachments.
Id PsColorOut(Translator& t, StageContext& sc, uint32_t target) {
  if (sc.color_outs[target])
    return sc.color_outs[target];
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::Location, {target});
  t.m.Name(v, "mrt" + std::to_string(target));
  sc.iface->push_back(v);
  sc.color_outs[target] = v;
  sc.r->ps_mrt_mask |= static_cast<uint8_t>(1u << target);
  return v;
}

// Lazily declare gl_FragDepth for the MRTZ export (adds DepthReplacing).
Id PsDepthOut(Translator& t, StageContext& sc) {
  if (sc.depth_out)
    return sc.depth_out;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_f),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::FragDepth)});
  sc.iface->push_back(v);
  sc.depth_out = v;
  if (sc.main_fn)
    t.m.ExecMode(sc.main_fn, spv::ExecutionMode::DepthReplacing);
  return v;
}

namespace {

// Vertex attribute recovered from the Gnm fetch shader: an s_load_dwordx4 of
// the V# (from the vertex-buffer table a user SGPR points at) + a
// buffer_load_format into the destination VGPRs, one per attribute in
// semantic order.
struct FetchAttr {
  uint32_t semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
  bool direct_fetch = false;
  uint32_t pc = ~0u;
  uint32_t dfmt = 0, nfmt = 0;  // MTBUF only: format from the instruction
};

std::vector<FetchAttr> ParseFetch(uint64_t fetch_addr) {
  std::vector<FetchAttr> out;
  if (!InGuest(fetch_addr))
    return out;
  const auto* code = reinterpret_cast<const uint32_t*>(fetch_addr);
  const Program program = Decode(code, 256);
  struct Load {
    uint32_t table_sgpr, dword_off;
  };
  std::unordered_map<uint32_t, Load> loads;
  uint32_t semantic = 0;
  for (const Inst& inst : program) {
    const uint32_t w = inst.raw[0];
    if (inst.enc == Enc::kSop1 && (inst.opcode == 0x20 || inst.opcode == 0x21))
      break;
    if (inst.enc == Enc::kSopp && inst.opcode == 1)
      break;
    if (inst.enc == Enc::kSmrd && inst.opcode == 0x02) {
      const uint32_t sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
      loads[sdst] = {sbase * 2u, w & 0xFF};
    } else if (inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf) {
      const uint32_t w1 = inst.raw[1];
      const uint32_t vdata = (w1 >> 8) & 0xFF;
      const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
      const uint32_t nc = (inst.opcode & 3) + 1;
      const bool typed = inst.enc == Enc::kMtbuf;
      const auto it = loads.find(srsrc);
      const uint32_t tbl = it != loads.end() ? it->second.table_sgpr : 0;
      const uint32_t off = it != loads.end() ? it->second.dword_off : 0;
      out.push_back({semantic, nc, vdata, tbl, off, false, ~0u,
                     typed ? (w >> 19) & 0xF : 0, typed ? (w >> 23) & 0x7 : 0});
      semantic++;
    }
  }
  return out;
}

// ---- per-instruction dispatch ----------------------------------------------
// Emit one non-terminator instruction (branches are handled by the CFG driver).
void EmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
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
      if (inst.opcode == 0x0a && sc.is_cs) {  // s_barrier
        // ControlBarrier(Workgroup, Workgroup, AcquireRelease|WorkgroupMemory)
        t.m.EmitVoid(spv::Op::OpControlBarrier,
                     {t.U32(2), t.U32(2), t.U32(0x108)});
      } else if (inst.opcode >= 0x16 && inst.opcode <= 0x19 &&
                 static_cast<int16_t>(w & 0xffff) == 0) {
        // Debug-state branch to the next instruction: both outcomes fall
        // through.
      } else if (inst.opcode != 0x00 && inst.opcode != 0x01 &&
                 inst.opcode != 0x02 &&
                 !(inst.opcode >= 0x04 && inst.opcode <= 0x0a) &&
                 inst.opcode != 0x0c) {
        WarnUnsupported("sopp", inst.opcode, w, w1);
      }
      break;  // s_nop / s_waitcnt / hints: no-ops in this model
    case Enc::kSmrd:
      if (sc.is_cs)
        EmitCsSmrd(t, inst, sc);
      else
        EmitCbufSmrd(t, inst, sc.cbuf_bind);
      break;
    case Enc::kVop1: {
      if (inst.isa == IsaMode::kNeo && EmitNeoVop1(t, inst))
        break;
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF,
                     src0 = w & 0x1FF;
      EmitVop1(t, op, vdst, t.SrcF(src0, inst.literal));
      break;
    }
    case Enc::kVop2: {
      if (inst.isa == IsaMode::kNeo && EmitNeoVop2(t, inst))
        break;
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const uint32_t vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      const uint32_t src1 = op == 0x01 || op == 0x02 ? vsrc1 : 256 + vsrc1;
      EmitVop2(t, op, vdst, t.SrcF(src0, inst.literal),
               t.SrcF(src1, inst.literal), inst.literal);
      break;
    }
    case Enc::kVop3: {
      if (inst.isa == IsaMode::kNeo && EmitNeoVop3(t, inst))
        break;
      const uint32_t op = inst.opcode, vdst = w & 0xFF;
      const bool vop3b = IsVop3b(op);
      const uint32_t sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const uint32_t abs = vop3b ? 0 : ((w >> 8) & 7);
      const bool clamp = !vop3b && ((w >> 11) & 1);
      const uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const uint32_t s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      const uint32_t omod = (w1 >> 27) & 3;
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
      EmitVop3(t, op, vdst, source0, t.SrcRawHi(s0, inst.literal, op == 0x163),
               t.SrcF(s1, inst.literal, neg & 2, abs & 2),
               t.SrcF(s2, inst.literal, neg & 4, abs & 4),
               t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp, omod);
      break;
    }
    case Enc::kVop3p:
      if (!EmitNeoVop3p(t, inst))
        WarnUnsupported("vop3p", inst.opcode, w, w1);
      break;
    case Enc::kVopc: {
      const uint32_t op = inst.opcode;
      const uint32_t vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      if (inst.isa == IsaMode::kNeo &&
          EmitNeoVopc(t, op, 106, src0, 256 + vsrc1, inst.literal))
        break;
      EmitVopc(t, op, t.SrcF(src0, inst.literal),
               t.SrcF(256 + vsrc1, inst.literal), t.SrcRaw(src0, inst.literal),
               t.SrcRaw(256 + vsrc1, inst.literal));
      break;
    }
    case Enc::kVintrp: {
      if (!sc.is_ps)
        break;
      const uint32_t chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F;
      const uint32_t op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
      if (op == 1 || (op == 2 && (w & 0xFF) == 2)) {
        // Vulkan provides the completed interpolation directly. P2 reads the
        // final value; MOV P0 reads the selected parameter input instead of
        // leaving the destination zero-initialised. (P1 is a no-op here.)
        const Id v = PsInputVar(t, sc, attr);
        const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
        t.SetVgF(vdst,
                 t.m.Load(t.t_f, t.m.AccessChain(p_in_f, v, {t.U32(chan)})));
      }
      break;
    }
    case Enc::kMubuf:
      if (sc.is_cs) {
        EmitCsMubuf(t, inst, sc);
      } else if (!sc.is_ps && sc.direct_vfetch.count(inst.pc)) {
        // Seeded from the vertex-input state instead.
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
      if (sc.is_cs || inst.opcode == 0x35)
        EmitDs(t, inst, sc);
      else
        WarnUnsupported("ds.graphics", inst.opcode, w, w1);
      break;
    case Enc::kMimg:
      if (sc.is_cs)
        EmitCsMimg(t, inst, sc);
      else if (sc.is_ps)
        EmitMimg(t, inst, sc);
      else
        // A VS sampling a texture (displacement, per-vertex lookup) has no
        // model in the graphics path; dropping it silently renders wrong.
        WarnUnsupported("mimg.vs", inst.opcode, w, w1);
      break;
    case Enc::kExp: {
      if (sc.is_cs) {
        WarnUnsupported("exp.cs", (w >> 4) & 0x3F, w, w1);
        sc.cs_unsupported = true;  // no exports in compute
        break;
      }
      const uint32_t en = w & 0xF, target = (w >> 4) & 0x3F;
      const uint32_t compr = (w >> 10) & 1;
      const uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF,
                             (w1 >> 24) & 0xFF};
      if (sc.is_ps) {
        if (target <= 7 && en) {  // MRT0..7; EN=0 is a null export
          sc.wrote_color = true;
          Id col;
          if (compr) {
            // EN pairs up under COMPR (as in the disassembler's OperandsExp):
            // bits 0-1 gate the register with the packed x/y halves, bits 2-3
            // the z/w pair. A disabled pair names no register, not VGPR 0.
            Id c[4];
            for (int i = 0; i < 4; i++)
              c[i] = t.F32(i == 3 ? 1.f : 0.f);
            for (uint32_t p = 0; p < 2; p++) {
              if (!(en & (0x3u << (2 * p))))
                continue;
              const Id pair =
                  t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[p])});
              c[2 * p] = t.m.CompositeExtract(t.t_f, pair, 0);
              c[2 * p + 1] = t.m.CompositeExtract(t.t_f, pair, 1);
            }
            col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
          } else {
            Id c[4];
            for (int i = 0; i < 4; i++)
              c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
            col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
          }
          t.m.Store(PsColorOut(t, sc, target), col);
          // Mark this fragment as having reached a color export, so the
          // discard idiom (control flow branching over the exp) can be
          // lowered to OpKill.
          if (sc.color_written_var)
            t.m.Store(sc.color_written_var, t.U32(1));
        } else if (target == 8 && (en & 1)) {  // MRTZ: depth export
          const Id depth =
              compr ? t.m.CompositeExtract(
                          t.t_f,
                          t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                                      {t.Vg(v[0])}),
                          0)
                    : t.VgF(v[0]);
          t.m.Store(PsDepthOut(t, sc), depth);
        }
      } else {
        if (target == 12) {  // POS0 -> gl_Position
          Id c[4];
          for (int i = 0; i < 4; i++)
            c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
          t.m.Store(sc.pos_out,
                    t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
        } else if (target >= 32 && target <= 63) {  // PARAM0..31
          const uint32_t p = target - 32;
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

// EmitInst wrapper feeding the shader audit (gcn_audit.h): per-instruction
// SPIR-V word counts expose instructions that silently emitted nothing, and
// with DELTA_GPU_SHDUMP each instruction's ops are preceded by an OpLine
// whose line number is the GCN pc (visible in spirv-dis / RenderDoc).
void EmitInstAudited(Translator& t,
                     const Inst& inst,
                     uint32_t index,
                     StageContext& sc) {
  if (!ShaderDebugEnabled()) {
    EmitInst(t, inst, sc);
    return;
  }
  AuditInstBegin(index, inst.pc);
  if (ShaderDumpEnabled()) {
    if (!t.dbg_file)
      t.dbg_file = t.m.String("gcn");
    t.m.Line(t.dbg_file, inst.pc);
  }
  const size_t before = t.m.BodyWords();
  EmitInst(t, inst, sc);
  AuditInstEnd(index, static_cast<uint32_t>(t.m.BodyWords() - before));
}

// ---- control flow: while/switch lowering -----------------------------------
// Branch classification. 0=none, 1=uncond, 2=scc0, 3=scc1, 4=vccz, 5=vccnz,
// 6=execz, 7=execnz, 8=endpgm.
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
  return std::any_of(program.begin(), program.end(), [](const Inst& inst) {
    const int k = BranchKind(inst);
    return k >= 1 && k <= 7;
  });
}

// "Take the branch" condition for a conditional branch kind.
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

// Block leaders under the same rule EmitCfg uses: entry, every branch target,
// the slot after a branch.
std::vector<uint32_t> BlockStarts(const Program& program, uint32_t max_pc) {
  std::vector<uint32_t> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0)
      continue;
    leaders.push_back(inst.pc + inst.size);
    if (k >= 1 && k <= 7) {
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                              static_cast<int32_t>(inst.size) +
                                              simm));
    }
  }
  std::sort(leaders.begin(), leaders.end());
  leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
  std::vector<uint32_t> starts;
  for (uint32_t l : leaders)
    if (l < max_pc)
      starts.push_back(l);
  return starts;
}

}  // namespace

namespace {

// Lower arbitrary control flow (reducible or not) to a while/switch state
// machine over basic blocks. `reachable` (optional, program-index aligned)
// suppresses instructions in dead blocks -- decoded footer padding must not
// influence translation.
void EmitCfg(Translator& t,
             const Program& program,
             StageContext& sc,
             const uint8_t* reachable = nullptr) {
  const uint32_t max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const std::vector<uint32_t> starts = BlockStarts(program, max_pc);
  const uint32_t num_blocks = static_cast<uint32_t>(starts.size());
  const uint32_t kExit = num_blocks;
  const auto block_of = [&](uint32_t pc) -> uint32_t {
    if (pc >= max_pc)
      return kExit;
    uint32_t b = 0;
    for (uint32_t i = 0; i < num_blocks; i++)
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
  std::vector<Id> case_labels(num_blocks);
  for (Id& l : case_labels)
    l = t.m.NewBlock();

  // Runaway guard: one mistranslated branch condition/target leaves the state
  // machine spinning forever, and a single spinning invocation takes the whole
  // VkDevice down (DEVICE_LOST). Cap block-steps per invocation; a capped
  // shader renders wrong, loudly bisectable, instead of killing the device.
  // DELTA_GPU_CFG_MAXITER overrides the cap (0 disables the guard).
  const Id iter_var = kCfgMaxIter
                          ? t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                         t.m.ConstNull(t.t_u))
                          : 0;

  t.SetState(0);
  t.m.Branch(header);
  t.m.OpenBlock(header);
  t.m.LoopMerge(merge, cont);
  t.m.Branch(dispatch);
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
  std::vector<std::pair<uint32_t, Id> > cases;
  for (uint32_t i = 0; i < num_blocks; i++)
    cases.push_back({i, case_labels[i]});
  t.m.Switch(state, exit_blk, cases);  // default (incl. EXIT state) -> exit

  for (uint32_t bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    const uint32_t blk_start = starts[bi];
    const uint32_t blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    uint32_t idx = 0;
    for (const Inst& inst : program) {
      const uint32_t inst_idx = idx++;
      if (inst.pc < blk_start || inst.pc >= blk_end)
        continue;
      if (reachable && !reachable[inst_idx])
        continue;  // dead block/padding
      const int k = BranchKind(inst);
      if (k == 0) {
        EmitInstAudited(t, inst, inst_idx, sc);
        continue;
      }
      // terminator
      const uint32_t fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (k == 8) {  // endpgm
        t.SetState(kExit);
      } else if (k == 1) {  // unconditional
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        t.SetState(block_of(
            static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                  static_cast<int32_t>(inst.size) + simm)));
      } else {  // conditional
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        const uint32_t target = block_of(
            static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                  static_cast<int32_t>(inst.size) + simm));
        t.SetStateId(t.SelectB(BranchTaken(t, k), t.U32(target), t.U32(fall)));
      }
      terminated = true;
      break;
    }
    if (!terminated)
      t.SetState((bi + 1 < num_blocks) ? bi + 1 : kExit);
    t.m.Branch(merge_sel);
  }
  t.m.OpenBlock(exit_blk);
  t.m.Branch(merge);
  t.m.OpenBlock(merge_sel);
  t.m.Branch(cont);
  t.m.OpenBlock(cont);
  t.m.Branch(header);
  t.m.OpenBlock(merge);  // left open; caller emits the stage epilogue here
}

bool ForceCfg() {
  return kGpuSpirvCfg;
}

// Emit a stage body: branchy shaders take the CFG (while-switch) path so
// their control flow (the GCN alpha-test/discard idiom, conditional shading)
// is honoured; single-basic-block shaders emit the same instruction stream
// straight-line.
void EmitBody(Translator& t,
              const Program& program,
              StageContext& sc,
              const uint8_t* reachable) {
  t.SeedExec();
  t.predicate_vector = true;
  if (ForceCfg() || HasControlFlow(program)) {
    EmitCfg(t, program, sc, reachable);
    return;
  }
  uint32_t index = 0;
  for (const Inst& inst : program) {
    const uint32_t inst_idx = index++;
    if (reachable && !reachable[inst_idx])
      continue;
    if (inst.enc == Enc::kSopp && inst.opcode == 1)
      break;  // s_endpgm
    EmitInstAudited(t, inst, inst_idx, sc);
  }
}

bool UsesDsSwizzle(const Program& program, const uint8_t* reachable) {
  for (uint32_t i = 0; i < program.size(); i++)
    if ((!reachable || reachable[i]) && program[i].enc == Enc::kDs &&
        program[i].opcode == 0x35)
      return true;
  return false;
}

void EnableDsSwizzle(Translator& t, StageContext& sc, std::vector<Id>& iface) {
  t.m.Capability(spv::Capability::GroupNonUniform);
  t.m.Capability(spv::Capability::GroupNonUniformShuffle);
  sc.subgroup_local_id =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                   spv::StorageClass::Input);
  t.m.Decorate(
      sc.subgroup_local_id, spv::Decoration::BuiltIn,
      {static_cast<uint32_t>(spv::BuiltIn::SubgroupLocalInvocationId)});
  t.m.Decorate(sc.subgroup_local_id, spv::Decoration::Flat);
  iface.push_back(sc.subgroup_local_id);
}

Id DeclareUserData(Translator& t) {
  const Id words = t.m.TypeArray(t.t_u, 16);
  t.m.Decorate(words, spv::Decoration::ArrayStride, {4});
  const Id block = t.m.TypeStruct({words});
  t.m.Decorate(block, spv::Decoration::Block);
  t.m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
  const Id v =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, block),
                   spv::StorageClass::PushConstant);
  t.m.Name(v, "user_data");
  return v;
}

void SeedUserData(Translator& t, Id user_data) {
  const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (uint32_t i = 0; i < 16; i++)
    t.SetSg(i, t.m.Load(t.t_u,
                        t.m.AccessChain(p_u, user_data, {t.U32(0), t.U32(i)})));
}

// ---- VS ---------------------------------------------------------------------
bool TranslateVs(const Program& program,
                 const uint32_t* vs_user_data,
                 const std::unordered_set<uint32_t>& flat_attrs,
                 Recompiled& r,
                 Translator& t) {
  const uint64_t fetch =
      (static_cast<uint64_t>(vs_user_data[1] & 0xFFFF) << 32) | vs_user_data[0];
  const std::vector<uint8_t> reachable = ComputeReachability(program);
  std::vector<FetchAttr> direct_attrs;
  for (uint32_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    // MTBUF addresses the buffer exactly as MUBUF does and its load opcodes
    // count components the same way, so a typed fetch of a vertex attribute
    // reaches the vertex-input path unchanged except for its format, which the
    // instruction carries and the attribute has to take along.
    if (!reachable[i] ||
        (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf) ||
        inst.opcode > 0x03)
      continue;
    const uint32_t w = inst.raw[0], w1 = inst.raw[1];
    const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
    const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
    const uint32_t soffset = (w1 >> 24) & 0xFF;
    if (!idxen || offen || soffset != 128)
      continue;
    const bool typed = inst.enc == Enc::kMtbuf;
    direct_attrs.push_back({static_cast<uint32_t>(direct_attrs.size()),
                            inst.opcode + 1, (w1 >> 8) & 0xFF, srsrc, 0, true,
                            inst.pc, typed ? (w >> 19) & 0xF : 0,
                            typed ? (w >> 23) & 0x7 : 0});
  }
  // A shader that calls a fetch shader states its real vertex layout there. A
  // buffer load in the body can have exactly the same shape as an attribute
  // fetch (indexed, no offset, zero soffset) and still be reading per-instance
  // data: P.T. gathers three consecutive 16-byte rows per bone that way, which
  // otherwise displaces every real attribute.
  std::vector<FetchAttr> attrs = ParseFetch(fetch);
  if (attrs.empty())
    attrs = std::move(direct_attrs);
  t.InitTypes();

  std::vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);

  StageContext sc;
  sc.r = &r;
  sc.iface = &iface;
  sc.pos_out = pos_out;
  sc.flat_attrs = &flat_attrs;
  for (const FetchAttr& attr : attrs)
    if (attr.direct_fetch)
      sc.direct_vfetch.insert(attr.pc);
  if (UsesDsSwizzle(program, reachable.data()))
    EnableDsSwizzle(t, sc, iface);
  const Id user_data = DeclareUserData(t);

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  SeedUserData(t, user_data);

  Id debug_vertex_index = 0;
  if (kGpuVsfull) {
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    debug_vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(debug_vertex_index, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::VertexIndex)});
    iface.push_back(debug_vertex_index);
  }

  if (attrs.empty() || !sc.direct_vfetch.empty()) {
    // A procedural or direct-fetch VS receives VertexID in v0 before its main
    // instruction stream. Attribute loads below overwrite their destination
    // VGPRs just as the direct MUBUF instructions would.
    // On GFX6-8, InstanceID/StepRate0 enters in v1 and raw InstanceID in v3.
    // Seed those ABI inputs from Vulkan's draw built-ins.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/be00f53d4d50b87a87f83e8fa243b77e614eb0b8/src/gallium/drivers/radeonsi/gfx/si_state_shaders.cpp#L269-307
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    Id vertex_index = debug_vertex_index;
    if (!vertex_index) {
      vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
      t.m.Decorate(vertex_index, spv::Decoration::BuiltIn,
                   {static_cast<uint32_t>(spv::BuiltIn::VertexIndex)});
      iface.push_back(vertex_index);
    }
    const Id instance_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(instance_index, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::InstanceIndex)});
    iface.push_back(instance_index);
    t.SetVg(0, t.m.Load(t.t_u, vertex_index));
    const Id instance = t.m.Load(t.t_u, instance_index);
    t.SetVg(1, instance);
    t.SetVg(3, instance);
  }

  // Seed destination VGPRs from fetched vertex attributes when present.
  for (const FetchAttr& a : attrs) {
    const Id comp_ty = a.num_comps == 1   ? t.t_f
                       : a.num_comps == 2 ? t.t_v2
                       : a.num_comps == 3 ? t.t_v3
                                          : t.t_v4;
    const Id in_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, comp_ty),
                     spv::StorageClass::Input);
    t.m.Decorate(in_var, spv::Decoration::Location, {a.semantic});
    t.m.Name(in_var, "v_attr" + std::to_string(a.semantic));
    iface.push_back(in_var);
    const Id val = t.m.Load(comp_ty, in_var);
    // DELTA_GPU_VSFLIPZ: negate the z of the position attribute (semantic 0,
    // >= 3 comps) -- a projection-convention diagnostic. Default off.
    for (uint32_t c = 0; c < a.num_comps; c++) {
      Id comp = a.num_comps == 1 ? val : t.m.CompositeExtract(t.t_f, val, c);
      if (kGpuVsflipz && a.semantic == 0 && c == 2 && a.num_comps >= 3)
        comp = t.FNeg(comp);
      t.SetVgF(a.dest_vgpr + c, comp);
    }
    r.attrs.push_back({a.semantic, a.num_comps, a.table_sgpr, a.dword_off,
                       a.direct_fetch, 0, a.pc, 0, a.dfmt, a.nfmt});
  }

  if (!PlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind, reachable.data()))
    return false;
  // Raw buffer reads left over once the vertex-input state has claimed the
  // direct fetches: the shader indexes them itself, so they become storage
  // buffers the renderer resolves per draw.
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

  // GL clip space (z in [-w,w]) -> Vulkan (z in [0,w]): z = (z + w) * 0.5.
  const Id p_out_f = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
  const Id z_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(2)});
  const Id w_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(3)});
  const Id z = t.m.Load(t.t_f, z_ptr), wv = t.m.Load(t.t_f, w_ptr);
  t.m.Store(z_ptr, t.FMul(t.FAdd(z, wv), t.F32(0.5f)));

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Vertex, main_fn, "main", iface);
  return true;
}

// ---- PS ---------------------------------------------------------------------
bool TranslatePs(const Program& program,
                  const std::unordered_set<uint32_t>& flat_attrs,
                  uint32_t ps_input_ena,
                  uint32_t tex_3d_mask,
                  Recompiled& r,
                  Translator& t) {
  // Color outputs (PsColorOut) are declared lazily per MRT target (location ==
  // target); PS inputs (PsInputVar) likewise as they are read.
  std::vector<Id> iface;
  const std::vector<uint8_t> reachable = ComputeReachability(program);
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  if (!PlanCbufs(program, r.vs_cbufs.size(), r.ps_cbufs, sc.cbuf_bind,
                 reachable.data()))
    return false;
  // Set 2 is shared with the VS, so PS bindings continue after the VS's.
  PlanGfxBuffers(program, r.vs_bufs.size(), nullptr, r.ps_bufs, sc.gfx_buf_bind,
                 reachable.data());

  // Sampler bindings: one per unique descriptor (shared plan with
  // TrackTextures). More unique samplers than the renderer's set-0 layout
  // provides cannot be expressed -- decline (the draw falls back).
  const MimgBindingPlan mimg_plan = PlanMimgBindings(program, reachable.data());
  if (mimg_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers) {
    WarnUnsupported("mimg.binding-count",
                    static_cast<uint32_t>(mimg_plan.binding_srsrc.size()));
    return false;
  }
  sc.mimg_plan = &mimg_plan;
  sc.tex_3d_mask = tex_3d_mask;
  for (uint32_t i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back({i, mimg_plan.binding_srsrc[i],
                         mimg_plan.binding_storage[i],
                         ((tex_3d_mask >> i) & 1u) != 0});

  if (UsesDsSwizzle(program, reachable.data()))
    EnableDsSwizzle(t, sc, iface);
  const Id user_data = DeclareUserData(t);
  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data);
  SeedPsInputVgprs(t, ps_input_ena, iface);

  // A PS with no color export writes nothing to the color targets (hardware
  // semantics: only exports write; e.g. depth-only or buffer-store passes).
  // ps_mrt_mask stays 0 and the renderer masks every color attachment.
  bool has_color_export = false;
  for (uint32_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (reachable[i] && inst.enc == Enc::kExp &&
        ((inst.raw[0] >> 4) & 0x3F) <= 7 && (inst.raw[0] & 0xF)) {
      has_color_export = true;
      break;
    }
  }

  const bool cfg = ForceCfg() || HasControlFlow(program);
  if (cfg && has_color_export) {
    // Default MRT0 to transparent so a fragment that never reaches an export
    // leaves a defined value even if the discard lowering is bypassed.
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(0.f)}));
    sc.color_written_var = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                        t.m.ConstNull(t.t_u));
  }
  if (!kGpuPswhite)
    EmitBody(t, program, sc, reachable.data());

  if (sc.wrote_color && sc.color_written_var) {
    // GCN alpha-test/kill idiom (CFG path): control flow branches over the
    // color export for failing fragments (e.g. s_cmp + s_cbranch_scc0 ->
    // s_endpgm). Discard those (OpKill) instead of leaving the output
    // undefined. DELTA_GPU_NOKILL skips the discard as a diagnostic.
    if (!kGpuNokill) {
      const Id wrote = t.IsNonZero(t.m.Load(t.t_u, sc.color_written_var));
      const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
      t.m.SelectionMerge(after_kill);
      t.m.BranchConditional(wrote, after_kill, kill_blk);
      t.m.OpenBlock(kill_blk);
      t.m.Kill();
      t.m.OpenBlock(after_kill);
    }
  }

  // DELTA_GPU_PSTEX: export a sampled texel instead of the shader's own
  // colour maths. PSWHITE proves the geometry/target/blend path; this separates
  // "the sample reads zero" from "the maths after it is wrong".
  if (kGpuPstex != 0 && has_color_export && t.last_texel)
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.CompositeConstruct(
                  t.t_v4,
                  {t.FMul(t.m.CompositeExtract(t.t_f, t.last_texel, 0),
                          t.F32(kGpuPstexScale)),
                   t.FMul(t.m.CompositeExtract(t.t_f, t.last_texel, 1),
                          t.F32(kGpuPstexScale)),
                   t.FMul(t.m.CompositeExtract(t.t_f, t.last_texel, 2),
                          t.F32(kGpuPstexScale)),
                   t.F32(1.f)}));

  // DELTA_GPU_PSWHITE: isolate VS/rasterization from fragment color math.
  if (kGpuPswhite && has_color_export)
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(1.f), t.F32(1.f), t.F32(1.f), t.F32(1.f)}));

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, sc.main_fn, "main", iface);
  t.m.ExecMode(sc.main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

// GCN permits depth-only rasterization with SPI_SHADER_PGM_LO_PS=0. Vulkan
// still requires a fragment stage, so emit an empty one: fixed-function depth
// testing/writes run, no color location is declared.
bool TranslateDepthOnlyPs(Translator& t) {
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, main_fn, "main", {});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

// ---- CS ---------------------------------------------------------------------
bool TranslateCs(const Program& program,
                 uint32_t num_thread_x,
                 uint32_t num_thread_y,
                 uint32_t num_thread_z,
                 uint32_t user_sgpr,
                 uint32_t tgid_enable,
                 uint32_t lds_dwords,
                 RecompiledCs& r,
                 Translator& t) {
  if (program.empty())
    return false;
  StageContext sc;
  sc.is_cs = true;
  // RSRC2.LDS_SIZE is in 128-dword granules.
  sc.lds_dwords = lds_dwords * 128;
  // Footer-bounded decode keeps blocks reached only after an early-out
  // s_endpgm, but also picks up dead padding between the real code and the
  // OrbShdr footer -- only reachable instructions may influence translation.
  const std::vector<uint8_t> reachable = ComputeReachability(program);
  if (!PlanCsResources(program, reachable.data(), sc.lds_dwords, r,
                       sc.cs_bind) ||
      r.resources.empty())
    return false;

  bool uses_ds_swizzle = false;
  for (uint32_t i = 0; i < program.size(); i++)
    if (reachable[i] && program[i].enc == Enc::kDs &&
        program[i].opcode == 0x35) {
      uses_ds_swizzle = true;
      break;
    }

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
  // params).
  const Id t_arr16 = t.m.TypeArray(t.t_u, 16);
  t.m.Decorate(t_arr16, spv::Decoration::ArrayStride, {4});
  const Id t_pc = t.m.TypeStruct({t_arr16});
  t.m.Decorate(t_pc, spv::Decoration::Block);
  t.m.MemberDecorate(t_pc, 0, spv::Decoration::Offset, {0});
  const Id pc_var =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, t_pc),
                   spv::StorageClass::PushConstant);
  t.m.Name(pc_var, "user_data");

  // Builtins: gl_LocalInvocationID (-> v0..v2) and gl_WorkGroupID (-> the
  // SGPRs after the user data, per tgid_enable).
  const Id t_uv3 = t.m.TypeVec(t.t_u, 3);
  const Id p_in_v3 = t.m.TypePointer(spv::StorageClass::Input, t_uv3);
  const Id local_id = t.m.Variable(p_in_v3, spv::StorageClass::Input);
  t.m.Decorate(local_id, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::LocalInvocationId)});
  const Id group_id = t.m.Variable(p_in_v3, spv::StorageClass::Input);
  t.m.Decorate(group_id, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::WorkgroupId)});
  std::vector<Id> iface{local_id, group_id};
  if (uses_ds_swizzle) {
    t.m.Capability(spv::Capability::GroupNonUniform);
    t.m.Capability(spv::Capability::GroupNonUniformShuffle);
    sc.subgroup_local_id =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                     spv::StorageClass::Input);
    t.m.Decorate(
        sc.subgroup_local_id, spv::Decoration::BuiltIn,
        {static_cast<uint32_t>(spv::BuiltIn::SubgroupLocalInvocationId)});
    iface.push_back(sc.subgroup_local_id);
  }

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  const Id p_pc_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (uint32_t i = 0; i < 16; i++)  // user data -> s0..s15
    t.SetSg(i, t.m.Load(t.t_u,
                        t.m.AccessChain(p_pc_u, pc_var, {t.U32(0), t.U32(i)})));
  const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
  const auto group_comp = [&](uint32_t c) {
    return t.m.Load(t.t_u, t.m.AccessChain(p_in_u, group_id, {t.U32(c)}));
  };
  uint32_t sg = user_sgpr;
  if ((tgid_enable & 1) && sg < 106)
    t.SetSg(sg++, group_comp(0));
  if ((tgid_enable & 2) && sg < 106)
    t.SetSg(sg++, group_comp(1));
  if ((tgid_enable & 4) && sg < 106)
    t.SetSg(sg++, group_comp(2));
  for (uint32_t c = 0; c < 3; c++)  // local invocation id (tidig) -> v0..v2
    t.SetVg(c, t.m.Load(t.t_u, t.m.AccessChain(p_in_u, local_id, {t.U32(c)})));
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
  std::fprintf(stderr, "[shdis] %s, %zu insts:\n", tag, program.size());
  for (const Inst& inst : program)
    std::fprintf(stderr, "[shdis]  %s\n", DisasmLine(inst).c_str());
}

// One-shot disassembly (DELTA_GPU_SHDIS): for the first branchy shaders, list
// each instruction's encoding + opcode.
void MaybeDumpBranchy(const char* tag, const Program& program) {
  if (!kGpuShdis)
    return;
  static int dumped = 0;
  if (!HasControlFlow(program) || dumped >= 2)
    return;
  dumped++;
  std::fprintf(stderr, "[shdis] (branchy)\n");
  DumpProgram(tag, program);
}

// DELTA_GPU_SHDIS_ADDR=hexaddr: dump the full instruction list of the shader
// whose GCN code lives at that guest address, once, whatever its shape.
void MaybeDumpByAddr(const char* tag,
                     const void* code,
                     const Program& program) {
  if (!kShDisAddr || reinterpret_cast<uint64_t>(code) != kShDisAddr)
    return;
  static bool dumped = false;
  if (dumped)
    return;
  dumped = true;
  std::fprintf(stderr, "[shdis] @%p:\n", code);
  DumpProgram(tag, program);
}

bool NoOpt() {
  // DELTA_GPU_SPIRV_NOOPT: skip the optimize pass (keep the naive
  // memory-backed register SPIR-V). Isolates an emission bug from a spirv-opt
  // mis-promotion.
  return kGpuSpirvNoopt;
}

// Dump-header summaries of the resource plan, so a shader dump is
// self-contained (which user-data slot each cbuffer/texture came from).
std::string PlanSummaryGfx(const Recompiled& r, bool ps) {
  std::string s = ps ? "ps plan:" : "vs plan:";
  if (!ps) {
    s += " attrs=" + std::to_string(r.attrs.size());
  }
  const auto& cbufs = ps ? r.ps_cbufs : r.vs_cbufs;
  const auto& bufs = ps ? r.ps_bufs : r.vs_bufs;
  s += " cbufs:";
  for (const ShaderCbuf& c : cbufs)
    s += " [b" + std::to_string(c.binding) +
         " ud=" + std::to_string(c.ud_sgpr) + (c.pointer ? " ptr" : "") + " " +
         std::to_string(c.num_dwords) + "dw]";
  s += " bufs:";
  for (const ShaderBuffer& b : bufs)
    s += " [b" + std::to_string(b.binding) + " srsrc=s" +
         std::to_string(b.srsrc_sgpr) + " pc=" + std::to_string(b.use_pc) + "]";
  if (ps) {
    s += " texs:";
    for (const ShaderTex& tex : r.ps_texs)
      s += " [t" + std::to_string(tex.binding) +
           " ud=" + std::to_string(tex.ud_sgpr) +
           (tex.storage ? " storage" : "") + (tex.is_3d ? " 3d" : "") + "]";
  }
  return s;
}

std::string PlanSummaryCs(const RecompiledCs& r) {
  std::string s = "cs plan:";
  for (const CsResource& res : r.resources)
    s += " [b" + std::to_string(res.binding) + " s" +
         std::to_string(res.base_sgpr) + " kind=" + std::to_string(res.kind) +
         (res.written ? " w" : "") + " min=" + std::to_string(res.min_bytes) +
         "]";
  return s;
}

}  // namespace

// ---- RECTLIST geometry expansion -------------------------------------------
// RECTLIST consumes three post-VS corners and rasterizes the fourth corner as
// a second triangle. Vulkan has no matching input topology, so insert a
// geometry stage that performs the fixed-function expansion without assuming
// anything about the guest VS.
std::vector<uint32_t> EmitRectListGeometry(
    uint32_t num_params,
    const std::unordered_set<uint32_t>& flat_attrs) {
  spirv::Module m;
  m.Capability(spv::Capability::Geometry);
  const Id t_void = m.TypeVoid(), t_f = m.TypeFloat(32),
           t_v4 = m.TypeVec(t_f, 4);
  const Id t_fn = m.TypeFunction(t_void);
  const Id t_in_v4 = m.TypeArray(t_v4, 3);
  const Id p_in_v4 = m.TypePointer(spv::StorageClass::Input, t_v4);
  const Id p_out_v4 = m.TypePointer(spv::StorageClass::Output, t_v4);

  std::vector<Id> iface;
  const Id in_pos = m.Variable(m.TypePointer(spv::StorageClass::Input, t_in_v4),
                               spv::StorageClass::Input);
  const Id out_pos = m.Variable(p_out_v4, spv::StorageClass::Output);
  m.Decorate(in_pos, spv::Decoration::BuiltIn,
             {static_cast<uint32_t>(spv::BuiltIn::Position)});
  m.Decorate(out_pos, spv::Decoration::BuiltIn,
             {static_cast<uint32_t>(spv::BuiltIn::Position)});
  iface.push_back(in_pos);
  iface.push_back(out_pos);

  std::vector<Id> inputs(num_params), outputs(num_params);
  for (uint32_t p = 0; p < num_params; p++) {
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
  const auto load_input = [&](Id input, uint32_t vertex) {
    return m.Load(t_v4, m.AccessChain(p_in_v4, input, {m.ConstU32(vertex)}));
  };
  const auto fourth_corner = [&](Id input) {
    const Id v0 = load_input(input, 0), v1 = load_input(input, 1);
    const Id v2 = load_input(input, 2);
    return m.Emit(spv::Op::OpFSub, t_v4,
                  {m.Emit(spv::Op::OpFAdd, t_v4, {v1, v2}), v0});
  };
  const Id pos3 = fourth_corner(in_pos);
  std::vector<Id> param3(num_params);
  for (uint32_t p = 0; p < num_params; p++)
    if (!flat_attrs.count(p))
      param3[p] = fourth_corner(inputs[p]);

  for (uint32_t vertex = 0; vertex < 4; vertex++) {
    m.Store(out_pos, vertex < 3 ? load_input(in_pos, vertex) : pos3);
    for (uint32_t p = 0; p < num_params; p++) {
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
  m.ExecMode(main_fn, spv::ExecutionMode::Triangles);
  m.ExecMode(main_fn, spv::ExecutionMode::OutputTriangleStrip);
  m.ExecMode(main_fn, spv::ExecutionMode::OutputVertices, {4});
  return m.Assemble();
}

// ---- entry points -----------------------------------------------------------
bool RecompileSpirv(const uint32_t* vs_code,
                     const uint32_t* ps_code,
                     const uint32_t* vs_user_data,
                     const uint32_t* ps_user_data,
                     uint32_t ps_input_ena,
                     uint32_t tex_3d_mask,
                     Recompiled& r) {
  if (!vs_code || !vs_user_data || !ps_user_data)
    return false;

  // Decode each stage exactly once; every later step works on these programs.
  const Program vs_program = DecodeShader(vs_code, 4096);
  const Program ps_program = ps_code ? DecodeShader(ps_code, 4096) : Program{};
  MaybeDumpBranchy("VS", vs_program);
  if (!ps_program.empty())
    MaybeDumpBranchy("PS", ps_program);
  MaybeDumpByAddr("VS", vs_code, vs_program);
  if (!ps_program.empty())
    MaybeDumpByAddr("PS", ps_code, ps_program);

  // V_INTERP_MOV P0 reads a per-primitive parameter rather than a smoothly
  // interpolated value: represent those locations as flat varyings in both
  // stages.
  std::unordered_set<uint32_t> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 &&
        (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  // VS and PS are separate SPIR-V modules.
  const bool dbg = ShaderDebugEnabled();
  Translator tv;
  ResetUnsupported();
  if (dbg)
    AuditBegin("vs", vs_code, vs_program);
  const bool vs_ok = TranslateVs(vs_program, vs_user_data, flat_attrs, r, tv) &&
                     !HadUnsupported();
  std::vector<uint32_t> vs;
  if (vs_ok)
    vs = tv.m.Assemble();
  if (dbg) {
    if (vs_ok)
      AuditPlan(PlanSummaryGfx(r, /*ps=*/false));
    else
      AuditDecline("vs translation rejected");
    AuditEnd(vs_ok ? &vs : nullptr);
  }
  if (!vs_ok) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] VS translation rejected @%p\n",
                   static_cast<const void*>(vs_code));
    return false;
  }

  Translator tp;
  tp.InitTypes();
  ResetUnsupported();
  if (dbg && ps_code)
    AuditBegin("ps", ps_code, ps_program);
  const bool ps_ok = (ps_code ? TranslatePs(ps_program, flat_attrs,
                                             ps_input_ena, tex_3d_mask, r, tp)
                               : TranslateDepthOnlyPs(tp)) &&
                     !HadUnsupported();
  std::vector<uint32_t> ps;
  if (ps_ok)
    ps = tp.m.Assemble();
  if (dbg && ps_code) {
    if (ps_ok)
      AuditPlan(PlanSummaryGfx(r, /*ps=*/true));
    else
      AuditDecline("ps translation rejected");
    AuditEnd(ps_ok ? &ps : nullptr);
  }
  if (!ps_ok) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] PS translation rejected @%p\n",
                   static_cast<const void*>(ps_code));
    return false;
  }

  const std::vector<uint32_t> gs =
      EmitRectListGeometry(r.num_params, flat_attrs);
  // A module the translator emitted but the validator rejects is a translator
  // bug (wrong codegen, not a guest gap): always loud.
  std::string err;
  if (!spirv::Validate(vs, &err)) {
    std::fprintf(stderr, "[gcnspv] VS invalid @%p: %s\n",
                 static_cast<const void*>(vs_code), err.c_str());
    return false;
  }
  if (!spirv::Validate(ps, &err)) {
    std::fprintf(stderr, "[gcnspv] PS invalid @%p: %s\n",
                 static_cast<const void*>(ps_code), err.c_str());
    return false;
  }
  if (!spirv::Validate(gs, &err)) {
    std::fprintf(stderr, "[gcnspv] RECTLIST GS invalid: %s\n", err.c_str());
    return false;
  }
  r.vs_spirv = NoOpt() ? vs : spirv::Optimize(vs);
  r.gs_spirv = NoOpt() ? gs : spirv::Optimize(gs);
  r.fs_spirv = NoOpt() ? ps : spirv::Optimize(ps);
  r.ok = !r.vs_spirv.empty() && !r.gs_spirv.empty() && !r.fs_spirv.empty();

  // Tally (DELTA_GPU_SPIRV): how many shaders the backend accepted vs had to
  // decline, and how many used the CFG path.
  if (kGpuSpirv) {
    static int ok_count = 0, cfg_count = 0, logged = 0;
    if (r.ok)
      ok_count++;
    if (HasControlFlow(vs_program) || HasControlFlow(ps_program))
      cfg_count++;
    if (logged < 12) {
      logged++;
      std::fprintf(stderr,
                   "[gcnspv] recompiled ok=%d (cfg-shaders=%d) this=%s\n",
                   ok_count, cfg_count, r.ok ? "spirv" : "FALLBACK");
    }
  }
  return r.ok;
}

bool RecompileComputeSpirv(const uint32_t* cs_code,
                           uint32_t num_thread_x,
                           uint32_t num_thread_y,
                           uint32_t num_thread_z,
                           uint32_t user_sgpr,
                           uint32_t tgid_enable,
                           uint32_t lds_dwords,
                           RecompiledCs& r) {
  if (!cs_code)
    return false;
  const Program program = DecodeShader(cs_code, 2048);
  MaybeDumpByAddr("CS", cs_code, program);
  const bool dbg = ShaderDebugEnabled();
  Translator t;
  RecompiledCs tmp;  // build into a temp so a mid-emit failure leaves r intact
  ResetUnsupported();
  if (dbg)
    AuditBegin("cs", cs_code, program);
  const bool cs_ok =
      TranslateCs(program, num_thread_x, num_thread_y, num_thread_z, user_sgpr,
                  tgid_enable, lds_dwords, tmp, t) &&
      !HadUnsupported();
  std::vector<uint32_t> spv_bin;
  if (cs_ok)
    spv_bin = t.m.Assemble();
  if (dbg) {
    if (cs_ok)
      AuditPlan(PlanSummaryCs(tmp));
    else
      AuditDecline("cs translation rejected (dispatch will be skipped)");
    AuditEnd(cs_ok ? &spv_bin : nullptr);
  }
  if (!cs_ok)
    return false;
  std::string err;
  if (!spirv::Validate(spv_bin, &err)) {
    std::fprintf(stderr, "[gcnspv] CS invalid @%p: %s\n",
                 static_cast<const void*>(cs_code), err.c_str());
    return false;
  }
  tmp.spirv = NoOpt() ? spv_bin : spirv::Optimize(spv_bin);
  if (tmp.spirv.empty())
    return false;
  tmp.ok = true;
  r = std::move(tmp);
  return true;
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
