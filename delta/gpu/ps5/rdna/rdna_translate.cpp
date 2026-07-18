/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) -> SPIR-V translator. Reuses the shared gpu::gcn SPIR-V
 * backend (the register-file model in Translator, the scalar/vector ALU
 * emitters, exports, and constant-buffer plumbing); only the RDNA2-specific
 * per-instruction field decode + opcode remap and the SMEM constant-buffer path
 * live here. Control-flow lowering is a self-contained copy of the GFX7
 * while/switch state machine (it dispatches through the RDNA2 emitter), so the
 * PS4 backend is untouched.
 */

#include "rdna_translate.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::rdna {
gpu::gcn::Recompiled Recompile(const uint32_t*, const uint32_t*, const uint32_t*,
                               const uint32_t*) {
  return {};
}
}  // namespace gpu::rdna
#else

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ps4/gcn/spirv/spv_post.h"
#include "ps4/gcn/spirv/translator.h"
#include "rdna_decode.h"

namespace gpu::rdna {
namespace {

using gpu::gcn::Enc;
using gpu::gcn::Id;
using gpu::gcn::Inst;
using gpu::gcn::Program;
using gpu::gcn::Recompiled;
using gpu::gcn::ShaderCbuf;
using gpu::gcn::StageContext;
using gpu::gcn::Translator;

// Bring-up debug: log decoded exports + recovered vertex fetches so the PS5
// pipeline can be verified against the guest shader. Honors either the shader
// trace knob (DELTA_GPU_SHTRACE) or the AGC command-stream trace (DELTA_AGC_TRACE).
bool ShDbg() {
  static const bool on = std::getenv("DELTA_GPU_SHTRACE") != nullptr ||
                         std::getenv("DELTA_AGC_TRACE") != nullptr;
  return on;
}

// Dwords loaded by an SMEM s_buffer_load / s_load opcode (x1/x2/x4/x8/x16).
uint32_t SmemLoadCount(uint32_t op) {
  switch (op) {
    case 0x00: case 0x08: return 1;
    case 0x01: case 0x09: return 2;
    case 0x02: case 0x0A: return 4;
    case 0x03: case 0x0B: return 8;
    case 0x04: case 0x0C: return 16;
    default: return 0;
  }
}

int32_t SignExt21(uint32_t v) {
  return static_cast<int32_t>(v << 11) >> 11;
}

// RDNA2 VOP2 opcodes that carry different numbers than the GFX7 emitter expects.
// The shared EmitVop2 switch is GFX7-numbered; most RDNA2 VOP2 ops share those
// numbers, but the fused-multiply-add forms moved. Map them onto the equivalent
// GFX7 mac/madmk/madak semantics (the shared emitter treats FMA as MAD, which
// is exact enough for the recompiler's single-precision model).
uint32_t RemapVop2(uint32_t op) {
  switch (op) {
    // RDNA2 renumbered v_cndmask_b32 from GFX7's VOP2 0x00 to 0x01 (0x01 is
    // v_readlane on GFX7); map it back so the shared emitter's VCC-select fires.
    case 0x01: return 0x00;  // v_cndmask_b32
    case 0x2B: return 0x1F;  // v_fmac_f32  -> v_mac_f32  (s0*s1 + dst)
    case 0x2C: return 0x20;  // v_fmamk_f32 -> v_madmk_f32 (s0*K + s1)
    case 0x2D: return 0x21;  // v_fmaak_f32 -> v_madak_f32 (s0*s1 + K)
    default: return op;
  }
}

// ---- SMEM (constant buffers) ------------------------------------------------
// Plan the set-1 UBO bindings a stage's s_buffer_load* ops reference. Each
// distinct V# base SGPR (a user-data dword pair) gets one binding; the renderer
// resolves the live V# at draw time (like the GFX7 PlanCbufs, but decoding the
// RDNA2 SMEM encoding instead of GCN SMRD).
bool RdnaPlanCbufs(const Program& program, uint32_t first_binding,
                   std::vector<ShaderCbuf>& cbufs,
                   std::unordered_map<uint32_t, uint32_t>& bindings) {
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kSmrd) continue;
    const uint32_t op = inst.opcode;
    if (op < 0x08 || op > 0x0C) continue;  // s_buffer_load_dword{,x2,x4,x8,x16}
    const uint32_t sbase = (inst.raw[0] & 0x3F) * 2;  // user-data dword of the V#
    const int32_t off = SignExt21(inst.raw[1] & 0x1FFFFF);
    const uint32_t hi = static_cast<uint32_t>(off < 0 ? 0 : off) / 4 +
                        SmemLoadCount(op);
    auto it = bindings.find(sbase);
    if (it == bindings.end()) {
      const uint32_t binding = first_binding + static_cast<uint32_t>(cbufs.size());
      bindings[sbase] = binding;
      cbufs.push_back({binding, sbase, hi});
    } else {
      for (ShaderCbuf& cb : cbufs)
        if (cb.ud_sgpr == sbase && hi > cb.num_dwords) cb.num_dwords = hi;
    }
  }
  return true;
}

void RdnaEmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t op = inst.opcode;
  const uint32_t sdst = (inst.raw[0] >> 6) & 0x7F;
  const uint32_t sbase = (inst.raw[0] & 0x3F) * 2;
  const int32_t off = SignExt21(inst.raw[1] & 0x1FFFFF);
  if (op >= 0x08 && op <= 0x0C) {  // s_buffer_load*: read from the bound UBO
    auto it = sc.cbuf_bind.find(sbase);
    if (it == sc.cbuf_bind.end()) {
      gpu::gcn::WarnUnsupported("smem.cbuf-unplanned", op, inst.raw[0], inst.raw[1]);
      return;
    }
    const uint32_t dword0 = static_cast<uint32_t>(off < 0 ? 0 : off) / 4;
    const uint32_t n = SmemLoadCount(op);
    for (uint32_t k = 0; k < n; k++)
      t.SetSg(sdst + k, t.CbufDword(it->second, dword0 + k));
    return;
  }
  // s_load_dword* (op 0x00-0x04) load descriptor tables (V#/T#) into SGPRs; the
  // renderer resolves those from user data via resource tracking, so they do not
  // emit into the shader body (matches the GFX7 path).
}

// ---- exports ----------------------------------------------------------------
// gfx10.3 export targets: MRT0..7 = 0..7, MRTZ = 8, NULL = 9, POS0..4 = 12..16,
// PRIM = 20 (NGG connectivity, ignored), PARAM0..31 = 32..63.
void EmitExport(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
  const uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF,
                         (w1 >> 24) & 0xFF};
  if (ShDbg())
    std::fprintf(stderr,
                 "[gcnspv] %s-exp target=%u en=%#x compr=%u done=%u vsrc=%08x\n",
                 sc.is_ps ? "ps" : "vs", target, en, compr, (w >> 11) & 1, w1);
  if (sc.is_ps) {
    if (target <= 7 && en) {  // MRT0..7 (EN=0 is a null export)
      sc.wrote_color = true;
      Id col;
      if (compr) {
        const Id c01 = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])});
        const Id c23 = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[1])});
        col = t.m.VectorShuffle(t.t_v4, c01, c23, {0, 1, 2, 3});
      } else {
        Id c[4];
        for (int i = 0; i < 4; i++)
          c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
        col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
      }
      t.m.Store(gpu::gcn::PsColorOut(t, sc, target), col);
      if (sc.color_written_var) t.m.Store(sc.color_written_var, t.U32(1));
    } else if (target == 8 && (en & 1)) {  // MRTZ: depth export
      const Id depth =
          compr ? t.m.CompositeExtract(
                      t.t_f,
                      t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])}), 0)
                : t.VgF(v[0]);
      t.m.Store(gpu::gcn::PsDepthOut(t, sc), depth);
    }
    return;
  }
  if (target == 12) {  // POS0 -> gl_Position
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
    t.m.Store(sc.pos_out, t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  } else if (target >= 32 && target <= 63) {  // PARAM0..31
    const uint32_t p = target - 32;
    if (p + 1 > sc.max_param) sc.max_param = p + 1;
    const Id out_var = gpu::gcn::VsParamOut(t, sc, p);
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(0.f);
    t.m.Store(out_var, t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  }
  // target == 20 (PRIM) / 9 (NULL): NGG bookkeeping, nothing to emit.
}

// ---- per-instruction dispatch ----------------------------------------------
// Decodes RDNA2 field layouts and calls the shared GFX7 emitters (which take
// pre-decoded operands + a GFX7-canonical opcode). The scalar and VOP1/2/C
// field layouts are identical to GFX7; VOP3 differs only in the opcode-field
// width and the CLAMP bit position (bit 15, not 11); SMEM replaces SMRD.
void RdnaEmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSop1: gpu::gcn::EmitSop1(t, inst); break;
    case Enc::kSop2: gpu::gcn::EmitSop2(t, inst); break;
    case Enc::kSopc: gpu::gcn::EmitSopc(t, inst); break;
    case Enc::kSopk: gpu::gcn::EmitSopk(t, inst); break;
    case Enc::kSopp:
      if (inst.opcode == 0x0A && sc.is_cs)  // s_barrier
        t.m.EmitVoid(spv::Op::OpControlBarrier, {t.U32(2), t.U32(2), t.U32(0x108)});
      break;  // s_nop / s_waitcnt / branches: no-op here (CFG handles branches)
    case Enc::kSmrd: RdnaEmitSmem(t, inst, sc); break;
    case Enc::kVop1: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
      gpu::gcn::EmitVop1(t, op, vdst, t.SrcF(src0, inst.literal));
      break;
    }
    case Enc::kVop2: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const uint32_t vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      const Id s0u = t.SrcRaw(src0, inst.literal);
      const Id s1u = t.SrcRaw(256 + vsrc1, inst.literal);
      // RDNA2-only VOP2 numbers the shared GFX7 emitter would misinterpret: the
      // no-carry integer add/sub forms must NOT write VCC (a later v_cndmask
      // reads it), and v_xnor_b32 sits where GFX7 has v_bfm_b32.
      switch (op) {
        case 0x1E: t.SetVg(vdst, t.Not(t.Xor(s0u, s1u))); break;  // v_xnor_b32
        case 0x25: t.SetVg(vdst, t.Add(s0u, s1u)); break;         // v_add_nc_u32
        case 0x26: t.SetVg(vdst, t.Sub(s0u, s1u)); break;         // v_sub_nc_u32
        case 0x27: t.SetVg(vdst, t.Sub(s1u, s0u)); break;         // v_subrev_nc_u32
        default:
          gpu::gcn::EmitVop2(t, RemapVop2(op), vdst, t.SrcF(src0, inst.literal),
                             t.SrcF(256 + vsrc1, inst.literal), inst.literal);
          break;
      }
      break;
    }
    case Enc::kVop3: {
      uint32_t op = inst.opcode;
      const uint32_t vdst = w & 0xFF;
      if (op & 0x400) {  // VOP3P (packed 16-bit math): not modelled
        gpu::gcn::WarnUnsupported("vop3p", op & 0x3FF, w, w1);
        break;
      }
      // VOP3-form v_cndmask is the VOP2 alias 0x101 on RDNA2 (0x01 renumber);
      // GFX7's explicit-S2 cndmask VOP3 op is 0x100.
      if (op == 0x101) op = 0x100;
      const bool vop3b = gpu::gcn::IsVop3b(op);
      const uint32_t sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const uint32_t abs = vop3b ? 0 : ((w >> 8) & 7);
      const bool clamp = !vop3b && ((w >> 15) & 1);  // RDNA2 CLAMP at bit 15
      const uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const uint32_t s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      gpu::gcn::EmitVop3(t, op, vdst, t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                         t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                         t.SrcF(s2, inst.literal, neg & 4, abs & 4),
                         t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp);
      break;
    }
    case Enc::kVopc: {
      const uint32_t op = inst.opcode;
      const uint32_t vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      gpu::gcn::EmitVopc(t, op, t.SrcF(src0, inst.literal),
                         t.SrcF(256 + vsrc1, inst.literal),
                         t.SrcRaw(src0, inst.literal),
                         t.SrcRaw(256 + vsrc1, inst.literal));
      break;
    }
    case Enc::kVintrp: {
      if (!sc.is_ps) break;
      const uint32_t chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F;
      const uint32_t op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
      if (op == 1 || (op == 2 && (w & 0xFF) == 2)) {
        const Id var = gpu::gcn::PsInputVar(t, sc, attr);
        const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
        t.SetVgF(vdst, t.m.Load(t.t_f, t.m.AccessChain(p_in_f, var, {t.U32(chan)})));
      }
      break;
    }
    case Enc::kExp: EmitExport(t, inst, sc); break;
    case Enc::kMubuf:
      // buffer_load_format_x..xyzw is the inline vertex fetch: TranslateVs lifts
      // it to a Location vertex input and seeds the destination VGPRs before the
      // body runs (see ParseFetchInsts), so nothing is emitted here. Other MUBUF
      // ops (stores / raw dword loads) don't occur in Isaac's VS/PS.
      if (inst.opcode > 0x03)
        gpu::gcn::WarnUnsupported("mubuf.rdna", inst.opcode, w, w1);
      break;
    case Enc::kMimg:
      // TODO(ps5): RDNA2 MIMG sampling (gfx10 T#/S# + NSA addressing).
      gpu::gcn::WarnUnsupported("mimg.rdna", inst.opcode, w, w1);
      break;
    default:
      gpu::gcn::WarnUnsupported("rdna", inst.opcode, w, w1);
      break;
  }
}

// ---- control flow (self-contained copy of the GFX7 while/switch lowering) ---
// 0=none, 1=uncond, 2=scc0, 3=scc1, 4=vccz, 5=vccnz, 6=execz, 7=execnz, 8=endpgm.
int BranchKind(const Inst& inst) {
  if (inst.enc != Enc::kSopp) return 0;
  switch (inst.opcode) {
    case 0x01: return 8;
    case 0x02: return 1;
    case 0x04: return 2;
    case 0x05: return 3;
    case 0x06: return 4;
    case 0x07: return 5;
    case 0x08: return 6;
    case 0x09: return 7;
    default: return 0;
  }
}

bool HasControlFlow(const Program& program) {
  return std::any_of(program.begin(), program.end(), [](const Inst& inst) {
    const int k = BranchKind(inst);
    return k >= 1 && k <= 7;
  });
}

Id BranchTaken(Translator& t, int kind) {
  switch (kind) {
    case 2: return t.IsZero(t.Scc());
    case 3: return t.IsNonZero(t.Scc());
    case 4: return t.IsZero(t.Sg(106));
    case 5: return t.IsNonZero(t.Sg(106));
    case 6: return t.IsZero(t.Exec());
    case 7: return t.IsNonZero(t.Exec());
    default: return t.m.ConstBool(false);
  }
}

std::vector<uint32_t> BlockStarts(const Program& program, uint32_t max_pc) {
  std::vector<uint32_t> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0) continue;
    leaders.push_back(inst.pc + inst.size);
    if (k >= 1 && k <= 7) {
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                              static_cast<int32_t>(inst.size) + simm));
    }
  }
  std::sort(leaders.begin(), leaders.end());
  leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
  std::vector<uint32_t> starts;
  for (uint32_t l : leaders)
    if (l < max_pc) starts.push_back(l);
  return starts;
}

void EmitCfg(Translator& t, const Program& program, StageContext& sc) {
  const uint32_t max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const std::vector<uint32_t> starts = BlockStarts(program, max_pc);
  const uint32_t num_blocks = static_cast<uint32_t>(starts.size());
  const uint32_t kExit = num_blocks;
  const auto block_of = [&](uint32_t pc) -> uint32_t {
    if (pc >= max_pc) return kExit;
    uint32_t b = 0;
    for (uint32_t i = 0; i < num_blocks; i++)
      if (starts[i] <= pc) b = i;
      else break;
    return b;
  };

  const Id header = t.m.NewBlock(), dispatch = t.m.NewBlock();
  const Id merge_sel = t.m.NewBlock();
  const Id cont = t.m.NewBlock(), merge = t.m.NewBlock();
  const Id exit_blk = t.m.NewBlock();
  std::vector<Id> case_labels(num_blocks);
  for (Id& l : case_labels) l = t.m.NewBlock();

  t.SetState(0);
  t.m.Branch(header);
  t.m.OpenBlock(header);
  t.m.LoopMerge(merge, cont);
  t.m.Branch(dispatch);
  t.m.OpenBlock(dispatch);
  const Id state = t.State();
  t.m.SelectionMerge(merge_sel);
  std::vector<std::pair<uint32_t, Id>> cases;
  for (uint32_t i = 0; i < num_blocks; i++) cases.push_back({i, case_labels[i]});
  t.m.Switch(state, exit_blk, cases);

  for (uint32_t bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    const uint32_t blk_start = starts[bi];
    const uint32_t blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < blk_start || inst.pc >= blk_end) continue;
      const int k = BranchKind(inst);
      if (k == 0) {
        RdnaEmitInst(t, inst, sc);
        continue;
      }
      const uint32_t fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (k == 8) {
        t.SetState(kExit);
      } else if (k == 1) {
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        t.SetState(block_of(static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                                  static_cast<int32_t>(inst.size) + simm)));
      } else {
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        const uint32_t target = block_of(static_cast<uint32_t>(
            static_cast<int32_t>(inst.pc) + static_cast<int32_t>(inst.size) + simm));
        t.SetStateId(t.SelectB(BranchTaken(t, k), t.U32(target), t.U32(fall)));
      }
      terminated = true;
      break;
    }
    if (!terminated) t.SetState((bi + 1 < num_blocks) ? bi + 1 : kExit);
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
  static const bool force = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  return force;
}

void EmitBody(Translator& t, const Program& program, StageContext& sc) {
  if (ForceCfg() || HasControlFlow(program)) {
    t.SeedExec();
    EmitCfg(t, program, sc);
    return;
  }
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSopp && inst.opcode == 1) break;  // s_endpgm
    RdnaEmitInst(t, inst, sc);
  }
}

// ---- vertex fetch -----------------------------------------------------------
// Best-effort parse of a Gnm/AGC fetch sub-shader (RDNA2 encoding): s_load_dwordx4
// of a V# table + buffer_load_format into destination VGPRs, one per attribute.
// PS5 NGG shaders often fetch inline instead; then no attributes are recovered
// and the VS seeds from VertexIndex/InstanceIndex (procedural path).
struct FetchAttr {
  uint32_t semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
};

// Scan an instruction stream for the s_load_dwordx4(V# table) + buffer_load_format
// fetch pattern. Works on both a stand-alone fetch sub-shader and the head of an
// NGG vertex program that fetches inline (PS5 AGC frequently inlines the fetch),
// so buffer_load_format never reaches RdnaEmitInst as an unsupported op.
std::vector<FetchAttr> ParseFetchInsts(const Program& insts) {
  std::vector<FetchAttr> out;
  struct Load {
    uint32_t table_sgpr, dword_off;
  };
  std::unordered_map<uint32_t, Load> loads;  // sdst -> table load
  uint32_t sem = 0;
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20) break;  // s_setpc_b64 (return)
    if (in.enc == Enc::kSopp && in.opcode == 1) break;     // s_endpgm
    if (in.enc == Enc::kSmrd && in.opcode == 0x02) {       // s_load_dwordx4 (V# table)
      const uint32_t sdst = (in.raw[0] >> 6) & 0x7F;
      const uint32_t sbase = (in.raw[0] & 0x3F) * 2;
      const int32_t off = SignExt21(in.raw[1] & 0x1FFFFF);
      loads[sdst] = {sbase, static_cast<uint32_t>(off < 0 ? 0 : off) / 4};
    } else if (in.enc == Enc::kMubuf && in.opcode <= 0x03) {  // buffer_load_format_x..xyzw
      const uint32_t vdata = (in.raw[1] >> 8) & 0xFF;
      const uint32_t srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
      const uint32_t nc = (in.opcode & 3) + 1;
      const bool via_load = loads.count(srsrc) != 0;
      // The descriptor is the V# in the buffer_load's own srsrc SGPRs. For the
      // PS5 inline NGG fetch that is the inline V# the guest placed in user data,
      // so the renderer decodes vud[srsrc..srsrc+3] directly (it validates that
      // and falls back to a table pointer only if it isn't a real V#). An s_load
      // into srsrc (via_load) is unreliable to trace and often spurious (e.g. an
      // unrelated s_load_dwordx4 that also targets s0), so ignore it and key off
      // srsrc, which is where the descriptor actually sits at draw time.
      const uint32_t tbl = srsrc;
      const uint32_t doff = 0;
      if (ShDbg())
        std::fprintf(stderr,
                     "[gcnspv] fetch attr sem=%u nc=%u vdst=v%u srsrc=s%u "
                     "table_sgpr=%u dword_off=%u via_load=%d\n",
                     sem, nc, vdata, srsrc, tbl, doff, via_load);
      out.push_back({sem, nc, vdata, tbl, doff});
      sem++;
    }
  }
  return out;
}

std::vector<FetchAttr> ParseFetch(uint64_t fetch_addr) {
  if (!gpu::gcn::InGuest(fetch_addr)) return {};
  const auto* code = reinterpret_cast<const uint32_t*>(fetch_addr);
  return ParseFetchInsts(Decode(code, 256));
}

// ---- VS / PS drivers --------------------------------------------------------
bool TranslateVs(const Program& program, const uint32_t* vs_user_data,
                 const std::unordered_set<uint32_t>& flat_attrs, Recompiled& r,
                 Translator& t) {
  const uint64_t fetch =
      (static_cast<uint64_t>(vs_user_data[1] & 0xFFFF) << 32) | vs_user_data[0];
  // Prefer a stand-alone fetch sub-shader; otherwise recover the fetch that the
  // NGG vertex program does inline (buffer_load_format in its own body). Either
  // way each attribute becomes a Location vertex input (RdnaEmitInst then treats
  // the inline buffer_load_format as a no-op) and the renderer binds the real
  // vertex buffers from r.attrs.
  std::vector<FetchAttr> attrs = ParseFetch(fetch);
  if (attrs.empty()) attrs = ParseFetchInsts(program);
  t.InitTypes();

  std::vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);

  if (attrs.empty()) {  // procedural VS: seed the ABI VGPRs from Vulkan built-ins
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    const Id vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    const Id instance_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(vertex_index, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::VertexIndex)});
    t.m.Decorate(instance_index, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::InstanceIndex)});
    iface.push_back(vertex_index);
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
    iface.push_back(in_var);
    const Id val = t.m.Load(comp_ty, in_var);
    for (uint32_t c = 0; c < a.num_comps; c++) {
      const Id comp = a.num_comps == 1 ? val : t.m.CompositeExtract(t.t_f, val, c);
      t.SetVgF(a.dest_vgpr + c, comp);
    }
    r.attrs.push_back({a.semantic, a.num_comps, a.table_sgpr, a.dword_off});
  }

  StageContext sc;
  sc.r = &r;
  sc.iface = &iface;
  sc.main_fn = main_fn;
  sc.pos_out = pos_out;
  sc.flat_attrs = &flat_attrs;
  if (!RdnaPlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind)) return false;

  EmitBody(t, program, sc);
  r.num_params = sc.max_param;

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

bool TranslatePs(const Program& program,
                 const std::unordered_set<uint32_t>& flat_attrs, Recompiled& r,
                 Translator& t) {
  std::vector<Id> iface;
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  if (!RdnaPlanCbufs(program, static_cast<uint32_t>(r.vs_cbufs.size()),
                     r.ps_cbufs, sc.cbuf_bind))
    return false;

  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);

  const bool has_color_export =
      std::any_of(program.begin(), program.end(), [](const Inst& inst) {
        return inst.enc == Enc::kExp && ((inst.raw[0] >> 4) & 0x3F) <= 7 &&
               (inst.raw[0] & 0xF);
      });

  const bool cfg = ForceCfg() || HasControlFlow(program);
  if (cfg && has_color_export) {
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(t.t_v4, {t.F32(0.f), t.F32(0.f), t.F32(0.f),
                                          t.F32(0.f)}));
    sc.color_written_var =
        t.m.Variable(t.p_priv_u, spv::StorageClass::Private, t.m.ConstNull(t.t_u));
  }
  EmitBody(t, program, sc);

  if (!sc.wrote_color) {
    // No color export at all: opaque white fallback (matches the GFX7 path).
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(t.t_v4, {t.F32(1.f), t.F32(1.f), t.F32(1.f),
                                          t.F32(1.f)}));
  } else if (sc.color_written_var) {
    static const bool no_kill = std::getenv("DELTA_GPU_NOKILL") != nullptr;
    if (!no_kill) {
      const Id wrote = t.IsNonZero(t.m.Load(t.t_u, sc.color_written_var));
      const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
      t.m.SelectionMerge(after_kill);
      t.m.BranchConditional(wrote, after_kill, kill_blk);
      t.m.OpenBlock(kill_blk);
      t.m.Kill();
      t.m.OpenBlock(after_kill);
    }
  }

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

bool NoOpt() {
  static const bool no_opt = std::getenv("DELTA_GPU_SPIRV_NOOPT") != nullptr;
  return no_opt;
}

}  // namespace

Recompiled Recompile(const uint32_t* vs_code, const uint32_t* ps_code,
                     const uint32_t* vs_user_data, const uint32_t* ps_user_data) {
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data) return r;

  const Program vs_program = DecodeShader(vs_code, 4096);
  const Program ps_program = ps_code ? DecodeShader(ps_code, 4096) : Program{};

  // V_INTERP_MOV P0 reads a per-primitive (flat) parameter; represent those
  // locations as flat varyings in both stages.
  std::unordered_set<uint32_t> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 && (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  Translator tv;
  if (!TranslateVs(vs_program, vs_user_data, flat_attrs, r, tv)) return r;
  Translator tp;
  tp.InitTypes();
  if (ps_code ? !TranslatePs(ps_program, flat_attrs, r, tp)
              : !TranslateDepthOnlyPs(tp))
    return r;

  const std::vector<uint32_t> vs = tv.m.Assemble();
  const std::vector<uint32_t> ps = tp.m.Assemble();
  std::string err;
  if (!gpu::gcn::spirv::Validate(vs, &err)) {
    if (gpu::gcn::TraceEnabled())
      std::fprintf(stderr, "[rdna] VS invalid: %s\n", err.c_str());
    return r;
  }
  if (!gpu::gcn::spirv::Validate(ps, &err)) {
    if (gpu::gcn::TraceEnabled())
      std::fprintf(stderr, "[rdna] PS invalid: %s\n", err.c_str());
    return r;
  }
  r.vs_spirv = NoOpt() ? vs : gpu::gcn::spirv::Optimize(vs);
  r.fs_spirv = NoOpt() ? ps : gpu::gcn::spirv::Optimize(ps);
  // No RECTLIST geometry stage for PS5 yet: gfx10.3 triangle-list draws render
  // VS+PS directly (RECTLIST/fullscreen passes are a follow-up).
  r.ok = !r.vs_spirv.empty() && !r.fs_spirv.empty();
  return r;
}

}  // namespace gpu::rdna

#endif  // DELTA_HAVE_SPIRV_BACKEND
