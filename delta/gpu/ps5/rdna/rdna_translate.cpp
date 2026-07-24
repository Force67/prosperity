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
                               const uint32_t*, uint32_t) {
  return {};
}
}  // namespace gpu::rdna
#else

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ps4/gcn/spirv/spv_post.h"
#include "ps4/gcn/spirv/translator.h"
#include "rdna_decode.h"
#include "rdna_resource.h"

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

// A buffer_load_format is a real PER-VERTEX fetch only when it is indexed by the
// vertex-index register (IDXEN with vaddr == v0, the ABI vertex-index VGPR). A
// load with no index, or indexed by a computed/non-v0 register, reads a CONSTANT
// (e.g. Isaac's 2D VS reads its ortho matrix via buffer_load): that must be bound
// as a UBO and read in the shader body, NOT lifted to a vertex input.
bool BufLoadIsVertexFetch(const Inst& in) {
  const bool idxen = (in.raw[0] >> 13) & 1;
  const uint32_t vaddr = in.raw[1] & 0xFF;
  return idxen && vaddr == 0;
}

const char* EncName(Enc e) {
  switch (e) {
    case Enc::kSop1: return "sop1"; case Enc::kSop2: return "sop2";
    case Enc::kSopk: return "sopk"; case Enc::kSopc: return "sopc";
    case Enc::kSopp: return "sopp"; case Enc::kSmrd: return "smem";
    case Enc::kVop1: return "vop1"; case Enc::kVop2: return "vop2";
    case Enc::kVop3: return "vop3"; case Enc::kVopc: return "vopc";
    case Enc::kVintrp: return "vintrp"; case Enc::kDs: return "ds";
    case Enc::kMubuf: return "mubuf"; case Enc::kMtbuf: return "mtbuf";
    case Enc::kMimg: return "mimg"; case Enc::kExp: return "exp";
    default: return "UNKNOWN";
  }
}

// Per-instruction decode trace: pc / decoded length / encoding / opcode / raw
// dword(s). A length that lands the next pc mid-instruction shows up as a garbage
// "UNKNOWN" op on the following line (a decoder desync).
void DumpProgram(const Program& prog, const char* tag) {
  std::fprintf(stderr, "[gcnspv] === %s decode: %zu insts ===\n", tag, prog.size());
  for (const Inst& in : prog) {
    std::fprintf(stderr, "[gcnspv]   pc=%04x len=%u %-6s op=%#05x  %08x", in.pc,
                 in.size, EncName(in.enc), in.opcode, in.raw[0]);
    if (in.size >= 2) std::fprintf(stderr, " %08x", in.raw[1]);
    if (in.has_literal) std::fprintf(stderr, " lit=%08x", in.literal);
    std::fprintf(stderr, "\n");
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

// RDNA2 VOP3-only integer ops the GFX7 emitter doesn't number (3-input fused +
// native add/sub_i32). Carry-out is dropped, like v_add_nc_u32.
bool RdnaEmitVop3Int(Translator& t, uint32_t op, uint32_t vdst, Id s0, Id s1,
                     Id s2) {
  switch (op) {
    case 0x30F: t.SetVg(vdst, t.Add(s0, s1)); return true;             // v_add_nc_i32
    case 0x310: t.SetVg(vdst, t.Sub(s0, s1)); return true;             // v_sub_nc_i32
    case 0x319: t.SetVg(vdst, t.Sub(s1, s0)); return true;             // v_subrev_nc_u32
    case 0x346: t.SetVg(vdst, t.Add(t.Shl(s0, s1), s2)); return true;  // v_lshl_add_u32
    case 0x347: t.SetVg(vdst, t.Shl(t.Add(s0, s1), s2)); return true;  // v_add_lshl_u32
    case 0x36D: t.SetVg(vdst, t.Add(t.Add(s0, s1), s2)); return true;  // v_add3_u32
    case 0x36F: t.SetVg(vdst, t.Or(t.Shl(s0, s1), s2)); return true;   // v_lshl_or_b32
    case 0x371: t.SetVg(vdst, t.Or(t.And(s0, s1), s2)); return true;   // v_and_or_b32
    case 0x372: t.SetVg(vdst, t.Or(t.Or(s0, s1), s2)); return true;    // v_or3_b32
    default: return false;
  }
}

// VOP3P packed f16: componentwise on the unpacked f32x2. op_sel/neg/clamp and
// packed-integer ops are not modelled (fall back).
bool RdnaEmitVop3p(Translator& t, uint32_t op, uint32_t vdst, uint32_t s0,
                   uint32_t s1, uint32_t s2, uint32_t lit) {
  auto unpack = [&](uint32_t f) {
    return t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.SrcRaw(f, lit)});
  };
  auto pack = [&](Id v2) {
    return t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16, {v2});
  };
  const Id a = unpack(s0), b = unpack(s1);
  switch (op & 0x7F) {
    case 0x0F: t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFAdd, t.t_v2, {a, b}))); return true;  // v_pk_add_f16
    case 0x10: t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFMul, t.t_v2, {a, b}))); return true;  // v_pk_mul_f16
    case 0x11: t.SetVg(vdst, pack(t.m.ExtInst(t.t_v2, GLSLstd450FMin, {a, b}))); return true; // v_pk_min_f16
    case 0x12: t.SetVg(vdst, pack(t.m.ExtInst(t.t_v2, GLSLstd450FMax, {a, b}))); return true; // v_pk_max_f16
    case 0x0E: {  // v_pk_fma_f16
      const Id mul = t.m.Emit(spv::Op::OpFMul, t.t_v2, {a, b});
      t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFAdd, t.t_v2, {mul, unpack(s2)})));
      return true;
    }
    default: return false;
  }
}

// ---- SMEM (constant buffers) ------------------------------------------------
// Walk the s_load pointer chain feeding a descriptor base SGPR back to a
// user-data root. `loads` maps an s_load's destination SGPR to its {source base
// SGPR, byte offset}. Fills chain_off[] (root-first resolution order) and returns
// the root user-data SGPR; *len is the number of dereferences (0 == already a
// user-data descriptor). Bounded to 3 levels.
uint32_t TraceCbufChain(
    uint32_t sbase,
    const std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>>& loads,
    uint32_t chain_off[3], uint32_t* len) {
  uint32_t cur = sbase, n = 0, tmp[3] = {};
  while (n < 3) {
    auto it = loads.find(cur);
    if (it == loads.end()) break;
    tmp[n++] = it->second.second;  // byte offset
    cur = it->second.first;        // the s_load's own source base SGPR
  }
  for (uint32_t i = 0; i < n; i++) chain_off[i] = tmp[n - 1 - i];  // reverse
  *len = n;
  return cur;
}

// Plan the set-1 UBO bindings a stage's SMEM loads reference. A leaf read is an
// s_buffer_load* (op 0x08-0x0C, V# in the sbase quad) or an s_load* (op 0x00-0x04,
// pointer in the sbase pair) whose result is used as data -- not as another SMEM's
// descriptor base. When the base SGPR was itself s_load'd (a runtime pointer
// chain, e.g. a 2D VS that loads its transform's V# from a root descriptor table),
// the chain back to the user-data root is recorded so the renderer can walk it.
bool RdnaPlanCbufs(const Program& program, uint32_t first_binding,
                   std::vector<ShaderCbuf>& cbufs,
                   std::unordered_map<uint32_t, uint32_t>& bindings) {
  std::unordered_set<uint32_t> used_as_base;
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kSmrd) continue;
    const uint32_t op = inst.opcode;
    if (op > 0x04 && (op < 0x08 || op > 0x0C)) continue;
    used_as_base.insert((inst.raw[0] & 0x3F) * 2);
  }

  // Walk in program order, growing the def map as s_loads appear, so each
  // SMEM's base traces through the defs live AT that instruction. Shaders
  // reuse SGPRs (the sprite VS s_buffer_loads its transform from the s[8:11]
  // user-data V#, then s_loads the vertex V# INTO s[8:11]); a whole-program
  // last-write map would misroute the transform to the vertex chain.
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> loads;  // sdst->{src,off}
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kSmrd) continue;
    const uint32_t op = inst.opcode;
    const bool sbufload = op >= 0x08 && op <= 0x0C, sload = op <= 0x04;
    if (!sbufload && !sload) continue;
    const uint32_t sbase = (inst.raw[0] & 0x3F) * 2;
    const uint32_t sdst = (inst.raw[0] >> 6) & 0x7F;
    const int32_t off = SignExt21(inst.raw[1] & 0x1FFFFF);
    if (sload && used_as_base.count(sdst)) {  // chain link: host-resolved
      loads[sdst] = {sbase, static_cast<uint32_t>(off < 0 ? 0 : off)};
      continue;
    }
    const uint32_t hi = static_cast<uint32_t>(off < 0 ? 0 : off) / 4 + SmemLoadCount(op);

    uint32_t chain_off[3] = {}, chain_len = 0;
    const uint32_t root = TraceCbufChain(sbase, loads, chain_off, &chain_len);

    auto it = bindings.find(sbase);
    if (it == bindings.end()) {
      const uint32_t binding = first_binding + static_cast<uint32_t>(cbufs.size());
      if (binding >= 8) return true;  // set 1 has 8 UBO bindings; ignore extras
      bindings[sbase] = binding;
      ShaderCbuf cb;
      cb.binding = binding;
      cb.ud_sgpr = root;
      cb.num_dwords = hi;
      cb.chain_len = chain_len;
      for (uint32_t i = 0; i < 3; i++) cb.chain_off[i] = chain_off[i];
      cbufs.push_back(cb);
    } else {
      for (ShaderCbuf& cb : cbufs)
        if (cb.binding == it->second && hi > cb.num_dwords) cb.num_dwords = hi;
    }
  }
  return true;
}

// Plan set-1 UBO bindings for CONSTANT buffer_load_format ops (a load whose index
// is not the vertex index reads a uniform, e.g. the 2D ortho matrix). Each distinct
// srsrc V# gets one binding; the renderer resolves the live V# from user data at
// draw time, exactly like the SMEM cbufs (decodeVBuffer(&vud[srsrc])).
void RdnaPlanBufLoadCbufs(const Program& program, uint32_t first_binding,
                          std::vector<ShaderCbuf>& cbufs,
                          std::unordered_map<uint32_t, uint32_t>& bindings,
                          std::unordered_map<uint32_t, uint32_t>& by_pc) {
  // Mirror ParseFetchInsts' walk: srsrc SGPRs written by an s_load hold V#s from
  // the user-data descriptor TABLE, whose entries are consumed by buffer loads
  // in program order (the sprite VS reads 6 vertex V#s then 2 constant-block
  // V#s). A constant load through such a V# becomes a chained cbuf (root = the
  // s_load's sbase pair, table offset = slot * 16 bytes); the renderer derefs
  // the table pointer at draw time. srsrc-keyed bindings collide when the shader
  // reuses an SGPR quad (s[8:11] = MVP V# then vertex V#), so constant loads are
  // bound per-instruction (by_pc) instead.
  int32_t loaded_from[128];
  for (int i = 0; i < 128; i++) loaded_from[i] = -1;
  uint32_t slot = 0;
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSop1 && inst.opcode == 0x20) break;  // s_setpc_b64
    if (inst.enc == Enc::kSopp && inst.opcode == 1) break;     // s_endpgm
    if (inst.enc == Enc::kSmrd && inst.opcode <= 0x04) {
      const uint32_t sdst = (inst.raw[0] >> 6) & 0x7F;
      const uint32_t sbase = (inst.raw[0] & 0x3F) * 2;
      const uint32_t nreg = inst.opcode == 0 ? 1 : inst.opcode == 1 ? 2
                          : inst.opcode == 2 ? 4 : inst.opcode == 3 ? 8 : 16;
      for (uint32_t k = 0; k < nreg && sdst + k < 128; k++)
        loaded_from[sdst + k] = static_cast<int32_t>(sbase);
      continue;
    }
    if (inst.enc != Enc::kMubuf || inst.opcode > 0x03) continue;
    const uint32_t srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    const bool chained = srsrc < 128 && loaded_from[srsrc] >= 0;
    const uint32_t my_slot = chained ? slot++ : 0;
    if (BufLoadIsVertexFetch(inst)) continue;  // real fetch -> vertex input path
    const uint32_t binding = first_binding + static_cast<uint32_t>(cbufs.size());
    if (chained) {
      if (binding >= 8) return;
      by_pc[inst.pc] = binding;
      ShaderCbuf cb;
      cb.binding = binding;
      cb.ud_sgpr = static_cast<uint32_t>(loaded_from[srsrc]);
      cb.num_dwords = 16;
      cb.chain_len = 1;
      cb.chain_off[0] = my_slot * 16;
      cbufs.push_back(cb);
      continue;
    }
    if (bindings.count(srsrc)) continue;
    if (binding >= 8) return;  // set 1 has 8 UBO bindings
    bindings[srsrc] = binding;
    cbufs.push_back({binding, srsrc, 16});  // mat4-sized default window (16 dwords)
  }
}

// Find `s_mov exec, sN` movs whose source SGPR is never written before the mov:
// sN then holds SPI launch state we do not model (the PS ABI saves the initial
// coverage mask in the first post-user-data SGPR and reloads EXEC from it).
// Emitting the mov would read our zero-initialised register file and turn EXEC
// off, making the CFG path skip every export (the PS then kills all fragments).
// Those movs are dropped so EXEC keeps its all-on seed.
std::unordered_set<uint32_t> LaunchExecMovPcs(const Program& program) {
  std::unordered_set<uint32_t> skip, written;
  for (const Inst& in : program) {
    uint32_t d0 = 0xFFFF, n = 1;
    switch (in.enc) {
      case Enc::kSop1: {
        const uint32_t sdst = (in.raw[0] >> 16) & 0x7F;
        const uint32_t src = in.raw[0] & 0xFF;
        const bool b64 = in.opcode == 0x04;
        if ((in.opcode == 0x03 || b64) && sdst == 126 && src <= 105 &&
            !written.count(src) && (!b64 || !written.count(src + 1))) {
          skip.insert(in.pc);
          if (ShDbg())
            std::fprintf(stderr, "[gcnspv] drop launch-state exec mov @pc=%04x (s%u)\n",
                         in.pc, src);
        }
        d0 = sdst;
        n = b64 ? 2 : 1;
        break;
      }
      case Enc::kSop2:
      case Enc::kSopk:
        d0 = (in.raw[0] >> 16) & 0x7F;
        break;
      case Enc::kSmrd:
        if (in.opcode <= 0x0C) {
          d0 = (in.raw[0] >> 6) & 0x7F;
          n = SmemLoadCount(in.opcode);
        }
        break;
      default:
        break;
    }
    if (d0 <= 105)
      for (uint32_t k = 0; k < n; k++) written.insert(d0 + k);
  }
  return skip;
}

void RdnaEmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t op = inst.opcode;
  const uint32_t sdst = (inst.raw[0] >> 6) & 0x7F;
  const uint32_t sbase = (inst.raw[0] & 0x3F) * 2;
  const int32_t off = SignExt21(inst.raw[1] & 0x1FFFFF);
  // s_load* (op 0x00-0x04, a pointer in the sbase pair) and s_buffer_load* (op
  // 0x08-0x0C, a V# in the sbase quad) read `off` bytes into sdst.. from the UBO
  // the renderer bound for this sbase. A 2D VS reads its transform matrix this way.
  if (op <= 0x04 || (op >= 0x08 && op <= 0x0C)) {
    auto it = sc.cbuf_bind.find(sbase);
    if (it == sc.cbuf_bind.end()) {
      // An s_load whose result feeds a later SMEM is a pointer-chain link: the
      // renderer resolves that descriptor host-side, so it has no UBO and emits
      // nothing. An unplanned s_buffer_load is a real gap.
      if (op >= 0x08)
        gpu::gcn::WarnUnsupported("smem.cbuf-unplanned", op, inst.raw[0], inst.raw[1]);
      return;
    }
    const uint32_t dword0 = static_cast<uint32_t>(off < 0 ? 0 : off) / 4;
    const uint32_t n = SmemLoadCount(op);
    for (uint32_t k = 0; k < n; k++)
      t.SetSg(sdst + k, t.CbufDword(it->second, dword0 + k));
    return;
  }
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
      // DELTA_GPU_FORCECOLOR: override the PS color output with opaque red, so a
      // rendered draw is unmistakably visible regardless of texture/vertex-color
      // math. Combined with FORCEQUAD this isolates rasterization/scanout from the
      // PS color path.
      static const bool force_col = std::getenv("DELTA_GPU_FORCECOLOR") != nullptr;
      if (force_col)
        col = t.m.CompositeConstruct(
            t.t_v4, {t.F32(1.f), t.F32(0.f), t.F32(0.f), t.F32(1.f)});
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

// SDWA (src0 field == 249) / DPP (== 250) put the real src0 VGPR in the extra
// control dword (its [7:0]); the decoder already consumed that dword (see
// valuSrc0Extra) and stashed it in inst.literal. Resolve the effective src0 field
// + literal so the shared VALU emitters see the real operand. The sub-dword byte/
// word selection (SDWA) and lane swizzle (DPP) are approximated as full-dword
// identity, so a VOP with a modifier at least reads the right register instead of
// decoding the escape (249/250) as zero.
void ResolveValuSrc0(const Inst& inst, uint32_t src0, uint32_t& field,
                     uint32_t& literal) {
  if (src0 == 249 || src0 == 250) {
    field = 256 + (inst.literal & 0xFF);
    literal = 0;
    gpu::gcn::WarnUnsupported("valu.sdwa/dpp-approx", src0, inst.raw[0], inst.literal);
  } else {
    field = src0;
    literal = inst.literal;
  }
}

// ---- per-instruction dispatch ----------------------------------------------
// Decodes RDNA2 field layouts and calls the shared GFX7 emitters (which take
// pre-decoded operands + a GFX7-canonical opcode). The scalar and VOP1/2/C
// field layouts are identical to GFX7; VOP3 differs only in the opcode-field
// width and the CLAMP bit position (bit 15, not 11); SMEM replaces SMRD.
void RdnaEmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSop1:
      if (sc.skip_launch_movs.count(inst.pc)) break;
      gpu::gcn::EmitSop1(t, inst);
      break;
    case Enc::kSop2: gpu::gcn::EmitSop2(t, inst); break;
    case Enc::kSopc: gpu::gcn::EmitSopc(t, inst); break;
    case Enc::kSopk: gpu::gcn::EmitSopk(t, inst); break;
    case Enc::kSopp:
      if (inst.opcode == 0x0A && sc.is_cs)  // s_barrier
        t.m.EmitVoid(spv::Op::OpControlBarrier, {t.U32(2), t.U32(2), t.U32(0x108)});
      break;  // s_nop / s_waitcnt / branches: no-op here (CFG handles branches)
    case Enc::kSmrd: RdnaEmitSmem(t, inst, sc); break;
    case Enc::kVop1: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      uint32_t src0, lit;
      ResolveValuSrc0(inst, w & 0x1FF, src0, lit);
      gpu::gcn::EmitVop1(t, op, vdst, t.SrcF(src0, lit));
      break;
    }
    case Enc::kVop2: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const uint32_t vsrc1 = (w >> 9) & 0xFF;
      uint32_t src0, lit;
      ResolveValuSrc0(inst, w & 0x1FF, src0, lit);
      const Id s0u = t.SrcRaw(src0, lit);
      const Id s1u = t.SrcRaw(256 + vsrc1, lit);
      // RDNA2-only VOP2 numbers the shared GFX7 emitter would misinterpret: the
      // no-carry integer add/sub forms must NOT write VCC (a later v_cndmask
      // reads it), and v_xnor_b32 sits where GFX7 has v_bfm_b32.
      switch (op) {
        case 0x1E: t.SetVg(vdst, t.Not(t.Xor(s0u, s1u))); break;  // v_xnor_b32
        case 0x25: t.SetVg(vdst, t.Add(s0u, s1u)); break;         // v_add_nc_u32
        case 0x26: t.SetVg(vdst, t.Sub(s0u, s1u)); break;         // v_sub_nc_u32
        case 0x27: t.SetVg(vdst, t.Sub(s1u, s0u)); break;         // v_subrev_nc_u32
        default:
          gpu::gcn::EmitVop2(t, RemapVop2(op), vdst, t.SrcF(src0, lit),
                             t.SrcF(256 + vsrc1, lit), lit);
          break;
      }
      break;
    }
    case Enc::kVop3: {
      uint32_t op = inst.opcode;
      const uint32_t vdst = w & 0xFF;
      if (op & 0x400) {  // VOP3P (packed 16-bit math)
        const uint32_t p0 = w1 & 0x1FF, p1 = (w1 >> 9) & 0x1FF;
        const uint32_t p2 = (w1 >> 18) & 0x1FF;
        if (!RdnaEmitVop3p(t, op, vdst, p0, p1, p2, inst.literal))
          gpu::gcn::WarnUnsupported("vop3p", op & 0x3FF, w, w1);
        break;
      }
      // VOP3-form v_cndmask is the VOP2 alias 0x101 on RDNA2 (0x01 renumber);
      // GFX7's explicit-S2 cndmask VOP3 op is 0x100.
      if (op == 0x101) op = 0x100;
      const uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const uint32_t s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      if (RdnaEmitVop3Int(t, op, vdst, t.SrcRaw(s0, inst.literal),
                          t.SrcRaw(s1, inst.literal), t.SrcRaw(s2, inst.literal)))
        break;
      const bool vop3b = gpu::gcn::IsVop3b(op);
      const uint32_t sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const uint32_t abs = vop3b ? 0 : ((w >> 8) & 7);
      const bool clamp = !vop3b && ((w >> 15) & 1);  // RDNA2 CLAMP at bit 15
      gpu::gcn::EmitVop3(t, op, vdst, t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                         t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                         t.SrcF(s2, inst.literal, neg & 4, abs & 4),
                         t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp);
      break;
    }
    case Enc::kVopc: {
      const uint32_t op = inst.opcode;
      const uint32_t vsrc1 = (w >> 9) & 0xFF;
      uint32_t src0, lit;
      ResolveValuSrc0(inst, w & 0x1FF, src0, lit);
      // RDNA2 f16 compares sit at 0xC8-0xCF, which the GFX7-numbered EmitVopc
      // would read as u32 integer compares; run the float predicate (op-0xC8) on
      // the low-half f16 operands instead. u32 compares stay at 0xC0-0xC7.
      if (op >= 0xC8 && op <= 0xCF) {
        auto f16 = [&](uint32_t f) {
          return t.m.CompositeExtract(
              t.t_f, t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                                 {t.SrcRaw(f, lit)}), 0);
        };
        gpu::gcn::EmitVopc(t, op - 0xC8, f16(src0), f16(256 + vsrc1),
                           t.SrcRaw(src0, lit), t.SrcRaw(256 + vsrc1, lit));
        break;
      }
      gpu::gcn::EmitVopc(t, op, t.SrcF(src0, lit), t.SrcF(256 + vsrc1, lit),
                         t.SrcRaw(src0, lit), t.SrcRaw(256 + vsrc1, lit));
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
    case Enc::kMubuf: {
      if (inst.opcode > 0x03) {  // stores / raw loads: not used by Isaac's VS/PS
        gpu::gcn::WarnUnsupported("mubuf.rdna", inst.opcode, w, w1);
        break;
      }
      // A per-vertex fetch was lifted to a Location vertex input and its VGPRs are
      // seeded before the body, so nothing is emitted here. A CONSTANT load (e.g.
      // the 2D ortho matrix) reads num_comps dwords from the bound UBO at the
      // computed byte offset into the destination VGPRs.
      if (BufLoadIsVertexFetch(inst)) break;
      const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
      uint32_t binding;
      auto pit = sc.mubuf_cbuf_by_pc.find(inst.pc);
      if (pit != sc.mubuf_cbuf_by_pc.end()) {
        binding = pit->second;
      } else {
        auto it = sc.cbuf_bind.find(srsrc);
        if (it == sc.cbuf_bind.end()) {
          gpu::gcn::WarnUnsupported("mubuf.cbuf-unplanned", inst.opcode, w, w1);
          break;
        }
        binding = it->second;
      }
      const uint32_t nc = (inst.opcode & 3) + 1;
      const uint32_t inst_offset = w & 0xFFF, vdata = (w1 >> 8) & 0xFF;
      const bool idxen = (w >> 13) & 1, offen = (w >> 12) & 1;
      const uint32_t vaddr = w1 & 0xFF;
      // byte offset = inst_offset + index*stride + voffset. The V# stride is not
      // available in the shader (the descriptor SGPRs are not seeded), so an
      // indexed uniform row-select assumes tight packing (stride == nc*4); an
      // unindexed load uses the immediate offset directly.
      Id byte_off = t.U32(inst_offset);
      if (idxen) byte_off = t.Add(byte_off, t.Mul(t.Vg(vaddr), t.U32(nc * 4)));
      if (offen) byte_off = t.Add(byte_off, t.Vg(vaddr + (idxen ? 1u : 0u)));
      const Id dword0 = t.Shr(byte_off, t.U32(2));
      for (uint32_t k = 0; k < nc; k++)
        t.SetVg(vdata + k, t.CbufDwordId(binding, t.Add(dword0, t.U32(k))));
      break;
    }
    case Enc::kMimg:
      // Fields line up with the shared emitter; NSA=0 (sequential vaddr) only.
      gpu::gcn::EmitMimg(t, inst, sc);
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
  uint32_t sem = 0;   // dense vertex-input location (per-vertex fetches only)
  uint32_t slot = 0;  // V#-table entry index (ALL table-chained buffer loads
                      // consume entries in program order, incl. constant loads)
  // Track SGPRs written by s_load* -- the NGG VS loads its vertex-fetch V# into
  // srsrc from a user_data pointer (a TABLE of V#s), so the real table root is
  // that s_load's sbase, not srsrc itself. Map each loaded SGPR -> its sbase.
  int32_t loaded_from[128];
  for (int i = 0; i < 128; i++) loaded_from[i] = -1;
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20) break;  // s_setpc_b64 (return)
    if (in.enc == Enc::kSopp && in.opcode == 1) break;     // s_endpgm
    if (in.enc == Enc::kSmrd && in.opcode <= 0x04) {  // s_load_dword{,x2,x4,x8,x16}
      const uint32_t sdst = (in.raw[0] >> 6) & 0x7F;
      const uint32_t sbase = (in.raw[0] & 0x3F) * 2;
      const uint32_t nreg = in.opcode == 0 ? 1 : in.opcode == 1 ? 2
                          : in.opcode == 2 ? 4 : in.opcode == 3 ? 8 : 16;
      for (uint32_t k = 0; k < nreg && sdst + k < 128; k++)
        loaded_from[sdst + k] = static_cast<int32_t>(sbase);
      continue;
    }
    if (in.enc != Enc::kMubuf || in.opcode > 0x03) continue;  // buffer_load_format_*
    const uint32_t vdata = (in.raw[1] >> 8) & 0xFF;
    const uint32_t srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    const uint32_t nc = (in.opcode & 3) + 1;
    const bool idxen = (in.raw[0] >> 13) & 1, offen = (in.raw[0] >> 12) & 1;
    const uint32_t vaddr = in.raw[1] & 0xFF, soffset = (in.raw[1] >> 24) & 0xFF;
    const bool vtx = BufLoadIsVertexFetch(in);
    // If srsrc's V# was loaded via s_load from a user_data pointer, this fetch
    // reads a TABLE of V#s at that pointer: table_sgpr = the s_load's sbase and
    // this attribute is the sem-th 16-byte (4-dword) V# in the table. Otherwise
    // the V# is inline in user data at srsrc (table_sgpr = srsrc, dword_off = 0).
    uint32_t table_sgpr = srsrc, dword_off = 0;
    const bool chained = srsrc < 128 && loaded_from[srsrc] >= 0;
    if (chained) {
      table_sgpr = static_cast<uint32_t>(loaded_from[srsrc]);
      dword_off = slot * 4;
      slot++;
    }
    if (ShDbg())
      std::fprintf(stderr,
                   "[gcnspv] buf_load nc=%u vdst=v%u srsrc=s%u idxen=%u offen=%u "
                   "vaddr=v%u soffset=s%u -> %s (table_sgpr=s%u doff=%u)\n",
                   nc, vdata, srsrc, idxen, offen, vaddr, soffset,
                   vtx ? "vertex-attr" : "const-ubo", table_sgpr, dword_off);
    // Only a genuine per-vertex fetch becomes a vertex input. A constant load is
    // left for the UBO path (RdnaPlanBufLoadCbufs assigns the same table slots).
    if (vtx) {
      out.push_back({sem, nc, vdata, table_sgpr, dword_off});
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
  if (ShDbg()) DumpProgram(program, "vs");
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

  // NGG merged-wave prologue: the VS derives its EXEC/lane bookkeeping from
  // merged_wave_info in s3 (verts-in-wave [7:0], prims [15:8]); model a
  // 1-vert/1-prim wave so that math yields a live lane instead of zeros.
  t.SetSg(3, t.U32(0x0101));

  Id vertex_index = 0;
  if (attrs.empty()) {  // procedural VS: seed the ABI VGPRs from Vulkan built-ins
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
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
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  if (!RdnaPlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind)) return false;
  // Constant buffer_load descriptors (e.g. the ortho matrix a procedural 2D VS
  // reads) become additional set-1 UBOs after the SMEM cbufs.
  RdnaPlanBufLoadCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind, sc.mubuf_cbuf_by_pc);
  if (ShDbg())
    std::fprintf(stderr, "[gcnspv] vs planned %zu cbufs\n", r.vs_cbufs.size());

  EmitBody(t, program, sc);
  r.num_params = sc.max_param;

  // DELTA_GPU_FORCEQUAD: ignore the VS's computed position and emit a full-screen
  // quad straight from gl_VertexIndex (0..3 -> the four NDC corners). If this
  // renders (with the PS forced white) the raster/PS pipeline is sound and the real
  // position math is the culprit (e.g. an unbound MVP -> degenerate); if it still
  // draws nothing the problem is downstream (index/vertexCount/render target).
  static const bool force_quad = std::getenv("DELTA_GPU_FORCEQUAD") != nullptr;
  if (force_quad) {
    // A fetch-path VS never created a VertexIndex input, and its v0 is clobbered by
    // the vertex fetch -- bind a real VertexIndex here so the quad is valid either way.
    Id vidx_var = vertex_index;
    if (!vidx_var) {
      vidx_var = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                              spv::StorageClass::Input);
      t.m.Decorate(vidx_var, spv::Decoration::BuiltIn,
                   {static_cast<uint32_t>(spv::BuiltIn::VertexIndex)});
      iface.push_back(vidx_var);
    }
    const Id vidx = t.m.Load(t.t_u, vidx_var);
    const Id fx = t.SelectF(t.IsNonZero(t.And(vidx, t.U32(1))), t.F32(1.f), t.F32(-1.f));
    const Id fy = t.SelectF(t.IsNonZero(t.And(vidx, t.U32(2))), t.F32(1.f), t.F32(-1.f));
    t.m.Store(pos_out, t.m.CompositeConstruct(t.t_v4, {fx, fy, t.F32(0.f), t.F32(1.f)}));
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

// SPI_PS_INPUT_ENA lays the PS's system input VGPRs into v0.. in ascending bit
// order (each set bit consumes the width below). The v_interp i/j barycentrics
// come from the first pair, which the Vulkan-interpolated Location inputs model,
// so those VGPRs need no value. But a shader that reads screen position directly
// (a 2D clip such as frag.x*a + frag.y*b < c) reads the POS_{X,Y,Z,W} VGPRs;
// unseeded they stay zero, collapsing the clip to "0 < c" and discarding every
// fragment. Seed those from gl_FragCoord (and FRONT_FACE from gl_FrontFacing).
void SeedPsInputVgprs(Translator& t, uint32_t ena, std::vector<Id>& iface) {
  if (!ena) return;
  static const uint8_t width[16] = {2, 2, 2, 3, 2, 2, 2, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t vg[16] = {}, next = 0;
  for (uint32_t b = 0; b < 16; b++)
    if (ena & (1u << b)) { vg[b] = next; next += width[b]; }

  if ((ena >> 8) & 0xF) {  // POS_X..W at bits 8..11 -> gl_FragCoord.xyzw
    const Id fc = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                               spv::StorageClass::Input);
    t.m.Decorate(fc, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::FragCoord)});
    iface.push_back(fc);
    const Id v = t.m.Load(t.t_v4, fc);
    for (uint32_t c = 0; c < 4; c++)
      if (ena & (1u << (8 + c)))
        t.SetVgF(vg[8 + c], t.m.CompositeExtract(t.t_f, v, c));
  }
  if (ena & (1u << 12)) {  // FRONT_FACE -> 0xffffffff front / 0 back
    const Id ff =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_bool),
                     spv::StorageClass::Input);
    t.m.Decorate(ff, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::FrontFacing)});
    iface.push_back(ff);
    t.SetVg(vg[12],
            t.SelectB(t.m.Load(t.t_bool, ff), t.U32(0xFFFFFFFFu), t.U32(0)));
  }
}

bool TranslatePs(const Program& program,
                 const std::unordered_set<uint32_t>& flat_attrs,
                 uint32_t ps_input_ena, Recompiled& r, Translator& t) {
  if (ShDbg()) DumpProgram(program, "ps");
  std::vector<Id> iface;
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  if (!RdnaPlanCbufs(program, static_cast<uint32_t>(r.vs_cbufs.size()),
                     r.ps_cbufs, sc.cbuf_bind))
    return false;
  const gpu::gcn::MimgBindingPlan mimg_plan = RdnaPlanMimg(program);
  sc.mimg_plan = &mimg_plan;  // borrowed by EmitBody
  for (uint32_t i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back({i, mimg_plan.binding_srsrc[i], mimg_plan.binding_storage[i]});

  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedPsInputVgprs(t, ps_input_ena, iface);

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
                     const uint32_t* vs_user_data, const uint32_t* ps_user_data,
                     uint32_t ps_input_ena) {
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
  if (ps_code ? !TranslatePs(ps_program, flat_attrs, ps_input_ena, r, tp)
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
