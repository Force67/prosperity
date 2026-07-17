/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> SPIR-V translator: per-instruction dispatch, control-flow lowering,
 * and the per-stage (VS/PS/CS) drivers. The ALU and memory emitters live in
 * translate_alu.cpp / translate_mem.cpp; the shared context in translator.h.
 */

#include "gcn_spirv.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
// Backend disabled at build time (SPIRV-Tools/Headers unavailable). There is
// no other recompiler: every recompile declines and the affected
// draws/dispatches are skipped.
namespace gpu::gcn {
bool RecompileSpirv(const uint32_t*, const uint32_t*, const uint32_t*,
                    const uint32_t*, Recompiled&) {
  return false;
}
bool RecompileComputeSpirv(const uint32_t*, uint32_t, uint32_t, uint32_t,
                           uint32_t, uint32_t, uint32_t, RecompiledCs&) {
  return false;
}
}  // namespace gpu::gcn
#else

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unordered_set>

#include "spv_post.h"
#include "translator.h"

namespace gpu::gcn {

bool TraceEnabled() {
  static const bool enabled = std::getenv("DELTA_GPU_SHTRACE") != nullptr;
  return enabled;
}

void WarnUnsupported(const char* enc, uint32_t op, uint32_t w0, uint32_t w1) {
  static std::unordered_set<uint64_t> seen;
  const uint64_t key =
      static_cast<uint64_t>(std::hash<std::string_view>{}(enc)) ^
      (static_cast<uint64_t>(op) << 40);
  if (seen.size() > 512 || !seen.insert(key).second) return;
  std::fprintf(stderr,
               "[gcnspv] UNSUPPORTED %s op=%#x (w0=%#x w1=%#x) -> approximated\n",
               enc, op, w0, w1);
}

// ---- stage-io helpers -------------------------------------------------------
Id PsInputVar(Translator& t, StageContext& sc, uint32_t attr) {
  auto it = sc.in_vars.find(attr);
  if (it != sc.in_vars.end()) return it->second;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                            spv::StorageClass::Input);
  t.m.Decorate(v, spv::Decoration::Location, {attr});
  if (sc.flat_attrs && sc.flat_attrs->count(attr))
    t.m.Decorate(v, spv::Decoration::Flat);
  sc.iface->push_back(v);
  sc.in_vars[attr] = v;
  return v;
}

Id VsParamOut(Translator& t, StageContext& sc, uint32_t p) {
  auto it = sc.param_outs.find(p);
  if (it != sc.param_outs.end()) return it->second;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::Location, {p});
  if (sc.flat_attrs && sc.flat_attrs->count(p))
    t.m.Decorate(v, spv::Decoration::Flat);
  sc.iface->push_back(v);
  sc.param_outs[p] = v;
  return v;
}

// Lazily declare the PS color output for an MRT target (location == target)
// and record it in ps_mrt_mask so the renderer masks unwritten attachments.
Id PsColorOut(Translator& t, StageContext& sc, uint32_t target) {
  if (sc.color_outs[target]) return sc.color_outs[target];
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::Location, {target});
  sc.iface->push_back(v);
  sc.color_outs[target] = v;
  sc.r->ps_mrt_mask |= static_cast<uint8_t>(1u << target);
  return v;
}

// Lazily declare gl_FragDepth for the MRTZ export (adds DepthReplacing).
Id PsDepthOut(Translator& t, StageContext& sc) {
  if (sc.depth_out) return sc.depth_out;
  const Id v = t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_f),
                            spv::StorageClass::Output);
  t.m.Decorate(v, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::FragDepth)});
  sc.iface->push_back(v);
  sc.depth_out = v;
  if (sc.main_fn) t.m.ExecMode(sc.main_fn, spv::ExecutionMode::DepthReplacing);
  return v;
}

namespace {

// Vertex attribute recovered from the Gnm fetch shader: an s_load_dwordx4 of
// the V# (from the vertex-buffer table a user SGPR points at) + a
// buffer_load_format into the destination VGPRs, one per attribute in
// semantic order.
struct FetchAttr {
  uint32_t semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
};

std::vector<FetchAttr> ParseFetch(uint64_t fetch_addr) {
  std::vector<FetchAttr> out;
  if (!InGuest(fetch_addr)) return out;
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
    if (inst.enc == Enc::kSopp && inst.opcode == 1) break;
    if (inst.enc == Enc::kSmrd && inst.opcode == 0x02) {
      const uint32_t sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
      loads[sdst] = {sbase * 2u, w & 0xFF};
    } else if (inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf) {
      const uint32_t w1 = inst.raw[1];
      const uint32_t vdata = (w1 >> 8) & 0xFF;
      const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
      const uint32_t nc = (inst.opcode & 3) + 1;
      const auto it = loads.find(srsrc);
      const uint32_t tbl = it != loads.end() ? it->second.table_sgpr : 0;
      const uint32_t off = it != loads.end() ? it->second.dword_off : 0;
      out.push_back({semantic, nc, vdata, tbl, off});
      semantic++;
    }
  }
  return out;
}

// ---- per-instruction dispatch ----------------------------------------------
// Emit one non-terminator instruction (branches are handled by the CFG driver).
void EmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSop1: EmitSop1(t, inst); break;
    case Enc::kSop2: EmitSop2(t, inst); break;
    case Enc::kSopc: EmitSopc(t, inst); break;
    case Enc::kSopk: EmitSopk(t, inst); break;
    case Enc::kSopp:
      if (inst.opcode == 0x0a && sc.is_cs) {  // s_barrier
        // ControlBarrier(Workgroup, Workgroup, AcquireRelease|WorkgroupMemory)
        t.m.EmitVoid(spv::Op::OpControlBarrier,
                     {t.U32(2), t.U32(2), t.U32(0x108)});
      }
      break;  // s_nop / s_waitcnt / hints: no-ops in this model
    case Enc::kSmrd:
      if (sc.is_cs) EmitCsSmrd(t, inst, sc);
      else EmitCbufSmrd(t, inst, sc.cbuf_bind);
      break;
    case Enc::kVop1: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
      EmitVop1(t, op, vdst, t.SrcF(src0, inst.literal));
      break;
    }
    case Enc::kVop2: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const uint32_t vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      EmitVop2(t, op, vdst, t.SrcF(src0, inst.literal),
               t.SrcF(256 + vsrc1, inst.literal), inst.literal);
      break;
    }
    case Enc::kVop3: {
      const uint32_t op = inst.opcode, vdst = w & 0xFF;
      const bool vop3b = IsVop3b(op);
      const uint32_t sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const uint32_t abs = vop3b ? 0 : ((w >> 8) & 7);
      const bool clamp = !vop3b && ((w >> 11) & 1);
      const uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const uint32_t s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      EmitVop3(t, op, vdst, t.SrcF(s0, inst.literal, neg & 1, abs & 1),
               t.SrcF(s1, inst.literal, neg & 2, abs & 2),
               t.SrcF(s2, inst.literal, neg & 4, abs & 4),
               t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp);
      break;
    }
    case Enc::kVopc: {
      const uint32_t op = inst.opcode;
      const uint32_t vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      EmitVopc(t, op, t.SrcF(src0, inst.literal),
               t.SrcF(256 + vsrc1, inst.literal), t.SrcRaw(src0, inst.literal),
               t.SrcRaw(256 + vsrc1, inst.literal));
      break;
    }
    case Enc::kVintrp: {
      if (!sc.is_ps) break;
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
      if (sc.is_cs) EmitCsMubuf(t, inst, sc);
      else WarnUnsupported("mubuf.graphics", inst.opcode, w, w1);
      break;
    case Enc::kMtbuf:
      if (sc.is_cs) EmitCsMtbuf(t, inst, sc);
      else WarnUnsupported("mtbuf.graphics", inst.opcode, w, w1);
      break;
    case Enc::kDs:
      if (sc.is_cs) EmitDs(t, inst, sc);
      else WarnUnsupported("ds.graphics", inst.opcode, w, w1);
      break;
    case Enc::kMimg:
      if (sc.is_cs) EmitCsMimg(t, inst, sc);
      else if (sc.is_ps) EmitMimg(t, inst, sc);
      break;
    case Enc::kExp: {
      if (sc.is_cs) {
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
            const Id c01 =
                t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])});
            const Id c23 =
                t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[1])});
            col = t.m.VectorShuffle(t.t_v4, c01, c23, {0, 1, 2, 3});
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
          if (p + 1 > sc.max_param) sc.max_param = p + 1;
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
    default:
      break;
  }
}

// ---- control flow: while/switch lowering -----------------------------------
// Branch classification. 0=none, 1=uncond, 2=scc0, 3=scc1, 4=vccz, 5=vccnz,
// 6=execz, 7=execnz, 8=endpgm.
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

// "Take the branch" condition for a conditional branch kind.
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

// Lower arbitrary control flow (reducible or not) to a while/switch state
// machine over basic blocks.
void EmitCfg(Translator& t, const Program& program, StageContext& sc) {
  const uint32_t max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  // Basic-block leaders: entry, every branch target, the slot after a branch.
  std::vector<uint32_t> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0) continue;
    leaders.push_back(inst.pc + inst.size);  // fall-through
    if (k >= 1 && k <= 7) {                  // has a PC-relative target
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<uint32_t>(
          static_cast<int32_t>(inst.pc) + static_cast<int32_t>(inst.size) +
          simm));
    }
  }
  std::sort(leaders.begin(), leaders.end());
  leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
  // Drop out-of-range leaders (targets past the decoded program -> EXIT).
  std::vector<uint32_t> starts;
  for (uint32_t l : leaders)
    if (l < max_pc) starts.push_back(l);
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
  t.m.Switch(state, exit_blk, cases);  // default (incl. EXIT state) -> exit

  for (uint32_t bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    const uint32_t blk_start = starts[bi];
    const uint32_t blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < blk_start || inst.pc >= blk_end) continue;
      const int k = BranchKind(inst);
      if (k == 0) {
        EmitInst(t, inst, sc);
        continue;
      }
      // terminator
      const uint32_t fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (k == 8) {  // endpgm
        t.SetState(kExit);
      } else if (k == 1) {  // unconditional
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        t.SetState(block_of(static_cast<uint32_t>(
            static_cast<int32_t>(inst.pc) + static_cast<int32_t>(inst.size) +
            simm)));
      } else {  // conditional
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        const uint32_t target = block_of(static_cast<uint32_t>(
            static_cast<int32_t>(inst.pc) + static_cast<int32_t>(inst.size) +
            simm));
        t.SetStateId(
            t.SelectB(BranchTaken(t, k), t.U32(target), t.U32(fall)));
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
  t.m.OpenBlock(merge);  // left open; caller emits the stage epilogue here
}

bool ForceCfg() {
  static const bool force = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  return force;
}

// Emit a stage body: branchy shaders take the CFG (while-switch) path so
// their control flow (the GCN alpha-test/discard idiom, conditional shading)
// is honoured; single-basic-block shaders emit the same instruction stream
// straight-line.
void EmitBody(Translator& t, const Program& program, StageContext& sc) {
  if (ForceCfg() || HasControlFlow(program)) {
    t.SeedExec();
    EmitCfg(t, program, sc);
    return;
  }
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSopp && inst.opcode == 1) break;  // s_endpgm
    EmitInst(t, inst, sc);
  }
}

// ---- VS ---------------------------------------------------------------------
bool TranslateVs(const Program& program, const uint32_t* vs_user_data,
                 const std::unordered_set<uint32_t>& flat_attrs, Recompiled& r,
                 Translator& t) {
  const uint64_t fetch =
      (static_cast<uint64_t>(vs_user_data[1] & 0xFFFF) << 32) | vs_user_data[0];
  const std::vector<FetchAttr> attrs = ParseFetch(fetch);
  t.InitTypes();

  std::vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);

  if (attrs.empty()) {
    // A procedural VS has no fetch shader. On GFX6-8, VertexID enters in v0;
    // InstanceID/StepRate0 enters in v1 and raw InstanceID is available in v3.
    // Seed those ABI inputs from Vulkan's draw built-ins.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/be00f53d4d50b87a87f83e8fa243b77e614eb0b8/src/gallium/drivers/radeonsi/gfx/si_state_shaders.cpp#L269-307
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

  // Seed destination VGPRs from fetched vertex attributes when present.
  for (const FetchAttr& a : attrs) {
    const Id comp_ty = a.num_comps == 1   ? t.t_f
                       : a.num_comps == 2 ? t.t_v2
                       : a.num_comps == 3 ? t.t_v3
                                          : t.t_v4;
    const Id in_var = t.m.Variable(
        t.m.TypePointer(spv::StorageClass::Input, comp_ty),
        spv::StorageClass::Input);
    t.m.Decorate(in_var, spv::Decoration::Location, {a.semantic});
    iface.push_back(in_var);
    const Id val = t.m.Load(comp_ty, in_var);
    // DELTA_GPU_VSFLIPZ: negate the z of the position attribute (semantic 0,
    // >= 3 comps) -- a projection-convention diagnostic. Default off.
    static const bool flip_z = std::getenv("DELTA_GPU_VSFLIPZ") != nullptr;
    for (uint32_t c = 0; c < a.num_comps; c++) {
      Id comp = a.num_comps == 1 ? val : t.m.CompositeExtract(t.t_f, val, c);
      if (flip_z && a.semantic == 0 && c == 2 && a.num_comps >= 3)
        comp = t.FNeg(comp);
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
  if (!PlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind)) return false;

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

// ---- PS ---------------------------------------------------------------------
bool TranslatePs(const Program& program,
                 const std::unordered_set<uint32_t>& flat_attrs, Recompiled& r,
                 Translator& t) {
  // Color outputs (PsColorOut) are declared lazily per MRT target (location ==
  // target); PS inputs (PsInputVar) likewise as they are read.
  std::vector<Id> iface;
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  if (!PlanCbufs(program, r.vs_cbufs.size(), r.ps_cbufs, sc.cbuf_bind))
    return false;

  // Sampler bindings: one per unique descriptor (shared plan with
  // TrackTextures). More unique samplers than the renderer's set-0 layout
  // provides cannot be expressed -- decline (the draw falls back).
  const MimgBindingPlan mimg_plan = PlanMimgBindings(program);
  if (mimg_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers) {
    WarnUnsupported("mimg.binding-count",
                    static_cast<uint32_t>(mimg_plan.binding_srsrc.size()));
    return false;
  }
  sc.mimg_plan = &mimg_plan;
  for (uint32_t i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back({i, mimg_plan.binding_srsrc[i]});

  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);

  const bool cfg = ForceCfg() || HasControlFlow(program);
  if (cfg) {
    // Default MRT0 to transparent so a fragment that never reaches an export
    // leaves a defined value even if the discard lowering is bypassed.
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.ConstComposite(t.t_v4, {t.F32(0.f), t.F32(0.f), t.F32(0.f),
                                          t.F32(0.f)}));
    sc.color_written_var =
        t.m.Variable(t.p_priv_u, spv::StorageClass::Private, t.m.ConstNull(t.t_u));
  }
  EmitBody(t, program, sc);

  if (!sc.wrote_color) {
    // Shader has no color export at all: opaque white fallback.
    t.m.Store(PsColorOut(t, sc, 0),
              t.m.ConstComposite(t.t_v4, {t.F32(1.f), t.F32(1.f), t.F32(1.f),
                                          t.F32(1.f)}));
  } else if (sc.color_written_var) {
    // GCN alpha-test/kill idiom (CFG path): control flow branches over the
    // color export for failing fragments (e.g. s_cmp + s_cbranch_scc0 ->
    // s_endpgm). Discard those (OpKill) instead of leaving the output
    // undefined. DELTA_GPU_NOKILL skips the discard as a diagnostic.
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
bool TranslateCs(const Program& program, uint32_t num_thread_x,
                 uint32_t num_thread_y, uint32_t num_thread_z,
                 uint32_t user_sgpr, uint32_t tgid_enable, uint32_t lds_dwords,
                 RecompiledCs& r, Translator& t) {
  if (program.empty()) return false;
  StageContext sc;
  sc.is_cs = true;
  // RSRC2.LDS_SIZE is in 128-dword granules.
  sc.lds_dwords = lds_dwords * 128;
  if (!PlanCsResources(program, sc.lds_dwords, r, sc.cs_bind) ||
      r.resources.empty())
    return false;

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

  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  sc.main_fn = main_fn;
  const Id p_pc_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (uint32_t i = 0; i < 16; i++)  // user data -> s0..s15
    t.SetSg(i, t.m.Load(t.t_u, t.m.AccessChain(p_pc_u, pc_var,
                                               {t.U32(0), t.U32(i)})));
  const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
  const auto group_comp = [&](uint32_t c) {
    return t.m.Load(t.t_u, t.m.AccessChain(p_in_u, group_id, {t.U32(c)}));
  };
  uint32_t sg = user_sgpr;
  if ((tgid_enable & 1) && sg < 106) t.SetSg(sg++, group_comp(0));
  if ((tgid_enable & 2) && sg < 106) t.SetSg(sg++, group_comp(1));
  if ((tgid_enable & 4) && sg < 106) t.SetSg(sg++, group_comp(2));
  for (uint32_t c = 0; c < 3; c++)  // local invocation id (tidig) -> v0..v2
    t.SetVg(c, t.m.Load(t.t_u, t.m.AccessChain(p_in_u, local_id, {t.U32(c)})));
  t.SeedExec();
  EmitCfg(t, const_cast<Program&>(program), sc);
  if (sc.cs_unsupported) return false;
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::GLCompute, main_fn, "main", iface);
  t.m.ExecMode(main_fn, spv::ExecutionMode::LocalSize,
               {num_thread_x ? num_thread_x : 1, num_thread_y ? num_thread_y : 1,
                num_thread_z ? num_thread_z : 1});
  r.local_size[0] = num_thread_x ? num_thread_x : 1;
  r.local_size[1] = num_thread_y ? num_thread_y : 1;
  r.local_size[2] = num_thread_z ? num_thread_z : 1;
  return true;
}

// ---- RECTLIST geometry expansion -------------------------------------------
// RECTLIST consumes three post-VS corners and rasterizes the fourth corner as
// a second triangle. Vulkan has no matching input topology, so insert a
// geometry stage that performs the fixed-function expansion without assuming
// anything about the guest VS.
std::vector<uint32_t> EmitRectListGeometry(
    uint32_t num_params, const std::unordered_set<uint32_t>& flat_attrs) {
  spirv::Module m;
  m.Capability(spv::Capability::Geometry);
  const Id t_void = m.TypeVoid(), t_f = m.TypeFloat(32), t_v4 = m.TypeVec(t_f, 4);
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
    if (!flat_attrs.count(p)) param3[p] = fourth_corner(inputs[p]);

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

// One-shot disassembly (DELTA_GPU_SHDIS): for the first branchy shaders, list
// each instruction's encoding + opcode.
void MaybeDumpBranchy(const char* tag, const Program& program) {
  static const bool enabled = std::getenv("DELTA_GPU_SHDIS") != nullptr;
  if (!enabled) return;
  static int dumped = 0;
  if (!HasControlFlow(program) || dumped >= 2) return;
  dumped++;
  static const char* kEncNames[] = {"unk",  "sop1",   "sop2",  "sopk", "sopc",
                                    "sopp", "smrd",   "vop1",  "vop2", "vop3",
                                    "vopc", "vintrp", "ds",    "mubuf",
                                    "mtbuf", "mimg",  "exp"};
  std::fprintf(stderr, "[shdis] %s branchy, %zu insts:\n", tag, program.size());
  for (const Inst& inst : program)
    std::fprintf(stderr, "[shdis]  pc=%u %s op=%#x w0=%#x w1=%#x\n", inst.pc,
                 kEncNames[static_cast<int>(inst.enc) <= 16
                               ? static_cast<int>(inst.enc)
                               : 0],
                 inst.opcode, inst.raw[0], inst.raw[1]);
}

bool NoOpt() {
  // DELTA_GPU_SPIRV_NOOPT: skip the optimize pass (keep the naive
  // memory-backed register SPIR-V). Isolates an emission bug from a spirv-opt
  // mis-promotion.
  static const bool no_opt = std::getenv("DELTA_GPU_SPIRV_NOOPT") != nullptr;
  return no_opt;
}

}  // namespace

// ---- entry points -----------------------------------------------------------
bool RecompileSpirv(const uint32_t* vs_code, const uint32_t* ps_code,
                    const uint32_t* vs_user_data, const uint32_t* ps_user_data,
                    Recompiled& r) {
  if (!vs_code || !vs_user_data || !ps_user_data) return false;

  // Decode each stage exactly once; every later step works on these programs.
  const Program vs_program = DecodeShader(vs_code, 4096);
  const Program ps_program = ps_code ? DecodeShader(ps_code, 4096) : Program{};
  MaybeDumpBranchy("VS", vs_program);
  if (!ps_program.empty()) MaybeDumpBranchy("PS", ps_program);

  // V_INTERP_MOV P0 reads a per-primitive parameter rather than a smoothly
  // interpolated value: represent those locations as flat varyings in both
  // stages.
  std::unordered_set<uint32_t> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 &&
        (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  // VS and PS are separate SPIR-V modules.
  Translator tv;
  if (!TranslateVs(vs_program, vs_user_data, flat_attrs, r, tv)) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] VS translation rejected @%p\n",
                   static_cast<const void*>(vs_code));
    return false;
  }
  Translator tp;
  tp.InitTypes();
  if (ps_code ? !TranslatePs(ps_program, flat_attrs, r, tp)
              : !TranslateDepthOnlyPs(tp)) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] PS translation rejected @%p\n",
                   static_cast<const void*>(ps_code));
    return false;
  }

  const std::vector<uint32_t> vs = tv.m.Assemble();
  const std::vector<uint32_t> gs = EmitRectListGeometry(r.num_params, flat_attrs);
  const std::vector<uint32_t> ps = tp.m.Assemble();
  std::string err;
  if (!spirv::Validate(vs, &err)) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] VS invalid: %s\n", err.c_str());
    return false;
  }
  if (!spirv::Validate(ps, &err)) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] PS invalid: %s\n", err.c_str());
    return false;
  }
  if (!spirv::Validate(gs, &err)) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] RECTLIST GS invalid: %s\n", err.c_str());
    return false;
  }
  r.vs_spirv = NoOpt() ? vs : spirv::Optimize(vs);
  r.gs_spirv = NoOpt() ? gs : spirv::Optimize(gs);
  r.fs_spirv = NoOpt() ? ps : spirv::Optimize(ps);
  r.ok = !r.vs_spirv.empty() && !r.gs_spirv.empty() && !r.fs_spirv.empty();

  // Tally (DELTA_GPU_SPIRV): how many shaders the backend accepted vs had to
  // decline, and how many used the CFG path.
  static const bool tally = std::getenv("DELTA_GPU_SPIRV") != nullptr;
  if (tally) {
    static int ok_count = 0, cfg_count = 0, logged = 0;
    if (r.ok) ok_count++;
    if (HasControlFlow(vs_program) || HasControlFlow(ps_program)) cfg_count++;
    if (logged < 12) {
      logged++;
      std::fprintf(stderr, "[gcnspv] recompiled ok=%d (cfg-shaders=%d) this=%s\n",
                   ok_count, cfg_count, r.ok ? "spirv" : "FALLBACK");
    }
  }
  return r.ok;
}

bool RecompileComputeSpirv(const uint32_t* cs_code, uint32_t num_thread_x,
                           uint32_t num_thread_y, uint32_t num_thread_z,
                           uint32_t user_sgpr, uint32_t tgid_enable,
                           uint32_t lds_dwords, RecompiledCs& r) {
  if (!cs_code) return false;
  const Program program = DecodeShader(cs_code, 2048);
  Translator t;
  RecompiledCs tmp;  // build into a temp so a mid-emit failure leaves r intact
  if (!TranslateCs(program, num_thread_x, num_thread_y, num_thread_z, user_sgpr,
                   tgid_enable, lds_dwords, tmp, t))
    return false;
  const std::vector<uint32_t> spv_bin = t.m.Assemble();
  std::string err;
  if (!spirv::Validate(spv_bin, &err)) {
    if (TraceEnabled())
      std::fprintf(stderr, "[gcnspv] CS invalid: %s\n", err.c_str());
    return false;
  }
  tmp.spirv = NoOpt() ? spv_bin : spirv::Optimize(spv_bin);
  if (tmp.spirv.empty()) return false;
  tmp.ok = true;
  r = std::move(tmp);
  return true;
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
