/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 -> GLCompute SPIR-V. See rdna_compute.h.
 */

#include "gpu/ps5/rdna/rdna_compute.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::rdna {

gpu::gcn::RecompiledCs RecompileCompute(const uint32_t*,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t) {
  return {};  // no SPIR-V backend: the caller skips the dispatch
}

}  // namespace gpu::rdna
#else

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
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
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kGpuSpirvNoopt, "DELTA_GPU_SPIRV_NOOPT", false);
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

// One set-0 storage buffer per distinct descriptor, mirroring the GFX7 planner
// (gpu/gcn PlanCsResources) with RDNA2's SMEM decode.
//
// Every descriptor must sit INLINE in the COMPUTE_USER_DATA window: the PS5
// dispatch path resolves a resource by reading base_sgpr straight out of that
// window, so a descriptor an s_load produced (an SRT chain) or one a scalar op
// moved into place would resolve to whatever those registers happen to hold. A
// compute dispatch writes guest memory, so that is declined, not guessed.
bool PlanResources(const Program& program,
                   uint32_t lds_dwords,
                   uint32_t user_sgpr,
                   RecompiledCs& r,
                   std::unordered_map<uint32_t, uint32_t>& bind) {
  bool scalar_written[136] = {};
  const auto inline_user_data = [&](uint32_t sgpr, uint32_t dwords) {
    if (sgpr + dwords > user_sgpr || sgpr + dwords > 136)
      return false;
    for (uint32_t k = 0; k < dwords; k++)
      if (scalar_written[sgpr + k])
        return false;
    return true;
  };

  std::unordered_map<uint32_t, uint32_t> resource_by_descriptor;
  const auto resource = [&](uint32_t pc, uint32_t base_sgpr, uint32_t dwords,
                            uint8_t kind, bool written, uint32_t min_bytes,
                            bool replayable = false) {
    // A descriptor an SRT chain produced is resolved at dispatch time by
    // replaying the shader's scalar ops (rdna::ResolveBuffers), keyed on this
    // pc. Only one that is neither inline nor replayable is declined, and the
    // dispatch path does that check.
    if (!inline_user_data(base_sgpr, dwords) && !replayable) {
      gpu::gcn::WarnUnsupported("cs.descriptor-not-inline.rdna", base_sgpr);
      return false;
    }
    const uint32_t key = (base_sgpr << 8) | kind;
    const auto it = resource_by_descriptor.find(key);
    if (it != resource_by_descriptor.end()) {
      CsResource& res = r.resources[it->second];
      res.written = res.written || written;
      res.min_bytes = std::max(res.min_bytes, min_bytes);
      bind[pc] = it->second;
      return true;
    }
    const uint32_t idx = static_cast<uint32_t>(r.resources.size());
    if (idx >= kMaxCsResources) {
      gpu::gcn::WarnUnsupported("cs.resource-count", idx + 1);
      return false;
    }
    resource_by_descriptor[key] = idx;
    bind[pc] = idx;
    r.resources.push_back({base_sgpr, pc, idx, kind, written, min_bytes});
    return true;
  };

  for (const Inst& inst : program) {
    const uint32_t w = inst.raw[0], w1 = inst.raw[1];
    switch (inst.enc) {
      case Enc::kSmrd: {
        const Smem smem = DecodeSmem(inst);
        const uint32_t n = SmemLoadCount(smem.op);
        // A negative immediate reads below the descriptor's base, which the
        // staged range cannot cover.
        if (!n || smem.offset < 0)
          return false;
        const bool buffer = smem.op >= 0x08;  // s_buffer_load_* takes a V#
        // The immediate is a BYTE offset on RDNA2 (a dword index on GFX7).
        if (!resource(inst.pc, smem.sbase, buffer ? 4 : 2, buffer ? 0 : 2,
                      false, static_cast<uint32_t>(smem.offset) + n * 4))
          return false;
        break;
      }
      case Enc::kMubuf: {
        const uint32_t op = inst.opcode;
        const bool load = op <= 0x03 || (op >= 0x08 && op <= 0x0f);
        const bool store = (op >= 0x04 && op <= 0x07) || op == 0x18 ||
                           op == 0x1a || (op >= 0x1c && op <= 0x1f);
        // Anything else is an atomic or a d16 form, and the shared emitter
        // masks the opcode to 7 bits, so the 0x80+ format variants would alias
        // a plain load. LDS and TFE add a destination it has no model for.
        if ((!load && !store) || ((w >> 16) & 1) || ((w1 >> 23) & 1))
          return false;
        if (!resource(inst.pc, ((w1 >> 16) & 0x1F) * 4, 4, 0, store, 0))
          return false;
        break;
      }
      case Enc::kMtbuf: {
        const uint32_t op = inst.opcode;
        if (op > 0x07 || ((w1 >> 23) & 1))
          return false;  // op[3] selects the d16 forms
        if (!resource(inst.pc, ((w1 >> 16) & 0x1F) * 4, 4, 0, op >= 4, 0))
          return false;
        break;
      }
      case Enc::kDs:
        // ds_swizzle is a cross-lane move rather than an LDS access, so it is
        // the one DS op that runs without an LDS allocation. GDS is not
        // modelled at all.
        if (((w >> 17) & 1) || (!lds_dwords && inst.opcode != 0x35))
          return false;
        break;
      case Enc::kMimg: {
        // The shared emitter reads the T# with the gfx10.3 field positions when
        // the translator is in RDNA mode, so the plan is the same as GCN's.
        const uint32_t op = inst.opcode;
        const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
        if (op == 0x0e)
          break;  // get_resinfo reads only descriptor SGPRs
        const bool store = op == 0x08 || op == 0x09;
        const bool load = op == 0x00 || op == 0x01;
        const bool sample = op == 0x24 || op == 0x27;
        if ((!store && !load && !sample) || ((w >> 15) & 1) || srsrc + 7 >= 136) {
          gpu::gcn::WarnUnsupported("mimg.cs.rdna", inst.opcode, w, w1);
          return false;
        }
        if (!resource(inst.pc, srsrc, 8, 1, store, 0, true))
          return false;
        break;
      }
      case Enc::kFlat:
        // global_/scratch_ addressing is a raw 64-bit pointer, which the
        // descriptor-bound resource model cannot express.
        gpu::gcn::WarnUnsupported("flat.cs.rdna", inst.opcode, w, w1);
        return false;
      default:
        break;
    }
    const ScalarWrite sw = DecodeScalarWrite(inst);
    for (uint32_t k = 0; k < sw.count && sw.first + k < 136; k++)
      scalar_written[sw.first + k] = true;
  }
  return true;
}

// s_load / s_buffer_load: scalar reads of guest memory, served from the storage
// buffer planned for this instruction.
void EmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const Smem smem = DecodeSmem(inst);
  const uint32_t n = SmemLoadCount(smem.op);
  const int b = gpu::gcn::CsBindingFor(sc, inst.pc);
  if (!n || b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  // Both offsets are in BYTES on RDNA2 (GFX7's immediate was a dword index);
  // soffset field 125 is NULL, meaning no register offset at all.
  const Id immediate = t.U32(static_cast<uint32_t>(smem.offset));
  const Id byte_off = smem.soffset == 125
                          ? immediate
                          : t.Add(t.SrcRaw(smem.soffset, 0), immediate);
  const Id dword0 = t.Shr(byte_off, t.U32(2));
  for (uint32_t k = 0; k < n; k++)
    t.SetSdst(smem.sdst, k,
              gpu::gcn::CsSsboLoad(t, sc, static_cast<uint32_t>(b),
                                   t.Add(dword0, t.U32(k))));
}

bool NoOpt() {
  return kGpuSpirvNoopt;
}

// A declined dispatch leaves the buffers it was meant to fill untouched, which
// is otherwise invisible in the frame. One line per distinct shader.
void ReportDecline(const uint32_t* cs_code, size_t insts) {
  static std::unordered_set<uint64_t> reported;
  const uint64_t address = reinterpret_cast<uintptr_t>(cs_code);
  if (!reported.insert(address).second)
    return;
  std::fprintf(stderr,
               "[rdnacs] declined CS @%#lx (%zu insts); dispatch skipped\n",
               static_cast<unsigned long>(address), insts);
}

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
  sc.lds_dwords = lds_dwords * 128;  // RSRC2.LDS_SIZE is in 128-dword granules
  if (!PlanResources(program, sc.lds_dwords, user_sgpr, r, sc.cs_bind) ||
      r.resources.empty())
    return false;

  bool uses_ds_swizzle = false;
  for (const Inst& inst : program)
    if (inst.enc == Enc::kDs && inst.opcode == 0x35) {
      uses_ds_swizzle = true;
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
  switch (inst.enc) {
    case Enc::kSmrd:
      EmitSmem(t, inst, sc);
      return true;
    case Enc::kMubuf:
      gpu::gcn::EmitCsMubuf(t, inst, sc);
      return true;
    case Enc::kMtbuf:
      gpu::gcn::EmitCsMtbuf(t, inst, sc);
      return true;
    case Enc::kDs:
      gpu::gcn::EmitDs(t, inst, sc);
      return true;
    case Enc::kMimg: {
      // Same lowering as the graphics path: dim 3 (cube) and 5 (2D array)
      // become the shared emitter's DA bit, and NSA names each address
      // component in its own VGPR.
      const uint32_t w = inst.raw[0];
      const uint32_t dim = (w >> 3) & 0x7;
      const bool arrayed_dim = dim == 3 || dim == 5;
      Inst lowered = inst;
      lowered.raw[0] = (w & ~0x4000u) | (arrayed_dim ? 0x4000u : 0u);
      const uint32_t nsa = (w >> 1) & 0x3;
      if (nsa) {
        std::array<gpu::gcn::Id, 13> address{};
        address[0] = t.Vg(inst.raw[1] & 0xFF);
        for (uint32_t d = 0; d < nsa; d++)
          for (uint32_t c = 0; c < 4; c++)
            address[1 + d * 4 + c] = t.Vg((inst.raw[2 + d] >> (c * 8)) & 0xFF);
        gpu::gcn::EmitCsMimg(t, lowered, sc, address.data());
        return true;
      }
      gpu::gcn::EmitCsMimg(t, lowered, sc);
      return true;
    }
    case Enc::kFlat:
      sc.cs_unsupported = true;  // the plan already declined these
      return true;
    default:
      return false;
  }
}

gpu::gcn::RecompiledCs RecompileCompute(const uint32_t* cs_code,
                                        uint32_t num_thread_x,
                                        uint32_t num_thread_y,
                                        uint32_t num_thread_z,
                                        uint32_t user_sgpr,
                                        uint32_t tgid_enable,
                                        uint32_t lds_dwords) {
  RecompiledCs r;
  constexpr uint64_t kMaxShaderBytes = 4096 * sizeof(uint32_t);
  if (!cs_code || !gpu::IsReadableRange(reinterpret_cast<uintptr_t>(cs_code),
                                        kMaxShaderBytes))
    return r;
  const Program program = ReachableProgram(DecodeShader(cs_code, 4096));

  Translator t;
  RecompiledCs tmp;  // build into a temp so a mid-emit failure leaves r intact
  gpu::gcn::ResetUnsupported();
  if (!TranslateCs(program, num_thread_x, num_thread_y, num_thread_z, user_sgpr,
                   tgid_enable, lds_dwords, tmp, t) ||
      gpu::gcn::HadUnsupported()) {
    ReportDecline(cs_code, program.size());
    return r;
  }

  const std::vector<uint32_t> spv_bin = t.m.Assemble();
  // A module the translator emitted but the validator rejects is a translator
  // bug (wrong codegen, not a guest gap): always loud.
  std::string err;
  if (!gpu::gcn::spirv::Validate(spv_bin, &err)) {
    std::fprintf(stderr, "[rdnacs] CS invalid @%p: %s\n",
                 static_cast<const void*>(cs_code), err.c_str());
    return r;
  }
  tmp.spirv = NoOpt() ? spv_bin : gpu::gcn::spirv::Optimize(spv_bin);
  if (tmp.spirv.empty())
    return r;
  tmp.ok = true;
  return tmp;
}

}  // namespace gpu::rdna

#endif  // DELTA_HAVE_SPIRV_BACKEND
