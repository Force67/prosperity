/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> SPIR-V: memory-op emitters.
 *
 * Graphics: SMRD s_buffer_load reads become UBO ("cbuffer") reads; MIMG
 * sample/load/resinfo become combined-image-sampler ops (set 0, binding ==
 * MIMG order, matching TrackTextures).
 *
 * Compute: guest memory is modelled as per-resource storage buffers that alias
 * the guest range behind each descriptor ([base, base+size) -> buffer offset
 * 0); LDS is a Workgroup-storage uint array. Anything the model cannot express
 * (atomics, sampling) declines the recompile via StageContext::cs_unsupported.
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include <initializer_list>
#include "base/arch.h"
#include <algorithm>

#include "gpu/gcn/gcn_audit.h"
#include "gpu/gcn/spirv/translator.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(float, kDebugUv, "DELTA_GPU_DEBUGUV", 0.f);
}  // namespace

namespace {
DELTA_OPTION(int, kPsTexBind, "DELTA_GPU_PSTEX", 0);
DELTA_OPTION(int, kIdxStride, "DELTA_GPU_IDXSTRIDE", 0);
}  // namespace

namespace gpu::gcn {
namespace {

// ---- SSBO access ------------------------------------------------------------
// Shared by the compute resource model and the graphics raw-buffer model: both
// alias a guest range as Buf { uint data[]; } and address it by dword index.
Id SsboPtr(Translator& t, Id var, Id dword_idx) {
  const Id p_u = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
  return t.m.AccessChain(p_u, var, {t.U32(0), dword_idx});
}
Id SsboLoad(Translator& t, Id var, Id dword_idx) {
  return t.m.Load(t.t_u, SsboPtr(t, var, dword_idx));
}
}  // namespace

// Exported: the compute resource model itself is ISA-neutral (a guest range
// aliased as Buf { uint data[]; }, addressed by dword index). The RDNA2 path
// binds the same buffers and only decodes its own scalar loads differently.
Id CsSsboPtr(Translator& t, StageContext& sc, u32 binding, Id dword_idx) {
  return SsboPtr(t, sc.cs_ssbo[binding], dword_idx);
}
// MUBUF atomics (GFX7): 0x30..0x3f on 32-bit values, 0x50..0x5f on 64-bit
// pairs. GLC returns the pre-op value into VDATA; without it the result is
// discarded, which SPIR-V expresses the same way (the result id is unused).
bool MubufAtomic(u32 op) {
  return (op >= 0x30 && op <= 0x3f) || (op >= 0x50 && op <= 0x5f);
}

Id CsSsboLoad(Translator& t, StageContext& sc, u32 binding, Id dword_idx) {
  return SsboLoad(t, sc.cs_ssbo[binding], dword_idx);
}
void CsSsboStore(Translator& t,
                 StageContext& sc,
                 u32 binding,
                 Id dword_idx,
                 Id value) {
  t.m.Store(CsSsboPtr(t, sc, binding, dword_idx), value);
}
int CsBindingFor(StageContext& sc, u32 pc) {
  auto it = sc.cs_bind.find(pc);
  return it != sc.cs_bind.end() ? static_cast<int>(it->second) : -1;
}

namespace {

Id Max1(Translator& t, Id value) {
  return t.UMax(value, t.U32(1));
}

// Next power of two >= max(value, 1) (bit-smearing form).
Id BitCeil(Translator& t, Id value) {
  Id v = t.Sub(Max1(t, value), t.U32(1));
  for (u32 s : {1u, 2u, 4u, 8u, 16u})
    v = t.Or(v, t.Shr(v, t.U32(s)));
  return t.Add(v, t.U32(1));
}

Id LinearMipPitch(Translator& t,
                  Id base_pitch,
                  Id height,
                  Id mip,
                  Id linear_general,
                  Id pow2_pad) {
  Id raw = Max1(t, t.Shr(base_pitch, mip));
  raw = t.SelectB(pow2_pad, BitCeil(t, raw), raw);
  Id aligned = t.And(t.Add(raw, t.U32(15)), t.U32(~15u));
  for (u32 i = 0; i < 3; i++) {
    const Id ok = t.IsZero(t.And(t.Mul(aligned, height), t.U32(63)));
    aligned = t.SelectB(ok, aligned, t.Add(aligned, t.U32(16)));
  }
  return t.SelectB(linear_general, raw, aligned);
}

// Byte offset of mip `physical_mip`'s first texel: the sizes of every level
// below it, summed.
//
// Emitted as a real loop, not unrolled over all 16 possible levels. Each
// iteration costs ~55 SPIR-V instructions (two BitCeils and a three-round
// 64-byte alignment search), so the unrolled form was ~880 per image access --
// which is what made Shadow of the Colossus's pixel shaders 190K words apiece
// and cost half a second each in spirv-opt and the same again in the driver.
// The accumulator is a Private variable, like the dispatch loop's state, so
// legalization turns it back into a phi.
Id MipChainOffset(Translator& t,
                  Id base_pitch,
                  Id base_height,
                  Id layers,
                  Id physical_mip,
                  Id linear_general,
                  Id pow2_pad) {
  const Id acc = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                              t.m.ConstNull(t.t_u));
  const Id level_var = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                    t.m.ConstNull(t.t_u));
  t.m.Store(acc, t.U32(0));
  t.m.Store(level_var, t.U32(0));
  const Id levels = t.UMin(physical_mip, t.U32(16));
  const Id head = t.m.NewBlock(), body = t.m.NewBlock(),
           cont = t.m.NewBlock(), merge = t.m.NewBlock();
  t.m.Branch(head);
  t.m.OpenBlock(head);
  const Id level = t.m.Load(t.t_u, level_var);
  const Id more = t.Ult(level, levels);
  t.m.LoopMerge(merge, cont);
  t.m.BranchConditional(more, body, merge);
  t.m.OpenBlock(body);
  const Id level_height = Max1(t, t.Shr(base_height, level));
  const Id level_stored =
      t.SelectB(pow2_pad, BitCeil(t, level_height), level_height);
  const Id level_pitch =
      LinearMipPitch(t, base_pitch, level_stored, level, linear_general,
                     pow2_pad);
  t.m.Store(acc, t.Add(t.m.Load(t.t_u, acc),
                       t.Mul(t.Mul(level_pitch, level_stored), layers)));
  t.m.Branch(cont);
  t.m.OpenBlock(cont);
  t.m.Store(level_var, t.Add(level, t.U32(1)));
  t.m.Branch(head);
  t.m.OpenBlock(merge);
  return t.m.Load(t.t_u, acc);
}

// MUBUF/MTBUF shared addressing: byte offset within the bound resource.
// soffset + instruction offset, plus the vaddr index/offset registers per the
// IDXEN/OFFEN bits (both set: vaddr = index, vaddr+1 = offset).
Id BufferByteOffset(Translator& t,
                    const Inst& inst,
                    u32 inst_offset,
                    bool idxen,
                    bool offen,
                    u32 vaddr,
                    u32 srsrc,
                    u32 soffset_field) {
  Id byte_off =
      t.Add(t.SrcRaw(soffset_field, inst.literal), t.U32(inst_offset));
  u32 va = vaddr;
  if (idxen) {
    // DELTA_GPU_IDXSTRIDE: force the indexed stride instead of reading it from
    // the V#'s SGPR. A V# that arrives through s_load (an SRT chain) is never
    // written into the SGPR file by the graphics path, so the stride there is
    // whatever user data happened to sit in that slot -- diagnostic.
    const Id stride =
        kIdxStride ? t.U32(static_cast<u32>(kIdxStride))
                   : t.And(t.Shr(t.Sg(srsrc + 1), t.U32(16)), t.U32(0x3FFF));
    byte_off = t.Add(byte_off, t.Mul(t.Vg(va++), stride));
  }
  if (offen)
    byte_off = t.Add(byte_off, t.Vg(va));
  return byte_off;
}

// A typed buffer op carries its own dfmt/nfmt, overriding the V#'s. Both buffer
// models move the components as raw dwords, which is only exact when the format
// really is one unconverted 32-bit dword per component; anything packed would
// need real conversion, so those still warn.
bool MtbufIsRawDwords(const Inst& inst, u32 n) {
  const u32 w = inst.raw[0];
  if (inst.opcode >= 8)  // d16 forms pack two components per dword
    return false;
  const u32 nfmt = (w >> 23) & 0x7;
  if (nfmt != 4 && nfmt != 5 && nfmt != 7)  // uint / sint / float
    return false;
  switch ((w >> 19) & 0xF) {
    case 4:
      return n == 1;  // 32
    case 11:
      return n == 2;  // 32_32
    case 13:
      return n == 3;  // 32_32_32
    case 14:
      return n == 4;  // 32_32_32_32
    default:
      return false;
  }
}

// Sub-dword read out of a dword-granular SSBO: value = bits [off*8 .. off*8+n)
// of the containing dword. `byte_off` may be misaligned only within the dword.
Id LoadSubDword(Translator& t,
                Id var,
                Id byte_off,
                u32 bits,
                bool sign_extend) {
  const Id word = SsboLoad(t, var, t.Shr(byte_off, t.U32(2)));
  const Id shift = t.Shl(t.And(byte_off, t.U32(3)), t.U32(3));
  if (sign_extend)
    return t.m.Bitcast(
        t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                        {t.m.Bitcast(t.t_i, word), shift, t.U32(bits)}));
  return t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u,
                  {word, shift, t.U32(bits)});
}

// Sub-dword store: read-modify-write the containing dword.
void StoreSubDword(Translator& t,
                   StageContext& sc,
                   u32 binding,
                   Id byte_off,
                   u32 bits,
                   Id value) {
  const Id idx = t.Shr(byte_off, t.U32(2));
  const Id shift = t.Shl(t.And(byte_off, t.U32(3)), t.U32(3));
  const Id old = CsSsboLoad(t, sc, binding, idx);
  const Id inserted = t.m.Emit(spv::Op::OpBitFieldInsert, t.t_u,
                               {old, value, shift, t.U32(bits)});
  CsSsboStore(t, sc, binding, idx, inserted);
}

}  // namespace

// ---- graphics: SMRD -> cbuffer ---------------------------------------------
u32 SmrdLoadCount(u32 op) {
  if (op < 0x08 || op > 0x0c)
    return 0;  // s_buffer_load_dword{,x2,x4,x8,x16}
  return op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
}

// Dwords moved by any SMRD scalar load: s_load_dword{,x2,x4,x8,x16} (op
// 0x00..0x04, addressed through a 2-dword flat pointer) as well as the
// s_buffer_load family. Both feed the same UBO window; only the descriptor
// they are addressed through differs.
u32 SmrdDwordCount(u32 op) {
  return op <= 0x04 ? (1u << op) : SmrdLoadCount(op);
}

// Cbuffer bindings are keyed by the SGPR the descriptor is read from. The same
// SGPR can hold a flat pointer for one load and a V# for another (user data
// s[0:1] as a table pointer, s[0:3] as a buffer), so the descriptor kind is
// part of the key -- otherwise whichever load came first would capture the
// other's window.
u32 CbufBindKey(u32 base_sgpr, bool pointer) {
  return base_sgpr | (pointer ? 0x100u : 0u);
}

void EmitCbufSmrd(Translator& t,
                  const Inst& inst,
                  const std::unordered_map<u32, u32>& bindings) {
  const u32 w = inst.raw[0], n = SmrdDwordCount(inst.opcode);
  const u32 sdst = (w >> 15) & 0x7F, base_sgpr = ((w >> 9) & 0x3F) * 2;
  const bool imm = (w >> 8) & 1;
  const auto it = bindings.find(CbufBindKey(base_sgpr, inst.opcode <= 0x04));
  if (!n || it == bindings.end())
    return;
  const SmrdOffset so = DecodeSmrdOffset(inst);
  if (!so.in_sgpr) {
    for (u32 i = 0; i < n; i++)
      t.SetSg(sdst + i, t.CbufDword(it->second, so.dwords + i));
  } else {
    const Id base = t.Shr(t.SrcRaw(so.sgpr, inst.literal), t.U32(2));
    for (u32 i = 0; i < n; i++)
      t.SetSg(sdst + i, t.CbufDwordId(it->second, t.Add(base, t.U32(i))));
  }
}

bool PlanCbufs(const Program& program,
               u32 first_binding,
               std::vector<ShaderCbuf>& cbufs,
               std::unordered_map<u32, u32>& bindings,
               const u8* reachable) {
  u32 index = 0;
  for (const Inst& inst : program) {
    if (reachable && !reachable[index]) {
      index++;
      continue;
    }
    index++;
    if (inst.enc != Enc::kSmrd)
      continue;
    const u32 n = SmrdDwordCount(inst.opcode);
    if (!n)
      continue;
    const u32 w = inst.raw[0], base_sgpr = ((w >> 9) & 0x3F) * 2;
    // The descriptor may live above user data when an earlier scalar load put
    // it there. Draw-time scalar replay resolves both V#s and flat pointers at
    // the consuming load, so do not discard those chained descriptors.
    const bool pointer = inst.opcode <= 0x04;
    const auto [it, inserted] = bindings.emplace(
        CbufBindKey(base_sgpr, pointer), first_binding + cbufs.size());
    if (inserted) {
      if (it->second >= kMaxCbufBindings) {
        // Say so: a shader declined here has no unsupported instruction, so an
        // audit that only lists ops reports it as rejected for no reason.
        WarnUnsupported("smrd.cbuf-binding-count", it->second, w, inst.raw[1]);
        return false;
      }
      ShaderCbuf cb{};
      cb.binding = it->second;
      cb.ud_sgpr = base_sgpr;
      cb.pointer = pointer;
      cbufs.push_back(cb);
    }
    const SmrdOffset so = DecodeSmrdOffset(inst);
    const u32 end = so.in_sgpr ? 256 : so.dwords + n;
    for (ShaderCbuf& cb : cbufs)
      if (cb.binding == it->second)
        cb.num_dwords = std::max(cb.num_dwords, end);
  }
  return true;
}

// ---- graphics: MIMG ---------------------------------------------------------
// References the texture as a combined sampler at set 0. The binding comes
// from the shared MimgBindingPlan (one binding per unique descriptor, matching
// TrackTextures); the variable is created on the binding's first use. MIMG DA
// selects a 2D-array resource and appends the layer coordinate after x/y. A 3D
// resource also takes a third coordinate, but says so nowhere in the
// instruction (its DA bit is 0): only the T# type distinguishes it, so it
// reaches the translator out of band in StageContext::tex_3d_mask.
// Does this MIMG op state its own LOD? Outside a fragment shader there are no
// implicit derivatives, so only these may run in a VS. In both the sample block
// (0x20-0x3f) and the gather4 block (0x40-0x5f) the low three bits are the LOD
// mode: 4 = _l (explicit level), 7 = _lz (level zero). Gathers admit only _lz,
// since EmitMimg translates no other gather form. image_load[_mip] indexes a
// level directly.
bool MimgNamesItsLod(u32 op) {
  if (op == 0x00 || op == 0x01)
    return true;
  const u32 lod_mode = op & 0x7;
  if (op >= 0x40 && op <= 0x5f)
    return lod_mode == 7;
  return op >= 0x20 && op <= 0x3f && (lod_mode == 4 || lod_mode == 7);
}

void EmitMimg(Translator& t,
              const Inst& inst,
              StageContext& sc,
              const Id* address) {
  const u32 w0 = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  const u32 dmask = (w0 >> 8) & 0xF, vaddr = w1 & 0xFF;
  const u32 vdata = (w1 >> 8) & 0xFF;
  // MIMG lays the sample block (0x20-0x3f) and the gather4 block (0x40-0x5f)
  // out the same way: bit 3 adds the z-compare, bit 4 adds the texel offset,
  // and the low three bits pick the LOD mode.
  //
  // Only the _lz forms (low three bits == 7) are translated. Two reasons, both
  // load-bearing: OpImageGather is fixed at level zero, so a form naming a LOD
  // or a bias cannot be honoured; and the ISA orders vaddr[] as
  // "Offsets, bias, zpcf, then coordinates", so a _b form carries a bias word
  // this emitter does not account for -- it would read the bias as the
  // z-compare and every coordinate one word early. Gather is NOT inherently
  // level zero (the ISA describes software trilinear via two gathers a LOD
  // step apart); the rest fall through `known` below and decline loudly.
  const bool gather = op >= 0x40 && op <= 0x5f && (op & 0x7) == 7;
  const bool dref = op == 0x28 || op == 0x2f || (gather && (op & 0x08));
  const bool offset = op == 0x37 || (gather && (op & 0x10));
  const auto addr_u = [&](u32 index) {
    return address ? address[index] : t.Vg(vaddr + index);
  };
  const auto addr_f = [&](u32 index) {
    return t.m.Bitcast(t.t_f, addr_u(index));
  };

  u32 bind = StageContext::kMaxPsSamplers;
  if (sc.mimg_plan) {
    const auto plan_it = sc.mimg_plan->binding_by_pc.find(inst.pc);
    if (plan_it != sc.mimg_plan->binding_by_pc.end())
      bind = plan_it->second;
  }
  if (bind >= StageContext::kMaxPsSamplers) {
    WarnUnsupported("mimg.unplanned", op, w0, w1);
    return;
  }

  const bool is_3d = ((sc.tex_3d_mask >> bind) & 1u) != 0;
  // An arrayed 3D sampled image is not a legal SPIR-V type; a 3D T# leaves DA
  // clear anyway, so the descriptor wins over a stray DA bit.
  const bool arrayed = (w0 & 0x4000) != 0 && !is_3d;
  const u32 coord_components = arrayed || is_3d ? 3u : 2u;
  // A 1D T# is bound as a height-1 2D image: the address body carries x
  // (+layer), and y is synthesized at the row centre. Bloodborne's gamma pass
  // samples a 1025x1 R32F 1D-array LUT per channel; reading the body as 2D
  // takes the layer index for y and samples nothing.
  const bool is_1d = ((sc.tex_1d_mask >> bind) & 1u) != 0 && !is_3d;
  const u32 addr_components =
      is_1d ? (arrayed ? 2u : 1u) : coord_components;
  // OpImageSampleDref* / OpImageGather are undefined on Dim3D.
  if (is_3d && (dref || gather)) {
    WarnUnsupported("mimg.3d-dref-gather", op, w0, w1);
    return;
  }

  if (op == 0x08 || op == 0x09) {  // image_store[_mip]
    const u32 type_idx = arrayed ? 1u : 0u;
    if (!t.storage_img_types[type_idx]) {
      t.m.Capability(spv::Capability::StorageImageWriteWithoutFormat);
      t.storage_img_types[type_idx] =
          t.m.TypeImage(t.t_f, spv::Dim::Dim2D, 0, arrayed ? 1 : 0, 0, 2,
                        spv::ImageFormat::Unknown);
      t.storage_img_ptrs[type_idx] = t.m.TypePointer(
          spv::StorageClass::UniformConstant, t.storage_img_types[type_idx]);
    }
    if (!sc.tex_vars[bind]) {
      const Id var = t.m.Variable(t.storage_img_ptrs[type_idx],
                                  spv::StorageClass::UniformConstant);
      t.m.Decorate(var, spv::Decoration::DescriptorSet, {0});
      t.m.Decorate(var, spv::Decoration::Binding, {bind});
      t.m.Name(var, "img" + std::to_string(bind));
      sc.tex_vars[bind] = var;
      sc.tex_types[bind] = type_idx;
    }
    const Id ix = t.m.Bitcast(t.t_i, addr_u(0));
    const Id iy =
        is_1d ? t.m.Bitcast(t.t_i, t.U32(0)) : t.m.Bitcast(t.t_i, addr_u(1));
    const Id coord =
        arrayed
            ? t.m.CompositeConstruct(
                  t.m.TypeVec(t.t_i, 3),
                  {ix, iy, t.m.Bitcast(t.t_i, addr_u(is_1d ? 1 : 2))})
            : t.m.CompositeConstruct(t.m.TypeVec(t.t_i, 2), {ix, iy});
    Id components[4] = {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(1.f)};
    u32 source = 0;
    for (u32 channel = 0; channel < 4; channel++)
      if (dmask & (1u << channel))
        components[channel] = t.VgF(vdata + source++);
    const Id texel = t.m.CompositeConstruct(
        t.t_v4, {components[0], components[1], components[2], components[3]});
    const Id image = t.m.Load(t.storage_img_types[type_idx], sc.tex_vars[bind]);
    if (!t.predicate_vector) {
      t.m.EmitVoid(spv::Op::OpImageWrite, {image, coord, texel});
      return;
    }
    const Id live = t.LaneActive(t.Exec());
    const Id write = t.m.NewBlock(), merge = t.m.NewBlock();
    t.m.SelectionMerge(merge);
    t.m.BranchConditional(live, write, merge);
    t.m.OpenBlock(write);
    t.m.EmitVoid(spv::Op::OpImageWrite, {image, coord, texel});
    t.m.Branch(merge);
    t.m.OpenBlock(merge);
    return;
  }

  if (!sc.tex_vars[bind]) {
    // A depth-compare read of an integer image is meaningless, so dref wins.
    const bool int_img =
        !dref && ((sc.tex_uint_mask >> bind) & 1u) != 0;
    const u32 type_idx = (arrayed ? 1u : 0u) | (dref ? 2u : 0u) |
                              (is_3d ? 4u : 0u) | (int_img ? 8u : 0u);
    if (!t.img_types[type_idx]) {
      t.img_types[type_idx] = t.m.TypeImage(
          int_img ? t.t_u : t.t_f, is_3d ? spv::Dim::Dim3D : spv::Dim::Dim2D,
          dref ? 1 : 0, arrayed ? 1 : 0, 0, 1, spv::ImageFormat::Unknown);
      t.sampled_types[type_idx] = t.m.TypeSampledImage(t.img_types[type_idx]);
      t.sampled_ptrs[type_idx] = t.m.TypePointer(
          spv::StorageClass::UniformConstant, t.sampled_types[type_idx]);
    }
    const Id var = t.m.Variable(t.sampled_ptrs[type_idx],
                                spv::StorageClass::UniformConstant);
    t.m.Decorate(var, spv::Decoration::DescriptorSet, {0});
    t.m.Decorate(var, spv::Decoration::Binding, {bind + sc.tex_binding_base});
    t.m.Name(var, "tex" + std::to_string(bind + sc.tex_binding_base));
    sc.tex_vars[bind] = var;
    sc.tex_types[bind] = type_idx;
  }
  const u32 type_idx = sc.tex_types[bind];
  const Id img_ty = t.img_types[type_idx];
  const Id si = t.m.Load(t.sampled_types[type_idx], sc.tex_vars[bind]);
  // An integer image samples to a uvec4 and its texels go to the VGPRs as raw
  // bits: the shader packed them, and reinterpreting them as floats is exactly
  // the loss this path exists to avoid.
  const bool int_img = (type_idx & 8u) != 0;
  const Id texel_ty = int_img ? t.m.TypeVec(t.t_u, 4) : t.t_v4;

  if (op == 0x0e) {  // image_get_resinfo: dimensions/levels for mip v[vaddr]
    t.RequireImageQuery();
    const Id img = t.m.Emit(spv::Op::OpImage, img_ty, {si});
    const Id levels = t.m.Emit(spv::Op::OpImageQueryLevels, t.t_u, {img});
    const Id lod = t.UMin(addr_u(0), t.Sub(t.UMax(levels, t.U32(1)), t.U32(1)));
    const Id size_ty = t.m.TypeVec(t.t_u, coord_components);
    const Id size = t.m.Emit(spv::Op::OpImageQuerySizeLod, size_ty, {img, lod});
    const Id comps[4] = {
        t.m.CompositeExtract(t.t_u, size, 0),  // width
        t.m.CompositeExtract(t.t_u, size, 1),  // height
        coord_components == 3 ? t.m.CompositeExtract(t.t_u, size, 2)
                              : t.U32(1),  // depth / layers
        levels,                            // mips
    };
    u32 out = 0;
    for (int i = 0; i < 4; i++)
      if (dmask & (1 << i))
        t.SetVg(vdata + out++, comps[i]);
    return;
  }

  // The ISA fixes the vaddr[] order as "Offsets, bias, zpcf, then coordinates",
  // so a compared-and-offset form (image_gather4_c_lz_o) finds its z-compare at
  // word 1, not word 0. Bias-carrying forms are rejected above rather than
  // modelled, which is why no bias word is accounted for here.
  const u32 dref_index = offset ? 1u : 0u;
  // _D / _CD carry user derivatives between the z-compare and the coordinates:
  // two words per sampled dimension. A compute stage has no implicit
  // derivatives, so these are the only sample forms it can use -- GTA:SA's
  // deferred lighting dispatch is written entirely in them, and reading its
  // derivative words as coordinates is how the whole pass came out black.
  const bool derivs = op == 0x22 || op == 0x2a || op == 0x68 || op == 0x6a;
  const u32 deriv_dims = is_1d ? 1u : (is_3d ? 3u : 2u);
  const u32 deriv_words = derivs ? deriv_dims * 2u : 0u;
  const u32 deriv_index = (offset ? 1u : 0u) + (dref ? 1u : 0u);
  const u32 body_addr =
      vaddr + (offset ? 1u : 0u) + (dref ? 1u : 0u) + deriv_words;
  const u32 body_index = body_addr - vaddr;
  Id x = addr_f(body_index);
  Id y = is_1d ? t.F32(0.5f) : addr_f(body_index + 1);
  if (offset) {
    // GFX7 packs signed six-bit X/Y texel offsets before the address body.
    // Vulkan 1.1 does not permit a dynamic Offset operand on OpImageSample*,
    // so apply the normalized-coordinate equivalent using the level-0 size.
    t.RequireImageQuery();
    const Id packed = t.m.Bitcast(t.t_i, addr_u(0));
    const Id ox = t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                           {packed, t.U32(0), t.U32(6)});
    const Id oy = t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                           {packed, t.U32(8), t.U32(6)});
    const Id image = t.m.Emit(spv::Op::OpImage, img_ty, {si});
    const Id size_ty = t.m.TypeVec(t.t_u, coord_components);
    const Id size =
        t.m.Emit(spv::Op::OpImageQuerySizeLod, size_ty, {image, t.U32(0)});
    const Id sx = t.m.Emit(spv::Op::OpConvertUToF, t.t_f,
                           {t.m.CompositeExtract(t.t_u, size, 0)});
    const Id sy = t.m.Emit(spv::Op::OpConvertUToF, t.t_f,
                           {t.m.CompositeExtract(t.t_u, size, 1)});
    x = t.FAdd(x, t.FDiv(t.m.Emit(spv::Op::OpConvertSToF, t.t_f, {ox}), sx));
    if (!is_1d)
      y = t.FAdd(y, t.FDiv(t.m.Emit(spv::Op::OpConvertSToF, t.t_f, {oy}), sy));
  }
  const Id uv =
      coord_components == 3
          ? t.m.CompositeConstruct(
                t.t_v3, {x, y, addr_f(body_index + (is_1d ? 1 : 2))})
          : t.m.CompositeConstruct(t.t_v2, {x, y});
  const u32 lod_operand =
      static_cast<u32>(spv::ImageOperandsMask::Lod);
  const bool known = op == 0x00 || op == 0x01 || op == 0x20 || op == 0x21 ||
                     op == 0x24 || op == 0x25 || op == 0x27 || op == 0x28 ||
                     op == 0x2f || op == 0x37 || gather || derivs;
  if (!known)
    WarnUnsupported("mimg", op, w0, w1);
  // The Grad operand takes one vector per axis, sized like the sampled
  // dimension -- an array layer is a coordinate but not a derivative.
  const u32 grad_operand = static_cast<u32>(spv::ImageOperandsMask::Grad);
  const auto deriv_vec = [&](u32 first) {
    if (deriv_dims == 1)
      return addr_f(deriv_index + first);
    Id c[3];
    for (u32 i = 0; i < deriv_dims; i++)
      c[i] = addr_f(deriv_index + first + i);
    return t.m.CompositeConstruct(
        t.m.TypeVec(t.t_f, deriv_dims),
        deriv_dims == 2 ? std::vector<Id>{c[0], c[1]}
                        : std::vector<Id>{c[0], c[1], c[2]});
  };

  Id texel;
  if (op == 0x00 || op == 0x01) {  // image_load[_mip]: integer fetch
    const Id ix = t.m.Bitcast(t.t_i, addr_u(0));
    const Id iy =
        is_1d ? t.m.Bitcast(t.t_i, t.U32(0)) : t.m.Bitcast(t.t_i, addr_u(1));
    const Id ic =
        coord_components == 3
            ? t.m.CompositeConstruct(
                  t.m.TypeVec(t.t_i, 3),
                  {ix, iy, t.m.Bitcast(t.t_i, addr_u(is_1d ? 1 : 2))})
            : t.m.CompositeConstruct(t.m.TypeVec(t.t_i, 2), {ix, iy});
    const Id img = t.m.Emit(spv::Op::OpImage, img_ty, {si});
    Id lod = t.U32(0);
    if (op == 0x01) {
      t.RequireImageQuery();
      const Id levels = t.m.Emit(spv::Op::OpImageQueryLevels, t.t_u, {img});
      lod = t.UMin(addr_u(addr_components), t.Sub(levels, t.U32(1)));
    }
    texel =
        t.m.Emit(spv::Op::OpImageFetch, texel_ty, {img, ic, lod_operand, lod});
  } else if (op == 0x24) {  // image_sample_l: explicit LOD after the body
    texel = t.m.Emit(
        spv::Op::OpImageSampleExplicitLod, texel_ty,
        {si, uv, lod_operand,
         addr_f(is_1d ? body_index + addr_components : coord_components)});
  } else if (derivs && dref) {  // image_sample_c_[c]d
    texel = t.m.Emit(spv::Op::OpImageSampleDrefExplicitLod, t.t_f,
                     {si, uv, addr_f(dref_index), grad_operand,
                      deriv_vec(0), deriv_vec(deriv_dims)});
  } else if (derivs) {  // image_sample_[c]d
    texel = t.m.Emit(
        spv::Op::OpImageSampleExplicitLod, texel_ty,
        {si, uv, grad_operand, deriv_vec(0), deriv_vec(deriv_dims)});
  } else if (op == 0x28) {  // image_sample_c: z-compare precedes the body
    texel = t.m.Emit(spv::Op::OpImageSampleDrefImplicitLod, t.t_f,
                     {si, uv, addr_f(dref_index)});
  } else if (op == 0x2f) {  // image_sample_c_lz: PCF forced to level zero
    texel = t.m.Emit(spv::Op::OpImageSampleDrefExplicitLod, t.t_f,
                     {si, uv, addr_f(dref_index), lod_operand, t.F32(0.0f)});
  } else if (op == 0x27 || op == 0x37) {  // image_sample_lz[_o]: forced LOD 0
    texel = t.m.Emit(spv::Op::OpImageSampleExplicitLod, texel_ty,
                     {si, uv, lod_operand, t.F32(0.0f)});
  } else if (gather && dref) {
    // image_gather4_c*: four PCF comparisons, one per texel of the footprint.
    // OpImageDrefGather takes no component operand -- the compare result IS the
    // gathered value -- and yields a vec4 like the uncompared gather.
    texel = t.m.Emit(spv::Op::OpImageDrefGather, t.t_v4,
                     {si, uv, addr_f(dref_index)});
  } else if (gather) {  // DMASK selects the gathered channel
    u32 component = 0;
    while (component < 3 && !(dmask & (1u << component)))
      component++;
    texel =
        t.m.Emit(spv::Op::OpImageGather, texel_ty, {si, uv, t.U32(component)});
  } else {  // image_sample / _cl / _b (bias/derivs ignored): implicit LOD
    texel = t.m.Emit(spv::Op::OpImageSampleImplicitLod, texel_ty, {si, uv});
  }

  // DELTA_GPU_PSTEX=<binding+1>: remember this binding's texel so the PS
  // epilogue can export it (0 = the last sample, whatever it was).
  // Recorded through a Private variable so a sample taken inside a branch
  // still reaches the epilogue (the SSA value would not dominate it). An
  // integer sample is converted, because the point of the diagnostic is the
  // magnitude the shader received, not its bit pattern.
  if (!dref && !gather &&
      (kPsTexBind == 0 || bind == (u32)(kPsTexBind - 1)) &&
      t.last_texel_var) {
    Id v = texel;
    if (int_img) {
      Id c[4];
      for (u32 i = 0; i < 4; i++)
        c[i] = t.m.Emit(spv::Op::OpConvertUToF, t.t_f,
                        {t.m.CompositeExtract(t.t_u, texel, i)});
      v = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
    }
    t.m.Store(t.last_texel_var, v);
    t.last_texel = t.last_texel_var;  // marks "a sample happened"
  }

  // DELTA_GPU_DEBUGUV: output the sample UV as R/G instead of the texel, to see
  // the coordinate distribution reaching the sampler (normalized 0..1 vs texel
  // units). Diagnostic only.
  if (kDebugUv != 0.f && !dref && !gather && !int_img)
    texel = t.m.CompositeConstruct(
        t.t_v4, {t.FMul(x, t.F32(kDebugUv)), t.FMul(y, t.F32(kDebugUv)),
                 t.F32(0.f), t.F32(1.f)});

  // Gather first: a COMPARED gather still returns four values, so testing dref
  // ahead of it would store the vec4 into a single VGPR as if it were scalar.
  if (gather) {
    for (int i = 0; i < 4; i++)
      if (int_img)
        t.SetVg(vdata + i, t.m.CompositeExtract(t.t_u, texel, i));
      else
        t.SetVgF(vdata + i, t.m.CompositeExtract(t.t_f, texel, i));
    return;
  }
  if (dref) {
    if (dmask)
      t.SetVgF(vdata, texel);
    return;
  }
  u32 out = 0;
  for (int i = 0; i < 4; i++)
    if (dmask & (1 << i)) {
      if (int_img)
        t.SetVg(vdata + out++, t.m.CompositeExtract(t.t_u, texel, i));
      else
        t.SetVgF(vdata + out++, t.m.CompositeExtract(t.t_f, texel, i));
    }
}

// ---- graphics: raw MUBUF/MTBUF -> storage buffer ----------------------------
// A buffer op the vertex-input state does not already cover is a hand-written
// buffer read: the shader computes a per-lane index (an s_load'd vertex id, a
// bone index, an instance number) and pulls dwords out of a V#-described
// resource. Model it exactly as the compute path models guest memory -- a
// storage buffer aliasing [V#.base, ...) addressed by dword index -- rather
// than as a constant window, because the address is not uniform.
namespace {

// Dwords a raw buffer load moves, or 0 for an op this path does not implement
// (every store, atomic and format store).
u32 GfxMubufLoadDwords(u32 op) {
  if (op <= 0x03)
    return op + 1;  // buffer_load_format_x..xyzw (raw dwords, no conversion)
  if (op >= 0x08 && op <= 0x0b)
    return 1;  // buffer_load_ubyte/sbyte/ushort/sshort
  if (op == 0x0c)
    return 1;
  if (op == 0x0d)
    return 2;
  if (op == 0x0e)
    return 4;
  if (op == 0x0f)
    return 3;
  return 0;
}

// Same for a typed load. Takes Inst::opcode, which carries the Neo d16 bit at
// bit 3 (see Decode), so anything above tbuffer_load_format_xyzw is excluded.
u32 GfxMtbufLoadDwords(u32 opcode) {
  return opcode <= 0x03 ? opcode + 1 : 0;
}

}  // namespace

void PlanGfxBuffers(const Program& program,
                    u32 first_binding,
                    const std::unordered_set<u32>* claimed,
                    std::vector<ShaderBuffer>& buffers,
                    std::unordered_map<u32, u32>& bindings,
                    const u8* reachable) {
  // Which scalar load last wrote each SGPR range, so a descriptor quad that is
  // reloaded with a second V# gets a second binding instead of silently
  // inheriting the first one's buffer (same reasoning as PlanCsResources).
  struct ScalarLoad {
    u32 sgpr;
    u32 dwords;
    u32 version;
  };
  std::vector<ScalarLoad> loads;
  const auto descriptor_version = [&](u32 sgpr) {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + 4 <= it->sgpr + it->dwords)
        return it->version;
    return UINT32_MAX;  // inline user data
  };

  std::unordered_map<u64, u32> binding_by_descriptor;
  u32 index = 0;
  for (const Inst& inst : program) {
    const u32 inst_idx = index++;
    if (reachable && !reachable[inst_idx])
      continue;
    const u32 w = inst.raw[0];
    if (inst.enc == Enc::kSmrd) {
      const u32 n = SmrdDwordCount(inst.opcode);
      if (!n)
        continue;
      const u32 sdst = (w >> 15) & 0x7F;
      loads.erase(std::remove_if(loads.begin(), loads.end(),
                                 [&](const ScalarLoad& ld) {
                                   return sdst < ld.sgpr + ld.dwords &&
                                          ld.sgpr < sdst + n;
                                 }),
                  loads.end());
      loads.push_back({sdst, n, inst_idx});
      continue;
    }
    if (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf)
      continue;
    const bool load = inst.enc == Enc::kMubuf
                          ? GfxMubufLoadDwords((w >> 18) & 0x7F) != 0
                          : GfxMtbufLoadDwords(inst.opcode) != 0;
    if (!load)
      continue;  // store/atomic: not modelled, and must not spend a binding
    if (claimed && claimed->count(inst.pc))
      continue;  // already seeded as a vertex input (see direct_vfetch)
    const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    const u64 key =
        static_cast<u64>(srsrc) |
        (static_cast<u64>(descriptor_version(srsrc)) << 8);
    const auto found = binding_by_descriptor.find(key);
    if (found != binding_by_descriptor.end()) {
      bindings[inst.pc] = found->second;
      continue;
    }
    const u32 binding =
        first_binding + static_cast<u32>(buffers.size());
    if (binding >= MaxGfxBuffers()) {
      WarnUnsupported("mubuf.binding-count", binding + 1);
      continue;  // over the cap: the emitter warns and leaves the VGPRs zero
    }
    binding_by_descriptor[key] = binding;
    bindings[inst.pc] = binding;
    buffers.push_back({binding, srsrc, inst.pc});
  }
}

void EmitGfxMubuf(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 op = (w >> 18) & 0x7F, inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const u32 vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const u32 n = GfxMubufLoadDwords(op);
  const auto it = sc.gfx_buf_bind.find(inst.pc);
  if (!n || it == sc.gfx_buf_bind.end()) {
    WarnUnsupported(sc.is_ps ? "mubuf.ps" : "mubuf.vs", op, w, w1);
    return;
  }
  const Id var = t.EnsureGfxBuffer(it->second);
  // Clamp into the staged window. The index is whatever the shader computed,
  // and the renderer stages only a fixed prefix of the resource, so an
  // unclamped access could read outside the bound range. Hardware returns zero
  // past NUM_RECORDS; pinning to the window's last dword keeps this in bounds
  // without a per-load branch, and the stride/record fields needed for exact
  // bounds are not reliably live in a graphics stage's SGPRs.
  const Id byte_off = t.UMin(BufferByteOffset(t, inst, inst_offset, idxen,
                                              offen, vaddr, srsrc, soffset),
                             t.U32(kGfxBufferDwords * 4 - 4));
  const Id dword_idx = t.Shr(byte_off, t.U32(2));
  if (op >= 0x08 && op <= 0x0b) {
    const bool sign_extend = op == 0x09 || op == 0x0b;
    t.SetVg(vdata,
            LoadSubDword(t, var, byte_off, op <= 0x09 ? 8 : 16, sign_extend));
    return;
  }
  for (u32 i = 0; i < n; i++)
    t.SetVg(vdata + i, SsboLoad(t, var,
                                t.UMin(t.Add(dword_idx, t.U32(i)),
                                       t.U32(kGfxBufferDwords - 1))));
}

void EmitGfxMtbuf(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const u32 vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const u32 n = GfxMtbufLoadDwords(inst.opcode);
  if (!n) {
    // A store has nowhere to land: the set-2 window is a per-draw copy of guest
    // memory that is never read back, and draws sharing a resource share one
    // window, so writing into it would corrupt what the others read. Drop it
    // rather than reject the shader, which would drop every draw using it.
    AuditNote("mtbuf.store.gfx", inst.opcode);
    return;
  }
  const auto it = sc.gfx_buf_bind.find(inst.pc);
  if (it == sc.gfx_buf_bind.end()) {
    WarnUnsupported(sc.is_ps ? "mtbuf.ps" : "mtbuf.vs", inst.opcode, w, w1);
    return;
  }
  if (!MtbufIsRawDwords(inst, n))
    WarnUnsupported("mtbuf.raw-format", inst.opcode, w, w1);
  const Id var = t.EnsureGfxBuffer(it->second);
  const Id byte_off = t.UMin(BufferByteOffset(t, inst, inst_offset, idxen,
                                              offen, vaddr, srsrc, soffset),
                             t.U32(kGfxBufferDwords * 4 - 4));
  const Id dword_idx = t.Shr(byte_off, t.U32(2));
  for (u32 i = 0; i < n; i++)
    t.SetVg(vdata + i, SsboLoad(t, var,
                                t.UMin(t.Add(dword_idx, t.U32(i)),
                                       t.U32(kGfxBufferDwords - 1))));
}

// ---- compute: resource plan -------------------------------------------------
bool PlanCsResources(const Program& program,
                     const u8* reachable,
                     u32 lds_dwords,
                     RecompiledCs& r,
                     std::unordered_map<u32, u32>& bind) {
  struct ScalarLoad {
    u32 sgpr;
    u32 dwords;
    u32 version;
  };
  std::vector<ScalarLoad> loads;
  const auto descriptor_version = [&](u32 sgpr, u32 dwords) {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->version;
    return UINT32_MAX;  // inline user data
  };
  std::unordered_map<u64, u32> resource_by_version;
  const auto resource = [&](u32 pc, u32 base_sgpr, u32 dwords,
                            u8 kind, bool written, u32 min_bytes,
                            bool read = true) {
    const u64 key =
        static_cast<u64>(kind) | (static_cast<u64>(base_sgpr) << 8) |
        (static_cast<u64>(descriptor_version(base_sgpr, dwords)) << 16);
    const auto it = resource_by_version.find(key);
    if (it != resource_by_version.end()) {
      CsResource& res = r.resources[it->second];
      res.written = res.written || written;
      res.read = res.read || read;
      if (min_bytes > res.min_bytes)
        res.min_bytes = min_bytes;
      bind[pc] = it->second;
      return true;
    }
    const u32 idx = static_cast<u32>(r.resources.size());
    if (idx >= kMaxCsResources) {
      WarnUnsupported("cs.resource-count", idx + 1);
      return false;
    }
    resource_by_version[key] = idx;
    bind[pc] = idx;
    r.resources.push_back({base_sgpr, pc, idx, kind, written, read, min_bytes});
    return true;
  };

  u32 idx = 0;
  for (const Inst& inst : program) {
    const u32 inst_idx = idx++;
    if (reachable && !reachable[inst_idx])
      continue;  // dead block/padding
    const u32 w = inst.raw[0], w1 = inst.raw[1];
    switch (inst.enc) {
      case Enc::kSmrd: {
        const u32 op = inst.opcode, sbase = (w >> 9) & 0x3F;
        if (op > 0x0c)
          return false;  // beyond s_buffer_load_dwordx16
        const u32 n = op < 0x08 ? (1u << op) : SmrdLoadCount(op);
        if (!n)
          return false;  // reserved scalar-load opcode
        const SmrdOffset so = DecodeSmrdOffset(inst);
        const u32 bytes = so.in_sgpr ? 0 : (so.dwords + n) * 4;
        const u32 base_sgpr = sbase * 2;
        const u8 kind = op < 0x08 ? 2 : 0;
        if (!resource(inst.pc, base_sgpr, kind == 2 ? 2 : 4, kind, false, bytes,
                      /*read=*/true))
          return false;
        const u32 sdst = (w >> 15) & 0x7F;
        loads.erase(std::remove_if(loads.begin(), loads.end(),
                                   [&](const auto& ld) {
                                     return sdst < ld.sgpr + ld.dwords &&
                                            ld.sgpr < sdst + n;
                                   }),
                    loads.end());
        loads.push_back({sdst, n, inst_idx});
        break;
      }
      case Enc::kMubuf: {
        const u32 op = (w >> 18) & 0x7F;
        const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
        const bool load = op <= 0x03 || (op >= 0x08 && op <= 0x0f);
        const bool store = (op >= 0x04 && op <= 0x07) || op == 0x18 ||
                           op == 0x1a || (op >= 0x1c && op <= 0x1f);
        // An atomic both reads and writes its buffer, so it is a resource like
        // any other; leaving it unplanned declined the whole dispatch.
        const bool atomic = MubufAtomic(op);
        if (!load && !store && !atomic)
          return false;
        if (!resource(inst.pc, srsrc, 4, 0, store || atomic, 0,
                      /*read=*/load || atomic))
          return false;
        break;
      }
      case Enc::kMtbuf: {
        const u32 op = (w >> 16) & 0x7;
        const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
        if (!resource(inst.pc, srsrc, 4, 0, op >= 4, 0, /*read=*/op < 4))
          return false;
        break;
      }
      case Enc::kMimg: {
        const u32 op = (w >> 18) & 0x7F;
        const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
        const bool r128 = (w & 0x8000) != 0;
        if (op == 0x0e)
          break;  // get_resinfo reads only descriptor SGPRs
        const bool store = op == 0x08 || op == 0x09;
        const bool load = op == 0x00 || op == 0x01;
        const bool sample = op >= 0x20 && MimgNamesItsLod(op);
        if ((!store && !load && !sample) || r128 || srsrc + 7 >= 136)
          return false;
        if (!resource(inst.pc, srsrc, 8, 1, store, 0,
                      /*read=*/load || sample))
          return false;
        break;
      }
      case Enc::kDs:
        // DS swizzle uses the wave cross-lane path, not LDS memory.
        if (!lds_dwords && inst.opcode != 0x35)
          return false;
        if ((w >> 17) & 1)
          return false;  // GDS not modelled
        break;
      default:
        break;
    }
  }
  return true;
}

// ---- compute: SMRD ----------------------------------------------------------
// s_load / s_buffer_load: read scalar constants from the bound buffer.
void EmitCsSmrd(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], op = inst.opcode;
  const u32 sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
  const u32 base_sgpr = sbase * 2;
  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const u32 n = op < 0x08 ? (1u << op) : SmrdLoadCount(op);
  const SmrdOffset so = DecodeSmrdOffset(inst);
  const Id dword_off =
      so.in_sgpr ? t.Shr(t.SrcRaw(so.sgpr, inst.literal), t.U32(2))
                 : t.U32(so.dwords);
  for (u32 i = 0; i < n; i++)
    t.SetSg(sdst + i, CsSsboLoad(t, sc, static_cast<u32>(b),
                                 t.Add(dword_off, t.U32(i))));
}

// ---- compute: MUBUF ---------------------------------------------------------
void EmitCsMubuf(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 op = (w >> 18) & 0x7F, inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const u32 vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const u32 binding = static_cast<u32>(b);
  const Id byte_off = BufferByteOffset(t, inst, inst_offset, idxen, offen,
                                       vaddr, srsrc, soffset);
  const Id dword_idx = t.Shr(byte_off, t.U32(2));

  // Format load/stores move raw dwords: the buffer-format -> image-store round
  // trip the copy shaders perform is identity, so no conversion is applied.
  const auto load_dwords = [&](u32 n) {
    for (u32 i = 0; i < n; i++)
      t.SetVg(vdata + i,
              CsSsboLoad(t, sc, binding, t.Add(dword_idx, t.U32(i))));
  };
  const auto store_dwords = [&](u32 n) {
    for (u32 i = 0; i < n; i++)
      CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(i)), t.Vg(vdata + i));
  };

  if (MubufAtomic(op)) {
    // GLC returns the pre-op value into VDATA; without it the result is
    // simply unused. Device scope: these order against other workgroups.
    const bool glc = (w >> 14) & 1;
    const Id ptr = CsSsboPtr(t, sc, binding, dword_idx);
    const Id scope = t.U32(static_cast<u32>(spv::Scope::Device));
    const Id relaxed = t.U32(0);
    const Id src = t.Vg(vdata);
    Id old_value = 0;
    const auto rmw = [&](spv::Op a) {
      old_value = t.m.Emit(a, t.t_u, {ptr, scope, relaxed, src});
    };
    switch (op) {
      case 0x30: rmw(spv::Op::OpAtomicExchange); break;
      case 0x31: {  // cmpswap: vdata = new value, vdata+1 = comparand
        old_value = t.m.Emit(spv::Op::OpAtomicCompareExchange, t.t_u,
                             {ptr, scope, relaxed, relaxed, src,
                              t.Vg(vdata + 1)});
        break;
      }
      case 0x32: rmw(spv::Op::OpAtomicIAdd); break;
      case 0x33: rmw(spv::Op::OpAtomicISub); break;
      case 0x35: rmw(spv::Op::OpAtomicSMin); break;
      case 0x36: rmw(spv::Op::OpAtomicUMin); break;
      case 0x37: rmw(spv::Op::OpAtomicSMax); break;
      case 0x38: rmw(spv::Op::OpAtomicUMax); break;
      case 0x39: rmw(spv::Op::OpAtomicAnd); break;
      case 0x3a: rmw(spv::Op::OpAtomicOr); break;
      case 0x3b: rmw(spv::Op::OpAtomicXor); break;
      default:
        // The wrapping inc/dec, the float forms and the 64-bit _x2 pairs have
        // no single SPIR-V op over a uint buffer. Named rather than faked.
        WarnUnsupported("mubuf.atomic", op, w, w1);
        sc.cs_unsupported = true;
        return;
    }
    if (glc && old_value)
      t.SetVg(vdata, old_value);
    return;
  }

  switch (op) {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03:  // buffer_load_format_x..xyzw
      load_dwords(op + 1);
      break;
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07:  // buffer_store_format_x..xyzw
      store_dwords(op - 3);
      break;
    case 0x08:  // buffer_load_ubyte
      t.SetVg(vdata, LoadSubDword(t, sc.cs_ssbo[binding], byte_off, 8, false));
      break;
    case 0x09:  // buffer_load_sbyte
      t.SetVg(vdata, LoadSubDword(t, sc.cs_ssbo[binding], byte_off, 8, true));
      break;
    case 0x0a:  // buffer_load_ushort
      t.SetVg(vdata, LoadSubDword(t, sc.cs_ssbo[binding], byte_off, 16, false));
      break;
    case 0x0b:  // buffer_load_sshort
      t.SetVg(vdata, LoadSubDword(t, sc.cs_ssbo[binding], byte_off, 16, true));
      break;
    case 0x0c:
      load_dwords(1);
      break;  // buffer_load_dword
    case 0x0d:
      load_dwords(2);
      break;  // buffer_load_dwordx2
    case 0x0e:
      load_dwords(4);
      break;  // buffer_load_dwordx4
    case 0x0f:
      load_dwords(3);
      break;    // buffer_load_dwordx3
    case 0x18:  // buffer_store_byte
      StoreSubDword(t, sc, binding, byte_off, 8, t.Vg(vdata));
      break;
    case 0x1a:  // buffer_store_short
      StoreSubDword(t, sc, binding, byte_off, 16, t.Vg(vdata));
      break;
    case 0x1c:
      store_dwords(1);
      break;  // buffer_store_dword
    case 0x1d:
      store_dwords(2);
      break;  // buffer_store_dwordx2
    case 0x1e:
      store_dwords(4);
      break;  // buffer_store_dwordx4
    case 0x1f:
      store_dwords(3);
      break;  // buffer_store_dwordx3
    default:
      WarnUnsupported("mubuf", op, w, w1);
      sc.cs_unsupported = true;
      break;
  }
}

// ---- compute: MTBUF ---------------------------------------------------------
void EmitCsMtbuf(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 op = (w >> 16) & 0x7, inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const u32 vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const u32 n = (op & 3) + 1;
  if (!MtbufIsRawDwords(inst, n))
    WarnUnsupported("mtbuf.raw-format", op, w, w1);
  const u32 binding = static_cast<u32>(b);
  const Id byte_off = BufferByteOffset(t, inst, inst_offset, idxen, offen,
                                       vaddr, srsrc, soffset);
  const Id dword_idx = t.Shr(byte_off, t.U32(2));
  if (op < 4) {  // tbuffer_load_format_x..xyzw
    for (u32 i = 0; i < n; i++)
      t.SetVg(vdata + i,
              CsSsboLoad(t, sc, binding, t.Add(dword_idx, t.U32(i))));
  } else {  // tbuffer_store_format_x..xyzw
    for (u32 i = 0; i < n; i++)
      CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(i)), t.Vg(vdata + i));
  }
}

// ---- compute: MIMG ----------------------------------------------------------
// image_load/store[_mip] against staged linear RGBA8 images modelled as
// storage buffers. Storage is mip-major; each level contains all physical
// array layers and explicit LOD is view-relative.
void EmitCsMimg(Translator& t,
                const Inst& inst,
                StageContext& sc,
                const Id* address) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 op = (w >> 18) & 0x7F, dmask = (w >> 8) & 0xF;
  const u32 vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
  const bool mip_op = op == 0x01 || op == 0x09;
  const bool store = op == 0x08 || op == 0x09;
  const bool load = op == 0x00 || op == 0x01;
  // image_sample_lz_o: sample_lz plus a constant texel offset. The ISA orders
  // vaddr as "offsets, bias, zpcf, then coordinates", and _lz_o carries neither
  // a bias nor a z-compare, so word 0 is the packed offset and the coordinates
  // start one word later -- which is what addr_shift below is for.
  const bool offset = op == 0x37;
  const bool sample = op == 0x24 || op == 0x27 || offset;
  // image_gather4_lz: the same 2x2 footprint the bilinear path already fetches,
  // returned as four texels instead of blended. Only the plain form -- the
  // z-compare (bit 3) and offset (bit 4) variants carry extra address words
  // this emitter does not read, so they would take the coordinates one word
  // early. GTA:SA's depth pyramid, the HZB every occlusion test reads, is this
  // instruction.
  const bool gather = op == 0x47;
  const bool resinfo = op == 0x0e;
  if (!store && !load && !sample && !gather && !resinfo) {
    // Silently setting the flag made the whole dispatch vanish with an empty
    // op list in the audit -- the one report that was supposed to say why.
    WarnUnsupported("mimg.cs", op, w, w1);
    sc.cs_unsupported = true;
    return;
  }
  const bool da = (w & 0x4000) != 0;
  // gfx10 NSA names every address component in its own VGPR; the caller hands
  // them over already loaded. Without this a 2D load reads y from vaddr+1,
  // which on an NSA instruction is some unrelated register -- the source image
  // is then addressed by x alone and the result is vertical stripes.
  const u32 addr_shift = offset ? 1u : 0u;
  const auto addr_vg = [&](u32 i) {
    return address ? address[i + addr_shift] : t.Vg(vaddr + i + addr_shift);
  };
  const auto addr_vgf = [&](u32 i) {
    return t.m.Bitcast(t.t_f, addr_vg(i));
  };

  // T# field extraction (see DecodeTImage for the dword layout).
  const auto field = [&](u32 dword, u32 shift, u32 mask) {
    return t.And(t.Shr(t.Sg(srsrc + dword), t.U32(shift)), t.U32(mask));
  };
  // gfx10.3 splits WIDTH's low two bits into dword 1 and drops the pitch and
  // pow2-pad fields; everything else sits where GFX7 puts it.
  const Id base_width =
      t.rdna_sources
          ? t.Add(t.Or(field(1, 30, 0x3), t.Shl(field(2, 0, 0xFFF), t.U32(2))),
                  t.U32(1))
          : t.Add(field(2, 0, 0x3FFF), t.U32(1));
  const Id base_height = t.Add(field(2, 14, 0x3FFF), t.U32(1));
  const Id base_mip = field(3, 12, 0xF);
  const Id last_mip = field(3, 16, 0xF);
  const Id safe_last_mip = t.UMax(base_mip, last_mip);
  const auto logical_or = [&](Id a, Id b) {
    return t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {a, b});
  };
  const Id image_type = field(3, 28, 0xF);
  // A 1D image (type 8/12) has a one-texel-high layout and drops the y
  // address component: the layer of a 1D array rides where 2D's y sits.
  const Id is_1d_img =
      logical_or(t.Eq(image_type, t.U32(8)), t.Eq(image_type, t.U32(12)));
  // A cube (type 11) addresses like a 2D array whose layer is the face: the
  // MIMG address already carries the face index, so the layout and the
  // addressing below are the array ones.
  const Id is_array = logical_or(
      logical_or(t.Eq(image_type, t.U32(13)), t.Eq(image_type, t.U32(12))),
      t.Eq(image_type, t.U32(11)));
  // A volume (type 10) stages slice by slice exactly as an array does, and its
  // slice count sits in the same descriptor field as an array's layer count --
  // so everything below can address it as an array. The one difference is
  // where the slice index comes from: a 3D MIMG leaves DA clear and carries z
  // as its third address component, so it must be read whether or not DA is
  // set. Without this the type fell outside supported_type, every store was
  // predicated off, and P.T.'s colour-grading LUT -- its only volume upload --
  // stayed zero, which graded the finished frame to black.
  const Id is_3d_img = t.Eq(image_type, t.U32(10));
  const Id has_slices = logical_or(is_array, is_3d_img);
  const Id descriptor_layers = t.Add(field(4, 0, 0x1FFF), t.U32(1));

  if (resinfo) {  // dimensions from the descriptor, no memory access
    const Id mip = t.UMin(addr_vg(0), t.Sub(safe_last_mip, base_mip));
    const Id physical = t.Add(base_mip, mip);
    const Id comps[4] = {
        Max1(t, t.Shr(base_width, physical)),
        Max1(t, t.Shr(base_height, physical)),
        t.SelectB(has_slices, descriptor_layers, t.U32(1)),
        t.Add(t.Sub(safe_last_mip, base_mip), t.U32(1)),
    };
    u32 out = 0;
    for (int i = 0; i < 4; i++)
      if (dmask & (1 << i))
        t.SetVg(vdata + out++, comps[i]);
    return;
  }

  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const u32 binding = static_cast<u32>(b);

  // A tiled gfx10.3 surface carries no pitch: the CPU-side layout uses the
  // width, so the shader must index the staging image the same way.
  const Id base_pitch =
      t.rdna_sources ? base_width : t.Add(field(4, 13, 0x3FFF), t.U32(1));
  // gfx10.3 replaces (dfmt, nfmt) with one 9-bit format enum.
  const Id gfmt = field(1, 20, 0x1FF);
  const Id dfmt = field(1, 20, 0x3F), nfmt = field(1, 26, 0xF);
  const Id is_unorm = t.IsZero(nfmt);  // nfmt 0 = UNORM
  const auto is_gfmt = [&](std::initializer_list<u32> values) {
    Id any = t.m.ConstBool(false);
    for (u32 v : values)
      any = logical_or(any, t.Eq(gfmt, t.U32(v)));
    return any;
  };
  const Id is_rgba8 =
      t.rdna_sources
          ? is_gfmt({56, 60, 130})
          : t.LAnd(t.Eq(dfmt, t.U32(10)),
                   logical_or(is_unorm, t.Eq(nfmt, t.U32(4))));
  const Id is_r32 =
      t.rdna_sources
          ? is_gfmt({20, 21, 22})
          : t.LAnd(t.Eq(dfmt, t.U32(4)),
                   logical_or(t.Eq(nfmt, t.U32(4)),
                              logical_or(t.Eq(nfmt, t.U32(5)),
                                         t.Eq(nfmt, t.U32(7)))));
  const Id is_rg16f = t.rdna_sources
                          ? is_gfmt({29})
                          : t.LAnd(t.Eq(dfmt, t.U32(5)), t.Eq(nfmt, t.U32(7)));
  const Id is_r16f = t.rdna_sources
                         ? is_gfmt({13})
                         : t.LAnd(t.Eq(dfmt, t.U32(2)), t.Eq(nfmt, t.U32(7)));
  const Id is_rg8 =
      t.rdna_sources ? is_gfmt({14}) : t.LAnd(t.Eq(dfmt, t.U32(3)), is_unorm);
  const Id is_rgba16f = t.rdna_sources ? is_gfmt({71})
                                       : t.LAnd(t.Eq(dfmt, t.U32(12)),
                                                t.Eq(nfmt, t.U32(7)));
  const Id is_r11g11b10f = t.rdna_sources
                               ? is_gfmt({36})
                               : t.LAnd(t.Eq(dfmt, t.U32(6)),
                                        t.Eq(nfmt, t.U32(7)));
  // A block-compressed surface a shader writes is described as an
  // uncompressed integer image whose texel is one BC block: 64 bpp (32_32 or
  // 16_16_16_16) for BC1/BC4, 128 bpp (32_32_32_32) for BC2/BC3/BC5. The
  // components are raw bits -- no normalisation, no float conversion -- so
  // they pass through the staging buffer unchanged. P.T.'s texture streamer
  // uploads every streamed surface this way; without these the access was
  // gated off and the copy stored nothing.
  const Id is_int = logical_or(t.Eq(nfmt, t.U32(4)), t.Eq(nfmt, t.U32(5)));
  const Id false_id = t.m.ConstBool(false);
  const Id is_rgba16u = t.rdna_sources
                            ? false_id
                            : t.LAnd(t.Eq(dfmt, t.U32(12)), is_int);
  const Id is_rg32u =
      t.rdna_sources ? false_id : t.LAnd(t.Eq(dfmt, t.U32(11)), is_int);
  const Id is_rgba32u =
      t.rdna_sources ? false_id : t.LAnd(t.Eq(dfmt, t.U32(14)), is_int);
  // Two dwords per texel (16_16_16_16 packs two components each, 32_32 is one
  // per dword), four for 32_32_32_32.
  const Id is_block2 = logical_or(is_rgba16u, is_rg32u);
  const Id is_block_int = logical_or(is_block2, is_rgba32u);
  Id supported_format = logical_or(is_rgba8, is_r32);
  supported_format = logical_or(supported_format, is_rg16f);
  supported_format = logical_or(supported_format, is_r16f);
  supported_format = logical_or(supported_format, is_rg8);
  supported_format = logical_or(supported_format, is_rgba16f);
  supported_format = logical_or(supported_format, is_r11g11b10f);
  supported_format = logical_or(supported_format, is_block_int);
  const Id supported_type = logical_or(
      logical_or(t.Eq(image_type, t.U32(9)), t.Eq(image_type, t.U32(8))),
      has_slices);
  Id requested_mip =
      mip_op ? t.SelectB(is_1d_img, addr_vg(da ? 2 : 1), addr_vg(da ? 3 : 2))
             : t.U32(0);
  if (op == 0x24) {
    // SelectF, not SelectB: the operands are the float LOD dwords, and a
    // uint-typed OpSelect over float operands fails spirv-val ("Expected both
    // objects to be of Result Type"), which DECLINED every CS containing
    // image_sample_l. SotC issued three such dispatches, skipped on every
    // frame since 4087a1a introduced the 1D select.
    const Id lod_addr = t.SelectF(is_1d_img, addr_vgf(da ? 2 : 1),
                                  addr_vgf(da ? 3 : 2));
    const Id lod = t.m.ExtInst(t.t_f, GLSLstd450FMax, {lod_addr, t.F32(0.f)});
    requested_mip = t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {lod});
  }
  const Id view_mip = t.UMin(requested_mip, t.Sub(safe_last_mip, base_mip));
  const Id physical_mip = t.Add(base_mip, view_mip);
  const Id width = Max1(t, t.Shr(base_width, physical_mip));
  const Id height = Max1(t, t.Shr(base_height, physical_mip));
  Id x = addr_vg(0);
  Id y = t.SelectB(is_1d_img, t.U32(0), addr_vg(1));
  Id sample_fx = t.F32(0.f), sample_fy = t.F32(0.f);
  if (sample || gather) {
    // Bilinear filter of the linear staging image with clamp addressing and
    // texel-centre coordinates. image_sample_l (0x24) takes an explicit LOD in
    // the address; _lz (0x27) forces LOD 0 -- both already folded into the mip
    // maths above. x/y hold the lower texel; the upper corner and fractional
    // weights are formed in the access block.
    const Id width_f = t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {width});
    const Id height_f = t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {height});
    sample_fx = t.FSub(t.FMul(t.FClamp01(addr_vgf(0)), width_f), t.F32(0.5f));
    sample_fy =
        t.FSub(t.FMul(t.FClamp01(addr_vgf(1)), height_f), t.F32(0.5f));
    const Id fx0 =
        t.Ext1(GLSLstd450Floor, t.Ext2(GLSLstd450FMax, sample_fx, t.F32(0.f)));
    const Id fy0 =
        t.Ext1(GLSLstd450Floor, t.Ext2(GLSLstd450FMax, sample_fy, t.F32(0.f)));
    x = t.UMin(t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {fx0}),
               t.Sub(width, t.U32(1)));
    y = t.UMin(t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {fy0}),
               t.Sub(height, t.U32(1)));
    if (offset) {
      // Six signed bits per component, x at [5:0] and y at [13:8]. The add
      // wraps when the offset is negative and larger than the coordinate, so
      // clamp that back to zero rather than to the far edge.
      const Id packed = address ? address[0] : t.Vg(vaddr);
      const auto shifted = [&](Id c, u32 bit, Id limit) {
        const Id raw = t.And(t.Shr(packed, t.U32(bit)), t.U32(0x3F));
        const Id signed_off = t.Sub(t.Xor(raw, t.U32(0x20)), t.U32(0x20));
        const Id sum = t.Add(c, signed_off);
        return t.SelectB(t.Uge(sum, t.U32(0x80000000u)), t.U32(0),
                         t.UMin(sum, t.Sub(limit, t.U32(1))));
      };
      x = shifted(x, 0, width);
      y = shifted(y, 8, height);
    }
    y = t.SelectB(is_1d_img, t.U32(0), y);
  }
  // gfx10.3 swizzle mode 0 is the linear one, and it has no pow2-pad bit.
  const Id linear_general = t.Eq(field(3, 20, 0x1F), t.U32(t.rdna_sources ? 0 : 31));
  const Id pow2_pad =
      t.rdna_sources ? t.m.ConstBool(false) : t.IsNonZero(field(3, 25, 1));
  const Id stored_height = t.SelectB(pow2_pad, BitCeil(t, height), height);
  const Id pitch = LinearMipPitch(t, base_pitch, stored_height, physical_mip,
                                  linear_general, pow2_pad);
  // A volume has no base-slice/last-slice pair: word 5 belongs to the array
  // view, so it addresses from slice 0 through depth - 1.
  const Id raw_base_array =
      t.rdna_sources ? field(4, 16, 0x1FFF) : field(5, 0, 0x1FFF);
  const Id raw_last_array = t.rdna_sources
                                ? t.Add(descriptor_layers, t.U32(~0u))
                                : field(5, 13, 0x1FFF);
  const Id base_array = t.SelectB(is_3d_img, t.U32(0), raw_base_array);
  const Id last_array = t.SelectB(
      is_3d_img, t.Add(descriptor_layers, t.U32(~0u)), raw_last_array);
  const Id addr_slice = t.SelectB(is_1d_img, addr_vg(1), addr_vg(2));
  const Id view_layer =
      da ? addr_slice : t.SelectB(is_3d_img, addr_slice, t.U32(0));
  const Id physical_layer = t.Add(base_array, view_layer);
  const Id padded_layers =
      t.SelectB(pow2_pad, BitCeil(t, descriptor_layers), descriptor_layers);
  const Id layers = t.SelectB(has_slices, padded_layers, t.U32(1));
  Id array_ok =
      t.LAnd(t.Ult(base_array, layers), t.Uge(last_array, base_array));
  array_ok = t.LAnd(array_ok, t.Ule(view_layer, t.Sub(last_array, base_array)));
  array_ok = t.LAnd(array_ok, t.Ult(physical_layer, layers));
  const Id layer_ok = t.m.Emit(spv::Op::OpSelect, t.t_bool,
                               {has_slices, array_ok, t.m.ConstBool(true)});
  Id valid = t.LAnd(t.Ult(x, width), t.Ult(y, height));
  valid = t.LAnd(valid, layer_ok);
  valid = t.LAnd(valid, t.Uge(last_mip, base_mip));
  valid = t.LAnd(valid, supported_type);
  valid = t.LAnd(valid, supported_format);
  if (load || sample || gather) {  // zero the destination for an invalid access
    if (gather) {
      // DMASK picks the component to gather, not how many registers come back:
      // a gather4 always returns one texel per destination register.
      for (u32 i = 0; i < 4; i++)
        t.SetVg(vdata + i, t.U32(0));
    } else {
      u32 out = 0;
      for (int i = 0; i < 4; i++)
        if (dmask & (1 << i))
          t.SetVg(vdata + out++, t.U32(0));
    }
  }

  const Id access_blk = t.m.NewBlock(), merge_blk = t.m.NewBlock();
  t.m.SelectionMerge(merge_blk);
  t.m.BranchConditional(valid, access_blk, merge_blk);
  t.m.OpenBlock(access_blk);

  const Id layer = t.SelectB(has_slices, physical_layer, t.U32(0));
  const Id mip_off = MipChainOffset(t, base_pitch, base_height, layers,
                                    physical_mip, linear_general, pow2_pad);
  const Id layer_off = t.Mul(layer, t.Mul(pitch, stored_height));
  const Id texel_idx =
      t.Add(mip_off, t.Add(layer_off, t.Add(t.Mul(y, pitch), x)));
  Id dword_idx = t.SelectB(is_rgba16f, t.Mul(texel_idx, t.U32(2)), texel_idx);
  dword_idx = t.SelectB(is_r11g11b10f, t.Mul(texel_idx, t.U32(4)), dword_idx);
  dword_idx = t.SelectB(is_block2, t.Mul(texel_idx, t.U32(2)), dword_idx);
  dword_idx = t.SelectB(is_rgba32u, t.Mul(texel_idx, t.U32(4)), dword_idx);
  const Id wide2 = logical_or(logical_or(is_rgba16f, is_r11g11b10f),
                              logical_or(is_block2, is_rgba32u));
  const Id wide4 = logical_or(is_r11g11b10f, is_rgba32u);

  if (load) {
    const Id raw = CsSsboLoad(t, sc, binding, dword_idx);
    const Id has_second = wide2;
    const Id raw_hi = CsSsboLoad(
        t, sc, binding,
        t.SelectB(has_second, t.Add(dword_idx, t.U32(1)), dword_idx));
    const Id raw_2 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(wide4, t.Add(dword_idx, t.U32(2)), dword_idx));
    const Id raw_3 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(wide4, t.Add(dword_idx, t.U32(3)), dword_idx));
    const Id halfs = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {raw});
    const Id halfs_hi = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {raw_hi});
    const Id float_component[4] = {raw, raw_hi, raw_2, raw_3};
    u32 out = 0;
    for (int i = 0; i < 4; i++) {
      if (!(dmask & (1 << i)))
        continue;
      const Id byte = t.And(t.Shr(raw, t.U32(i * 8u)), t.U32(0xFF));
      const Id normalized =
          t.FMul(t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {byte}),
                 t.F32(1.0f / 255.0f));
      Id value = t.SelectB(is_unorm, t.m.Bitcast(t.t_u, normalized), byte);
      const Id half =
          i < 2 ? t.m.Bitcast(t.t_u, t.m.CompositeExtract(t.t_f, halfs, i))
                : t.U32(0);
      value = t.SelectB(is_rg16f, half, value);
      value = t.SelectB(is_r16f, i == 0 ? half : t.U32(0), value);
      value = t.SelectB(is_rg8, i < 2 ? value : t.U32(0), value);
      const Id wide_half = t.m.Bitcast(
          t.t_u, t.m.CompositeExtract(t.t_f, i < 2 ? halfs : halfs_hi,
                                      i < 2 ? i : i - 2));
      value = t.SelectB(is_rgba16f, wide_half, value);
      value = t.SelectB(is_r11g11b10f, float_component[i], value);
      value = t.SelectB(is_r32, i == 0 ? raw : t.U32(0), value);
      // Raw block bits: 16_16_16_16 puts two components in each dword, the
      // 32-bit forms one per dword.
      value = t.SelectB(
          is_rgba16u,
          t.And(t.Shr(i < 2 ? raw : raw_hi, t.U32((i & 1) * 16u)),
                t.U32(0xFFFF)),
          value);
      value = t.SelectB(is_rg32u,
                        i == 0 ? raw : (i == 1 ? raw_hi : t.U32(0)), value);
      value = t.SelectB(is_rgba32u, float_component[i], value);
      t.SetVg(vdata + out++, value);
    }
  } else if (sample || gather) {
    const Id has_second =
        t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_rgba16f, is_r11g11b10f});
    // Storage-buffer index of texel (xx,yy) in the current mip/layer, scaled
    // for the multi-dword formats (matches the single-texel index above).
    const auto texel_at = [&](Id xx, Id yy) {
      Id ti = t.Add(mip_off, t.Add(layer_off, t.Add(t.Mul(yy, pitch), xx)));
      ti = t.SelectB(is_rgba16f, t.Mul(ti, t.U32(2)), ti);
      ti = t.SelectB(is_r11g11b10f, t.Mul(ti, t.U32(4)), ti);
      ti = t.SelectB(is_block2, t.Mul(ti, t.U32(2)), ti);
      ti = t.SelectB(is_rgba32u, t.Mul(ti, t.U32(4)), ti);
      return ti;
    };
    // Decode one texel to float RGBA, honouring the storage format. Sampling
    // always yields floats (the shader reads the filtered colour), unlike the
    // raw integer fetch of image_load above.
    const auto decode = [&](Id idx, Id out[4]) {
      const Id raw = CsSsboLoad(t, sc, binding, idx);
      const Id raw_hi = CsSsboLoad(
          t, sc, binding, t.SelectB(has_second, t.Add(idx, t.U32(1)), idx));
      const Id raw_2 = CsSsboLoad(
          t, sc, binding, t.SelectB(is_r11g11b10f, t.Add(idx, t.U32(2)), idx));
      const Id raw_3 = CsSsboLoad(
          t, sc, binding, t.SelectB(is_r11g11b10f, t.Add(idx, t.U32(3)), idx));
      const Id halfs = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {raw});
      const Id halfs_hi =
          t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {raw_hi});
      const Id fcomp[4] = {raw, raw_hi, raw_2, raw_3};
      for (int i = 0; i < 4; i++) {
        const Id byte = t.And(t.Shr(raw, t.U32(i * 8u)), t.U32(0xFF));
        const Id byte_f =
            t.FMul(t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {byte}),
                   t.F32(1.0f / 255.0f));
        Id v = byte_f;  // rgba8 unorm default
        v = t.SelectF(is_r32, i == 0 ? t.m.Bitcast(t.t_f, raw) : t.F32(0.f), v);
        const Id half =
            i < 2 ? t.m.CompositeExtract(t.t_f, halfs, i) : t.F32(0.f);
        v = t.SelectF(is_rg16f, half, v);
        v = t.SelectF(
            is_r16f,
            i == 0 ? t.m.CompositeExtract(t.t_f, halfs, 0) : t.F32(0.f), v);
        v = t.SelectF(is_rg8, i < 2 ? byte_f : t.F32(0.f), v);
        const Id wide = t.m.CompositeExtract(t.t_f, i < 2 ? halfs : halfs_hi,
                                             i < 2 ? i : i - 2);
        v = t.SelectF(is_rgba16f, wide, v);
        v = t.SelectF(is_r11g11b10f, t.m.Bitcast(t.t_f, fcomp[i]), v);
        out[i] = v;
      }
    };
    const Id x1 = t.UMin(t.Add(x, t.U32(1)), t.Sub(width, t.U32(1)));
    const Id y1 = t.UMin(t.Add(y, t.U32(1)), t.Sub(height, t.U32(1)));
    const Id wx = t.FClamp01(
        t.FSub(sample_fx, t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {x})));
    const Id wy = t.FClamp01(
        t.FSub(sample_fy, t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {y})));
    Id c00[4], c10[4], c01[4], c11[4];
    decode(texel_at(x, y), c00);
    decode(texel_at(x1, y), c10);
    decode(texel_at(x, y1), c01);
    decode(texel_at(x1, y1), c11);
    if (gather) {
      // The gathered component is the one DMASK names; the four destination
      // registers are the footprint counter-clockwise from the lower left,
      // (x,y+1) (x+1,y+1) (x+1,y) (x,y) -- the order both the GCN ISA and
      // OpImageGather use.
      u32 comp = 0;
      for (u32 i = 0; i < 4; i++)
        if (dmask & (1u << i)) {
          comp = i;
          break;
        }
      t.SetVgF(vdata + 0, c01[comp]);
      t.SetVgF(vdata + 1, c11[comp]);
      t.SetVgF(vdata + 2, c10[comp]);
      t.SetVgF(vdata + 3, c00[comp]);
    } else {
      const auto mix = [&](Id a, Id b, Id w) {
        return t.FAdd(a, t.FMul(t.FSub(b, a), w));
      };
      u32 out = 0;
      for (int i = 0; i < 4; i++) {
        if (!(dmask & (1 << i)))
          continue;
        const Id top = mix(c00[i], c10[i], wx);
        const Id bot = mix(c01[i], c11[i], wx);
        t.SetVgF(vdata + out++, mix(top, bot, wy));
      }
    }
  } else {
    const auto store_byte = [&](u32 reg) {
      const Id value = t.Vg(reg);
      Id normalized = t.FClamp01(t.m.Bitcast(t.t_f, value));
      normalized = t.m.ExtInst(t.t_f, GLSLstd450RoundEven,
                               {t.FMul(normalized, t.F32(255.0f))});
      const Id unorm = t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {normalized});
      return t.And(t.SelectB(is_unorm, unorm, value), t.U32(0xFF));
    };
    const Id old_raw = CsSsboLoad(t, sc, binding, dword_idx);
    const Id has_second = wide2;
    const Id old_raw_hi = CsSsboLoad(
        t, sc, binding,
        t.SelectB(has_second, t.Add(dword_idx, t.U32(1)), dword_idx));
    const Id old_raw_2 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(wide4, t.Add(dword_idx, t.U32(2)), dword_idx));
    const Id old_raw_3 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(wide4, t.Add(dword_idx, t.U32(3)), dword_idx));
    Id packed;
    if (dmask == 0xF) {
      packed = store_byte(vdata);
      packed = t.Or(packed, t.Shl(store_byte(vdata + 1), t.U32(8)));
      packed = t.Or(packed, t.Shl(store_byte(vdata + 2), t.U32(16)));
      packed = t.Or(packed, t.Shl(store_byte(vdata + 3), t.U32(24)));
    } else {
      packed = old_raw;  // read-modify-write
      u32 comp = 0;
      for (int i = 0; i < 4; i++) {
        if (!(dmask & (1 << i)))
          continue;
        const Id keep = t.And(packed, t.U32(~(0xFFu << (i * 8))));
        packed = t.Or(keep, t.Shl(store_byte(vdata + comp++), t.U32(i * 8u)));
      }
    }
    const Id old_halfs =
        t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {old_raw});
    Id half[2] = {t.m.CompositeExtract(t.t_f, old_halfs, 0),
                  t.m.CompositeExtract(t.t_f, old_halfs, 1)};
    u32 half_reg = 0;
    for (u32 i = 0; i < 2; i++)
      if (dmask & (1u << i))
        half[i] = t.m.Bitcast(t.t_f, t.Vg(vdata + half_reg++));
    const Id packed_rg16f =
        t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16,
                    {t.m.CompositeConstruct(t.t_v2, {half[0], half[1]})});
    Id r16f = t.m.CompositeExtract(t.t_f, old_halfs, 0);
    if (dmask & 1)
      r16f = t.m.Bitcast(t.t_f, t.Vg(vdata));
    const Id packed_r16f =
        t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16,
                    {t.m.CompositeConstruct(t.t_v2, {r16f, t.F32(0.f)})});
    const Id old_halfs_hi =
        t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {old_raw_hi});
    Id wide_half[4] = {
        t.m.CompositeExtract(t.t_f, old_halfs, 0),
        t.m.CompositeExtract(t.t_f, old_halfs, 1),
        t.m.CompositeExtract(t.t_f, old_halfs_hi, 0),
        t.m.CompositeExtract(t.t_f, old_halfs_hi, 1),
    };
    u32 wide_reg = 0;
    for (u32 i = 0; i < 4; i++)
      if (dmask & (1u << i))
        wide_half[i] = t.m.Bitcast(t.t_f, t.Vg(vdata + wide_reg++));
    const Id packed_rgba16f_lo = t.m.ExtInst(
        t.t_u, GLSLstd450PackHalf2x16,
        {t.m.CompositeConstruct(t.t_v2, {wide_half[0], wide_half[1]})});
    const Id packed_rgba16f_hi = t.m.ExtInst(
        t.t_u, GLSLstd450PackHalf2x16,
        {t.m.CompositeConstruct(t.t_v2, {wide_half[2], wide_half[3]})});
    Id packed_float[4] = {old_raw, old_raw_hi, old_raw_2, old_raw_3};
    u32 packed_float_reg = 0;
    for (u32 i = 0; i < 4; i++) {
      if (!(dmask & (1u << i)))
        continue;
      packed_float[i] = t.Vg(vdata + packed_float_reg);
      packed_float_reg++;
    }
    Id packed_rg8 = old_raw;
    u32 rg8_reg = 0;
    for (u32 i = 0; i < 4; i++) {
      if (!(dmask & (1u << i)))
        continue;
      if (i < 2) {
        const Id keep = t.And(packed_rg8, t.U32(~(0xFFu << (i * 8))));
        packed_rg8 =
            t.Or(keep, t.Shl(store_byte(vdata + rg8_reg), t.U32(i * 8)));
      }
      rg8_reg++;
    }
    // Raw block bits, straight from the VGPRs: 16_16_16_16 packs two
    // components per dword, the 32-bit forms are one per dword. Components the
    // dmask leaves out keep what memory already held.
    Id blk[4] = {old_raw, old_raw_hi, old_raw_2, old_raw_3};
    {
      Id comp[4] = {t.U32(0), t.U32(0), t.U32(0), t.U32(0)};
      u32 reg = 0;
      for (u32 i = 0; i < 4; i++)
        if (dmask & (1u << i))
          comp[i] = t.Vg(vdata + reg++);
      const Id lo16 = t.Or(t.And(comp[0], t.U32(0xFFFF)),
                           t.Shl(t.And(comp[1], t.U32(0xFFFF)), t.U32(16)));
      const Id hi16 = t.Or(t.And(comp[2], t.U32(0xFFFF)),
                           t.Shl(t.And(comp[3], t.U32(0xFFFF)), t.U32(16)));
      blk[0] = t.SelectB(is_rgba16u, lo16, comp[0]);
      blk[1] = t.SelectB(is_rgba16u, hi16, comp[1]);
      blk[2] = comp[2];
      blk[3] = comp[3];
    }
    packed = t.SelectB(is_rg16f, packed_rg16f, packed);
    packed = t.SelectB(is_r16f, packed_r16f, packed);
    packed = t.SelectB(is_rg8, packed_rg8, packed);
    packed = t.SelectB(is_rgba16f, packed_rgba16f_lo, packed);
    packed = t.SelectB(is_r11g11b10f, packed_float[0], packed);
    packed = t.SelectB(is_block_int, blk[0], packed);
    CsSsboStore(t, sc, binding, dword_idx,
                t.SelectB(is_r32, t.Vg(vdata), packed));
    // Both operands of the branch must exist before OpSelectionMerge: nothing
    // may sit between the merge and its OpBranchConditional.
    const Id store_second = logical_or(is_rgba16f, is_block2);
    const Id wide_store = t.m.NewBlock(), store_done = t.m.NewBlock();
    t.m.SelectionMerge(store_done);
    t.m.BranchConditional(store_second, wide_store, store_done);
    t.m.OpenBlock(wide_store);
    CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(1)),
                t.SelectB(is_rgba16f, packed_rgba16f_hi, blk[1]));
    t.m.Branch(store_done);
    t.m.OpenBlock(store_done);
    const Id packed_store = t.m.NewBlock(), packed_done = t.m.NewBlock();
    t.m.SelectionMerge(packed_done);
    t.m.BranchConditional(wide4, packed_store, packed_done);
    t.m.OpenBlock(packed_store);
    for (u32 i = 1; i < 4; i++)
      CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(i)),
                  t.SelectB(is_r11g11b10f, packed_float[i], blk[i]));
    t.m.Branch(packed_done);
    t.m.OpenBlock(packed_done);
  }
  t.m.Branch(merge_blk);
  t.m.OpenBlock(merge_blk);
}

// ---- DS (LDS / subgroup swizzle) -------------------------------------------
// LDS is a Workgroup uint array; addresses are byte-based. Single ops add the
// 16-bit offset to the address; pair ops address elements at offset0/1 *
// element size (x64 for the st64 forms). Indices clamp into the allocation.
bool DsGraphicsSupported(u32 op) {
  switch (op) {
    case 13:   // ds_write_b32
    case 14:   // ds_write2_b32
    case 15:   // ds_write2st64_b32
    case 54:   // ds_read_b32
    case 55:   // ds_read2_b32
    case 56:   // ds_read2st64_b32
    case 77:   // ds_write_b64
    case 78:   // ds_write2_b64
    case 118:  // ds_read_b64
    case 119:  // ds_read2_b64
    case 222:  // ds_write_b96
    case 223:  // ds_write_b128
    case 254:  // ds_read_b96
    case 255:  // ds_read_b128
      return true;
    default:
      return false;  // atomics and the rest: no Private lowering exists
  }
}

// DELTA_GPU_DSMARK: see the trace store in EmitDs.
bool DsMark() {
  static const bool on = std::getenv("DELTA_GPU_DSMARK") != nullptr;
  return on;
}

u32 GraphicsLdsDwords(const Program& program, const u8* reachable) {
  u32 max_bytes = 0;
  bool any = false;
  for (size_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (inst.enc != Enc::kDs || (reachable && !reachable[i]))
      continue;
    if (!DsGraphicsSupported(inst.opcode))
      continue;
    any = true;
    const u32 w = inst.raw[0];
    u32 reach = 0;
    switch (inst.opcode) {
      case 14:
      case 55:  // pair forms: two byte offsets, each scaled by the element size
      case 78:
      case 119:
        reach = std::max(w & 0xFF, (w >> 8) & 0xFF) * 8u;
        break;
      case 15:
      case 56:  // ...and the st64 forms stride 64 elements per unit
        reach = std::max(w & 0xFF, (w >> 8) & 0xFF) * 8u * 64u;
        break;
      default:
        reach = w & 0xFFFF;  // single form: one 16-bit byte offset
        break;
    }
    max_bytes = std::max(max_bytes, reach);
  }
  // Keyed on whether any DS instruction exists, not on the largest offset: a
  // shader whose accesses all sit at offset 0 still needs the array.
  if (!any)
    return 0;
  // Plus the per-lane span the address itself can carry (a 64-lane wave, one
  // dword each) and room for the widest element.
  const u32 dwords = (max_bytes + 64u * 4u + 8u) / 4u;
  return std::min(dwords, 4096u);
}

void EmitDs(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  if (op != 0x35 && ((!sc.lds_var && !sc.lds_wave_base) || !sc.lds_dwords)) {
    sc.cs_unsupported = true;
    return;
  }
  const u32 offset0 = w & 0xFF, offset1 = (w >> 8) & 0xFF;
  const u32 offset16 = w & 0xFFFF;
  const u32 addr_reg = w1 & 0xFF, data0 = (w1 >> 8) & 0xFF;
  const u32 data1 = (w1 >> 16) & 0xFF, vdst = (w1 >> 24) & 0xFF;

  // DS word 0 bit 17 selects the GLOBAL data share, which is shared across
  // wavefronts and thread groups. Private storage is maximally wrong for it,
  // and the Workgroup array is not right either.
  if ((w >> 17) & 1) {
    WarnUnsupported("ds.gds", op, w, w1);
    return;
  }
  const Id p_lds = t.m.TypePointer(sc.lds_storage, t.t_u);
  // The shared path puts this wave's block at lds_wave_base, so a slot another
  // lane wrote is the same slot this lane reads.
  const auto lds_at = [&](Id byte_addr, bool store = false) {
    const Id idx = t.UMin(t.Shr(byte_addr, t.U32(2)), t.U32(sc.lds_dwords - 1));
    if (sc.lds_wave_base) {
      Id abs = t.Add(sc.lds_wave_base, idx);
      // EXEC gates an LDS write. Without this an inactive lane writes its
      // stale address register over the block the active lanes staged their
      // vertices in.
      if (store && t.wave_masks)
        abs = t.SelectB(t.LaneActive(t.Exec()), abs, t.U32(kLdsTrashDword));
      return t.m.AccessChain(p_lds, t.lds_buf_var, {t.U32(0), abs});
    }
    return t.m.AccessChain(p_lds, sc.lds_var, {idx});
  };
  const Id addr = t.Vg(addr_reg);
  // DELTA_GPU_DSMARK: stamp the trace area with this store's pc, so a dump of
  // the shared scratch says which DS instructions a draw actually reached.
  if (DsMark() && sc.lds_wave_base) {
    const Id p = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
    const u32 slot = kLdsTraceBase + (inst.pc & 0x1FF) * 2;
    t.m.Store(t.m.AccessChain(p, t.lds_buf_var, {t.U32(0), t.U32(slot)}),
              t.U32(0xD5000000u | (inst.pc & 0xFFFF)));
    t.m.Store(t.m.AccessChain(p, t.lds_buf_var, {t.U32(0), t.U32(slot + 1)}),
              t.Sg(106));
  }
  const auto single_addr = [&] { return t.Add(addr, t.U32(offset16)); };
  const auto pair_addr = [&](u32 which, u32 elem_bytes, bool st64) {
    const u32 off =
        (which == 0 ? offset0 : offset1) * elem_bytes * (st64 ? 64u : 1u);
    return t.Add(addr, t.U32(off));
  };

  // The LDS atomics, no-return at 0..11 and returning the old value at 32..43.
  // One shape covers both: the opcode picks the SPIR-V atomic, the base picks
  // whether the old value lands in vdst.
  const auto lds_atomic = [&](spv::Op atomic) {
    const Id old = t.m.Emit(
        atomic, t.t_u,
        {lds_at(single_addr()),
         t.U32(static_cast<u32>(spv::Scope::Workgroup)),
         t.U32(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease) |
               static_cast<u32>(spv::MemorySemanticsMask::WorkgroupMemory)),
         t.Vg(data0)});
    if (op >= 32)
      t.SetVg(vdst, old);
  };
  switch (op) {
    case 0:
    case 32:
      lds_atomic(spv::Op::OpAtomicIAdd);  // ds_add[_rtn]_u32
      break;
    case 1:
    case 33:
      lds_atomic(spv::Op::OpAtomicISub);  // ds_sub[_rtn]_u32
      break;
    case 5:
    case 37:
      lds_atomic(spv::Op::OpAtomicSMin);  // ds_min[_rtn]_i32
      break;
    case 6:
    case 38:
      lds_atomic(spv::Op::OpAtomicSMax);  // ds_max[_rtn]_i32
      break;
    case 7:
    case 39:
      lds_atomic(spv::Op::OpAtomicUMin);  // ds_min[_rtn]_u32
      break;
    case 8:
    case 40:
      lds_atomic(spv::Op::OpAtomicUMax);  // ds_max[_rtn]_u32
      break;
    case 9:
    case 41:
      lds_atomic(spv::Op::OpAtomicAnd);  // ds_and[_rtn]_b32
      break;
    case 10:
    case 42:
      lds_atomic(spv::Op::OpAtomicOr);  // ds_or[_rtn]_b32
      break;
    case 11:
    case 43:
      lds_atomic(spv::Op::OpAtomicXor);  // ds_xor[_rtn]_b32
      break;
    case 53: {  // ds_swizzle_b32
      if (!sc.subgroup_local_id) {
        if (sc.is_cs)
          sc.cs_unsupported = true;
        break;
      }
      const Id lane = t.m.Load(t.t_u, sc.subgroup_local_id);
      Id source_lane;
      if (offset16 & 0x8000) {
        const Id quad_lane = t.And(lane, t.U32(3));
        const Id selector =
            t.And(t.Shr(t.U32(offset16), t.Mul(quad_lane, t.U32(2))), t.U32(3));
        source_lane = t.Or(t.And(lane, t.U32(~3u)), selector);
      } else {
        const u32 and_mask = offset16 & 0x1f;
        const u32 or_mask = (offset16 >> 5) & 0x1f;
        const u32 xor_mask = (offset16 >> 10) & 0x1f;
        Id low = t.And(lane, t.U32(and_mask));
        low = t.Or(low, t.U32(or_mask));
        low = t.Xor(low, t.U32(xor_mask));
        source_lane = t.Or(t.And(lane, t.U32(~31u)), low);
      }
      const Id scope = t.U32(static_cast<u32>(spv::Scope::Subgroup));
      const Id value = t.m.Emit(spv::Op::OpGroupNonUniformShuffle, t.t_u,
                                {scope, t.Vg(addr_reg), source_lane});
      const Id source_exec = t.m.Emit(spv::Op::OpGroupNonUniformShuffle, t.t_u,
                                      {scope, t.Exec(), source_lane});
      t.SetVg(vdst, t.SelectB(t.IsNonZero(source_exec), value, t.U32(0)));
      break;
    }
    case 13:  // ds_write_b32
      t.m.Store(lds_at(single_addr(), true), t.Vg(data0));
      break;
    case 14:
    case 15: {  // ds_write2_b32 / ds_write2st64_b32
      const bool st64 = op == 15;
      t.m.Store(lds_at(pair_addr(0, 4, st64), true), t.Vg(data0));
      t.m.Store(lds_at(pair_addr(1, 4, st64), true), t.Vg(data1));
      break;
    }
    case 54:  // ds_read_b32
      t.SetVg(vdst, t.m.Load(t.t_u, lds_at(single_addr())));
      break;
    case 55:
    case 56: {  // ds_read2_b32 / ds_read2st64_b32
      const bool st64 = op == 56;
      t.SetVg(vdst, t.m.Load(t.t_u, lds_at(pair_addr(0, 4, st64))));
      t.SetVg(vdst + 1, t.m.Load(t.t_u, lds_at(pair_addr(1, 4, st64))));
      break;
    }
    case 77: {  // ds_write_b64
      const Id a = single_addr();
      t.m.Store(lds_at(a, true), t.Vg(data0));
      t.m.Store(lds_at(t.Add(a, t.U32(4)), true), t.Vg(data0 + 1));
      break;
    }
    case 222:
    case 223: {  // ds_write_b96 / ds_write_b128
      const u32 n = op == 222 ? 3u : 4u;
      const Id a = single_addr();
      for (u32 i = 0; i < n; i++)
        t.m.Store(lds_at(t.Add(a, t.U32(i * 4)), true), t.Vg(data0 + i));
      break;
    }
    case 254:
    case 255: {  // ds_read_b96 / ds_read_b128
      const u32 n = op == 254 ? 3u : 4u;
      const Id a = single_addr();
      for (u32 i = 0; i < n; i++)
        t.SetVg(vdst + i, t.m.Load(t.t_u, lds_at(t.Add(a, t.U32(i * 4)))));
      break;
    }
    case 118: {  // ds_read_b64
      const Id a = single_addr();
      t.SetVg(vdst, t.m.Load(t.t_u, lds_at(a)));
      t.SetVg(vdst + 1, t.m.Load(t.t_u, lds_at(t.Add(a, t.U32(4)))));
      break;
    }
    case 78: {  // ds_write2_b64, the store half of case 119
      const u32 src[2] = {data0, data1};
      for (u32 i = 0; i < 2; i++) {
        const Id a = t.And(pair_addr(i, 8, false), t.U32(~7u));
        t.m.Store(lds_at(a, true), t.Vg(src[i]));
        t.m.Store(lds_at(t.Add(a, t.U32(4)), true), t.Vg(src[i] + 1));
      }
      break;
    }
    case 119: {  // ds_read2_b64
      for (u32 i = 0; i < 2; i++) {
        const Id a = t.And(pair_addr(i, 8, false), t.U32(~7u));
        t.SetVg(vdst + i * 2, t.m.Load(t.t_u, lds_at(a)));
        t.SetVg(vdst + i * 2 + 1, t.m.Load(t.t_u, lds_at(t.Add(a, t.U32(4)))));
      }
      break;
    }
    default:
      WarnUnsupported("ds", op, w, w1);
      sc.cs_unsupported = true;
      break;
  }
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
