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
#include <algorithm>

#include "gpu/gcn/gcn_audit.h"
#include "gpu/gcn/spirv/translator.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(float, kDebugUv, "DELTA_GPU_DEBUGUV", 0.f);
}  // namespace

namespace {
DELTA_OPTION(int, kPsTexBind, "DELTA_GPU_PSTEX", 0);
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
Id CsSsboPtr(Translator& t, StageContext& sc, uint32_t binding, Id dword_idx) {
  return SsboPtr(t, sc.cs_ssbo[binding], dword_idx);
}
Id CsSsboLoad(Translator& t, StageContext& sc, uint32_t binding, Id dword_idx) {
  return SsboLoad(t, sc.cs_ssbo[binding], dword_idx);
}
void CsSsboStore(Translator& t,
                 StageContext& sc,
                 uint32_t binding,
                 Id dword_idx,
                 Id value) {
  t.m.Store(CsSsboPtr(t, sc, binding, dword_idx), value);
}
int CsBindingFor(StageContext& sc, uint32_t pc) {
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
  for (uint32_t s : {1u, 2u, 4u, 8u, 16u})
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
  for (uint32_t i = 0; i < 3; i++) {
    const Id ok = t.IsZero(t.And(t.Mul(aligned, height), t.U32(63)));
    aligned = t.SelectB(ok, aligned, t.Add(aligned, t.U32(16)));
  }
  return t.SelectB(linear_general, raw, aligned);
}

// MUBUF/MTBUF shared addressing: byte offset within the bound resource.
// soffset + instruction offset, plus the vaddr index/offset registers per the
// IDXEN/OFFEN bits (both set: vaddr = index, vaddr+1 = offset).
Id BufferByteOffset(Translator& t,
                    const Inst& inst,
                    uint32_t inst_offset,
                    bool idxen,
                    bool offen,
                    uint32_t vaddr,
                    uint32_t srsrc,
                    uint32_t soffset_field) {
  Id byte_off =
      t.Add(t.SrcRaw(soffset_field, inst.literal), t.U32(inst_offset));
  uint32_t va = vaddr;
  if (idxen) {
    const Id stride = t.And(t.Shr(t.Sg(srsrc + 1), t.U32(16)), t.U32(0x3FFF));
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
bool MtbufIsRawDwords(const Inst& inst, uint32_t n) {
  const uint32_t w = inst.raw[0];
  if (inst.opcode >= 8)  // d16 forms pack two components per dword
    return false;
  const uint32_t nfmt = (w >> 23) & 0x7;
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
                uint32_t bits,
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
                   uint32_t binding,
                   Id byte_off,
                   uint32_t bits,
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
uint32_t SmrdLoadCount(uint32_t op) {
  if (op < 0x08 || op > 0x0c)
    return 0;  // s_buffer_load_dword{,x2,x4,x8,x16}
  return op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
}

// Dwords moved by any SMRD scalar load: s_load_dword{,x2,x4,x8,x16} (op
// 0x00..0x04, addressed through a 2-dword flat pointer) as well as the
// s_buffer_load family. Both feed the same UBO window; only the descriptor
// they are addressed through differs.
uint32_t SmrdDwordCount(uint32_t op) {
  return op <= 0x04 ? (1u << op) : SmrdLoadCount(op);
}

// Cbuffer bindings are keyed by the SGPR the descriptor is read from. The same
// SGPR can hold a flat pointer for one load and a V# for another (user data
// s[0:1] as a table pointer, s[0:3] as a buffer), so the descriptor kind is
// part of the key -- otherwise whichever load came first would capture the
// other's window.
uint32_t CbufBindKey(uint32_t base_sgpr, bool pointer) {
  return base_sgpr | (pointer ? 0x100u : 0u);
}

void EmitCbufSmrd(Translator& t,
                  const Inst& inst,
                  const std::unordered_map<uint32_t, uint32_t>& bindings) {
  const uint32_t w = inst.raw[0], n = SmrdDwordCount(inst.opcode);
  const uint32_t sdst = (w >> 15) & 0x7F, base_sgpr = ((w >> 9) & 0x3F) * 2;
  const bool imm = (w >> 8) & 1;
  const auto it = bindings.find(CbufBindKey(base_sgpr, inst.opcode <= 0x04));
  if (!n || it == bindings.end())
    return;
  const uint32_t off = w & 0xFF;
  if (imm) {
    for (uint32_t i = 0; i < n; i++)
      t.SetSg(sdst + i, t.CbufDword(it->second, off + i));
  } else {
    const Id base = t.Shr(t.SrcRaw(off, inst.literal), t.U32(2));
    for (uint32_t i = 0; i < n; i++)
      t.SetSg(sdst + i, t.CbufDwordId(it->second, t.Add(base, t.U32(i))));
  }
}

bool PlanCbufs(const Program& program,
               uint32_t first_binding,
               std::vector<ShaderCbuf>& cbufs,
               std::unordered_map<uint32_t, uint32_t>& bindings,
               const uint8_t* reachable) {
  uint32_t index = 0;
  for (const Inst& inst : program) {
    if (reachable && !reachable[index]) {
      index++;
      continue;
    }
    index++;
    if (inst.enc != Enc::kSmrd)
      continue;
    const uint32_t n = SmrdDwordCount(inst.opcode);
    if (!n)
      continue;
    const uint32_t w = inst.raw[0], base_sgpr = ((w >> 9) & 0x3F) * 2;
    // The descriptor may live above user data when an earlier scalar load put
    // it there. Draw-time scalar replay resolves both V#s and flat pointers at
    // the consuming load, so do not discard those chained descriptors.
    const bool pointer = inst.opcode <= 0x04;
    const auto [it, inserted] = bindings.emplace(
        CbufBindKey(base_sgpr, pointer), first_binding + cbufs.size());
    if (inserted) {
      if (it->second >= kMaxCbufBindings)
        return false;
      ShaderCbuf cb{};
      cb.binding = it->second;
      cb.ud_sgpr = base_sgpr;
      cb.pointer = pointer;
      cbufs.push_back(cb);
    }
    const uint32_t off = w & 0xFF;
    const uint32_t end =
        ((w >> 8) & 1)
            ? off + n
            : (off == 0xFF && inst.has_literal ? inst.literal / 4 + n : 256);
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
void EmitMimg(Translator& t,
              const Inst& inst,
              StageContext& sc,
              const Id* address) {
  const uint32_t w0 = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  const uint32_t dmask = (w0 >> 8) & 0xF, vaddr = w1 & 0xFF;
  const uint32_t vdata = (w1 >> 8) & 0xFF;
  const bool dref = op == 0x28 || op == 0x2f;
  const bool offset = op == 0x37;
  const bool gather = op == 0x47;
  const auto addr_u = [&](uint32_t index) {
    return address ? address[index] : t.Vg(vaddr + index);
  };
  const auto addr_f = [&](uint32_t index) {
    return t.m.Bitcast(t.t_f, addr_u(index));
  };

  uint32_t bind = StageContext::kMaxPsSamplers;
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
  const uint32_t coord_components = arrayed || is_3d ? 3u : 2u;
  // A 1D T# is bound as a height-1 2D image: the address body carries x
  // (+layer), and y is synthesized at the row centre. Bloodborne's gamma pass
  // samples a 1025x1 R32F 1D-array LUT per channel; reading the body as 2D
  // takes the layer index for y and samples nothing.
  const bool is_1d = ((sc.tex_1d_mask >> bind) & 1u) != 0 && !is_3d;
  const uint32_t addr_components =
      is_1d ? (arrayed ? 2u : 1u) : coord_components;
  // OpImageSampleDref* / OpImageGather are undefined on Dim3D.
  if (is_3d && (dref || gather)) {
    WarnUnsupported("mimg.3d-dref-gather", op, w0, w1);
    return;
  }

  if (op == 0x08 || op == 0x09) {  // image_store[_mip]
    const uint32_t type_idx = arrayed ? 1u : 0u;
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
    uint32_t source = 0;
    for (uint32_t channel = 0; channel < 4; channel++)
      if (dmask & (1u << channel))
        components[channel] = t.VgF(vdata + source++);
    const Id texel = t.m.CompositeConstruct(
        t.t_v4, {components[0], components[1], components[2], components[3]});
    const Id image = t.m.Load(t.storage_img_types[type_idx], sc.tex_vars[bind]);
    if (!t.predicate_vector) {
      t.m.EmitVoid(spv::Op::OpImageWrite, {image, coord, texel});
      return;
    }
    const Id live = t.IsNonZero(t.Exec());
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
    const uint32_t type_idx =
        (arrayed ? 1u : 0u) | (dref ? 2u : 0u) | (is_3d ? 4u : 0u);
    if (!t.img_types[type_idx]) {
      t.img_types[type_idx] = t.m.TypeImage(
          t.t_f, is_3d ? spv::Dim::Dim3D : spv::Dim::Dim2D, dref ? 1 : 0,
          arrayed ? 1 : 0, 0, 1, spv::ImageFormat::Unknown);
      t.sampled_types[type_idx] = t.m.TypeSampledImage(t.img_types[type_idx]);
      t.sampled_ptrs[type_idx] = t.m.TypePointer(
          spv::StorageClass::UniformConstant, t.sampled_types[type_idx]);
    }
    const Id var = t.m.Variable(t.sampled_ptrs[type_idx],
                                spv::StorageClass::UniformConstant);
    t.m.Decorate(var, spv::Decoration::DescriptorSet, {0});
    t.m.Decorate(var, spv::Decoration::Binding, {bind});
    t.m.Name(var, "tex" + std::to_string(bind));
    sc.tex_vars[bind] = var;
    sc.tex_types[bind] = type_idx;
  }
  const uint32_t type_idx = sc.tex_types[bind];
  const Id img_ty = t.img_types[type_idx];
  const Id si = t.m.Load(t.sampled_types[type_idx], sc.tex_vars[bind]);

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
    uint32_t out = 0;
    for (int i = 0; i < 4; i++)
      if (dmask & (1 << i))
        t.SetVg(vdata + out++, comps[i]);
    return;
  }

  const uint32_t body_addr = vaddr + (offset ? 1u : 0u) + (dref ? 1u : 0u);
  const uint32_t body_index = body_addr - vaddr;
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
  const uint32_t lod_operand =
      static_cast<uint32_t>(spv::ImageOperandsMask::Lod);
  const bool known = op == 0x00 || op == 0x01 || op == 0x20 || op == 0x21 ||
                     op == 0x24 || op == 0x25 || op == 0x27 || op == 0x28 ||
                     op == 0x2f || op == 0x37 || op == 0x47;
  if (!known)
    WarnUnsupported("mimg", op, w0, w1);

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
        t.m.Emit(spv::Op::OpImageFetch, t.t_v4, {img, ic, lod_operand, lod});
  } else if (op == 0x24) {  // image_sample_l: explicit LOD after the body
    texel = t.m.Emit(
        spv::Op::OpImageSampleExplicitLod, t.t_v4,
        {si, uv, lod_operand,
         addr_f(is_1d ? body_index + addr_components : coord_components)});
  } else if (op == 0x28) {  // image_sample_c: z-compare precedes the body
    texel = t.m.Emit(spv::Op::OpImageSampleDrefImplicitLod, t.t_f,
                     {si, uv, addr_f(0)});
  } else if (op == 0x2f) {  // image_sample_c_lz: PCF forced to level zero
    texel = t.m.Emit(spv::Op::OpImageSampleDrefExplicitLod, t.t_f,
                     {si, uv, addr_f(0), lod_operand, t.F32(0.0f)});
  } else if (op == 0x27 || op == 0x37) {  // image_sample_lz[_o]: forced LOD 0
    texel = t.m.Emit(spv::Op::OpImageSampleExplicitLod, t.t_v4,
                     {si, uv, lod_operand, t.F32(0.0f)});
  } else if (gather) {  // DMASK selects the gathered channel
    uint32_t component = 0;
    while (component < 3 && !(dmask & (1u << component)))
      component++;
    texel =
        t.m.Emit(spv::Op::OpImageGather, t.t_v4, {si, uv, t.U32(component)});
  } else {  // image_sample / _cl / _b (bias/derivs ignored): implicit LOD
    texel = t.m.Emit(spv::Op::OpImageSampleImplicitLod, t.t_v4, {si, uv});
  }

  // DELTA_GPU_PSTEX=<binding+1>: remember this binding's texel so the PS
  // epilogue can export it (0 = the last sample, whatever it was).
  if (!dref && !gather &&
      (kPsTexBind == 0 || bind == (uint32_t)(kPsTexBind - 1)))
    t.last_texel = texel;

  // DELTA_GPU_DEBUGUV: output the sample UV as R/G instead of the texel, to see
  // the coordinate distribution reaching the sampler (normalized 0..1 vs texel
  // units). Diagnostic only.
  if (kDebugUv != 0.f && !dref && !gather)
    texel = t.m.CompositeConstruct(
        t.t_v4, {t.FMul(x, t.F32(kDebugUv)), t.FMul(y, t.F32(kDebugUv)),
                 t.F32(0.f), t.F32(1.f)});

  if (dref) {
    if (dmask)
      t.SetVgF(vdata, texel);
    return;
  }
  if (gather) {
    for (int i = 0; i < 4; i++)
      t.SetVgF(vdata + i, t.m.CompositeExtract(t.t_f, texel, i));
    return;
  }
  uint32_t out = 0;
  for (int i = 0; i < 4; i++)
    if (dmask & (1 << i))
      t.SetVgF(vdata + out++, t.m.CompositeExtract(t.t_f, texel, i));
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
uint32_t GfxMubufLoadDwords(uint32_t op) {
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
uint32_t GfxMtbufLoadDwords(uint32_t opcode) {
  return opcode <= 0x03 ? opcode + 1 : 0;
}

}  // namespace

void PlanGfxBuffers(const Program& program,
                    uint32_t first_binding,
                    const std::unordered_set<uint32_t>* claimed,
                    std::vector<ShaderBuffer>& buffers,
                    std::unordered_map<uint32_t, uint32_t>& bindings,
                    const uint8_t* reachable) {
  // Which scalar load last wrote each SGPR range, so a descriptor quad that is
  // reloaded with a second V# gets a second binding instead of silently
  // inheriting the first one's buffer (same reasoning as PlanCsResources).
  struct ScalarLoad {
    uint32_t sgpr;
    uint32_t dwords;
    uint32_t version;
  };
  std::vector<ScalarLoad> loads;
  const auto descriptor_version = [&](uint32_t sgpr) {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + 4 <= it->sgpr + it->dwords)
        return it->version;
    return UINT32_MAX;  // inline user data
  };

  std::unordered_map<uint64_t, uint32_t> binding_by_descriptor;
  uint32_t index = 0;
  for (const Inst& inst : program) {
    const uint32_t inst_idx = index++;
    if (reachable && !reachable[inst_idx])
      continue;
    const uint32_t w = inst.raw[0];
    if (inst.enc == Enc::kSmrd) {
      const uint32_t n = SmrdDwordCount(inst.opcode);
      if (!n)
        continue;
      const uint32_t sdst = (w >> 15) & 0x7F;
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
    const uint32_t srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    const uint64_t key =
        static_cast<uint64_t>(srsrc) |
        (static_cast<uint64_t>(descriptor_version(srsrc)) << 8);
    const auto found = binding_by_descriptor.find(key);
    if (found != binding_by_descriptor.end()) {
      bindings[inst.pc] = found->second;
      continue;
    }
    const uint32_t binding =
        first_binding + static_cast<uint32_t>(buffers.size());
    if (binding >= kMaxGfxBuffers) {
      WarnUnsupported("mubuf.binding-count", binding + 1);
      continue;  // over the cap: the emitter warns and leaves the VGPRs zero
    }
    binding_by_descriptor[key] = binding;
    bindings[inst.pc] = binding;
    buffers.push_back({binding, srsrc, inst.pc});
  }
}

void EmitGfxMubuf(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t op = (w >> 18) & 0x7F, inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const uint32_t n = GfxMubufLoadDwords(op);
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
  for (uint32_t i = 0; i < n; i++)
    t.SetVg(vdata + i, SsboLoad(t, var,
                                t.UMin(t.Add(dword_idx, t.U32(i)),
                                       t.U32(kGfxBufferDwords - 1))));
}

void EmitGfxMtbuf(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const uint32_t n = GfxMtbufLoadDwords(inst.opcode);
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
  for (uint32_t i = 0; i < n; i++)
    t.SetVg(vdata + i, SsboLoad(t, var,
                                t.UMin(t.Add(dword_idx, t.U32(i)),
                                       t.U32(kGfxBufferDwords - 1))));
}

// ---- compute: resource plan -------------------------------------------------
bool PlanCsResources(const Program& program,
                     const uint8_t* reachable,
                     uint32_t lds_dwords,
                     RecompiledCs& r,
                     std::unordered_map<uint32_t, uint32_t>& bind) {
  struct ScalarLoad {
    uint32_t sgpr;
    uint32_t dwords;
    uint32_t version;
  };
  std::vector<ScalarLoad> loads;
  const auto descriptor_version = [&](uint32_t sgpr, uint32_t dwords) {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->version;
    return UINT32_MAX;  // inline user data
  };
  std::unordered_map<uint64_t, uint32_t> resource_by_version;
  const auto resource = [&](uint32_t pc, uint32_t base_sgpr, uint32_t dwords,
                            uint8_t kind, bool written, uint32_t min_bytes) {
    const uint64_t key =
        static_cast<uint64_t>(kind) | (static_cast<uint64_t>(base_sgpr) << 8) |
        (static_cast<uint64_t>(descriptor_version(base_sgpr, dwords)) << 16);
    const auto it = resource_by_version.find(key);
    if (it != resource_by_version.end()) {
      CsResource& res = r.resources[it->second];
      res.written = res.written || written;
      if (min_bytes > res.min_bytes)
        res.min_bytes = min_bytes;
      bind[pc] = it->second;
      return true;
    }
    const uint32_t idx = static_cast<uint32_t>(r.resources.size());
    if (idx >= kMaxCsResources) {
      WarnUnsupported("cs.resource-count", idx + 1);
      return false;
    }
    resource_by_version[key] = idx;
    bind[pc] = idx;
    r.resources.push_back({base_sgpr, pc, idx, kind, written, min_bytes});
    return true;
  };

  uint32_t idx = 0;
  for (const Inst& inst : program) {
    const uint32_t inst_idx = idx++;
    if (reachable && !reachable[inst_idx])
      continue;  // dead block/padding
    const uint32_t w = inst.raw[0], w1 = inst.raw[1];
    switch (inst.enc) {
      case Enc::kSmrd: {
        const uint32_t op = inst.opcode, sbase = (w >> 9) & 0x3F;
        const bool imm = (w >> 8) & 1;
        const uint32_t off = w & 0xFF;
        if (op > 0x0c)
          return false;  // beyond s_buffer_load_dwordx16
        const uint32_t n = op < 0x08 ? (1u << op) : SmrdLoadCount(op);
        if (!n)
          return false;  // reserved scalar-load opcode
        const uint32_t bytes =
            imm ? (off + n) * 4
                : (off == 0xFF && inst.has_literal ? inst.literal + n * 4 : 0);
        const uint32_t base_sgpr = sbase * 2;
        const uint8_t kind = op < 0x08 ? 2 : 0;
        if (!resource(inst.pc, base_sgpr, kind == 2 ? 2 : 4, kind, false,
                      bytes))
          return false;
        const uint32_t sdst = (w >> 15) & 0x7F;
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
        const uint32_t op = (w >> 18) & 0x7F;
        const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
        const bool load = op <= 0x03 || (op >= 0x08 && op <= 0x0f);
        const bool store = (op >= 0x04 && op <= 0x07) || op == 0x18 ||
                           op == 0x1a || (op >= 0x1c && op <= 0x1f);
        if (!load && !store)
          return false;  // atomics etc.
        if (!resource(inst.pc, srsrc, 4, 0, store, 0))
          return false;
        break;
      }
      case Enc::kMtbuf: {
        const uint32_t op = (w >> 16) & 0x7;
        const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
        if (!resource(inst.pc, srsrc, 4, 0, op >= 4, 0))
          return false;
        break;
      }
      case Enc::kMimg: {
        const uint32_t op = (w >> 18) & 0x7F;
        const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
        const bool r128 = (w & 0x8000) != 0;
        if (op == 0x0e)
          break;  // get_resinfo reads only descriptor SGPRs
        const bool store = op == 0x08 || op == 0x09;
        const bool load = op == 0x00 || op == 0x01;
        const bool sample = op == 0x24 || op == 0x27;
        if ((!store && !load && !sample) || r128 || srsrc + 7 >= 136)
          return false;
        if (!resource(inst.pc, srsrc, 8, 1, store, 0))
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
  const uint32_t w = inst.raw[0], op = inst.opcode;
  const uint32_t sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
  const bool imm = (w >> 8) & 1;
  const uint32_t off = w & 0xFF, base_sgpr = sbase * 2;
  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const uint32_t n = op < 0x08 ? (1u << op) : SmrdLoadCount(op);
  // Immediate offset is a dword index; a register offset is a byte offset.
  const Id dword_off =
      imm ? t.U32(off) : t.Shr(t.SrcRaw(off, inst.literal), t.U32(2));
  for (uint32_t i = 0; i < n; i++)
    t.SetSg(sdst + i, CsSsboLoad(t, sc, static_cast<uint32_t>(b),
                                 t.Add(dword_off, t.U32(i))));
}

// ---- compute: MUBUF ---------------------------------------------------------
void EmitCsMubuf(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t op = (w >> 18) & 0x7F, inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const uint32_t binding = static_cast<uint32_t>(b);
  const Id byte_off = BufferByteOffset(t, inst, inst_offset, idxen, offen,
                                       vaddr, srsrc, soffset);
  const Id dword_idx = t.Shr(byte_off, t.U32(2));

  // Format load/stores move raw dwords: the buffer-format -> image-store round
  // trip the copy shaders perform is identity, so no conversion is applied.
  const auto load_dwords = [&](uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
      t.SetVg(vdata + i,
              CsSsboLoad(t, sc, binding, t.Add(dword_idx, t.U32(i))));
  };
  const auto store_dwords = [&](uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
      CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(i)), t.Vg(vdata + i));
  };

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
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t op = (w >> 16) & 0x7, inst_offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, soffset = (w1 >> 24) & 0xFF;
  const int b = CsBindingFor(sc, inst.pc);
  if (b < 0) {
    sc.cs_unsupported = true;
    return;
  }
  const uint32_t n = (op & 3) + 1;
  if (!MtbufIsRawDwords(inst, n))
    WarnUnsupported("mtbuf.raw-format", op, w, w1);
  const uint32_t binding = static_cast<uint32_t>(b);
  const Id byte_off = BufferByteOffset(t, inst, inst_offset, idxen, offen,
                                       vaddr, srsrc, soffset);
  const Id dword_idx = t.Shr(byte_off, t.U32(2));
  if (op < 4) {  // tbuffer_load_format_x..xyzw
    for (uint32_t i = 0; i < n; i++)
      t.SetVg(vdata + i,
              CsSsboLoad(t, sc, binding, t.Add(dword_idx, t.U32(i))));
  } else {  // tbuffer_store_format_x..xyzw
    for (uint32_t i = 0; i < n; i++)
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
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t op = (w >> 18) & 0x7F, dmask = (w >> 8) & 0xF;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
  const bool mip_op = op == 0x01 || op == 0x09;
  const bool store = op == 0x08 || op == 0x09;
  const bool load = op == 0x00 || op == 0x01;
  const bool sample = op == 0x24 || op == 0x27;
  const bool resinfo = op == 0x0e;
  if (!store && !load && !sample && !resinfo) {
    sc.cs_unsupported = true;
    return;
  }
  const bool da = (w & 0x4000) != 0;
  // gfx10 NSA names every address component in its own VGPR; the caller hands
  // them over already loaded. Without this a 2D load reads y from vaddr+1,
  // which on an NSA instruction is some unrelated register -- the source image
  // is then addressed by x alone and the result is vertical stripes.
  const auto addr_vg = [&](uint32_t i) {
    return address ? address[i] : t.Vg(vaddr + i);
  };
  const auto addr_vgf = [&](uint32_t i) {
    return t.m.Bitcast(t.t_f, addr_vg(i));
  };

  // T# field extraction (see DecodeTImage for the dword layout).
  const auto field = [&](uint32_t dword, uint32_t shift, uint32_t mask) {
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
  const Id is_array = logical_or(t.Eq(image_type, t.U32(13)),
                                 t.Eq(image_type, t.U32(12)));
  const Id descriptor_layers = t.Add(field(4, 0, 0x1FFF), t.U32(1));

  if (resinfo) {  // dimensions from the descriptor, no memory access
    const Id mip = t.UMin(addr_vg(0), t.Sub(safe_last_mip, base_mip));
    const Id physical = t.Add(base_mip, mip);
    const Id comps[4] = {
        Max1(t, t.Shr(base_width, physical)),
        Max1(t, t.Shr(base_height, physical)),
        t.SelectB(is_array, descriptor_layers, t.U32(1)),
        t.Add(t.Sub(safe_last_mip, base_mip), t.U32(1)),
    };
    uint32_t out = 0;
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
  const uint32_t binding = static_cast<uint32_t>(b);

  // A tiled gfx10.3 surface carries no pitch: the CPU-side layout uses the
  // width, so the shader must index the staging image the same way.
  const Id base_pitch =
      t.rdna_sources ? base_width : t.Add(field(4, 13, 0x3FFF), t.U32(1));
  // gfx10.3 replaces (dfmt, nfmt) with one 9-bit format enum.
  const Id gfmt = field(1, 20, 0x1FF);
  const Id dfmt = field(1, 20, 0x3F), nfmt = field(1, 26, 0xF);
  const Id is_unorm = t.IsZero(nfmt);  // nfmt 0 = UNORM
  const auto is_gfmt = [&](std::initializer_list<uint32_t> values) {
    Id any = t.m.ConstBool(false);
    for (uint32_t v : values)
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
  Id supported_format = logical_or(is_rgba8, is_r32);
  supported_format = logical_or(supported_format, is_rg16f);
  supported_format = logical_or(supported_format, is_r16f);
  supported_format = logical_or(supported_format, is_rg8);
  supported_format = logical_or(supported_format, is_rgba16f);
  supported_format = logical_or(supported_format, is_r11g11b10f);
  const Id supported_type = logical_or(
      logical_or(t.Eq(image_type, t.U32(9)), t.Eq(image_type, t.U32(8))),
      is_array);
  Id requested_mip =
      mip_op ? t.SelectB(is_1d_img, addr_vg(da ? 2 : 1), addr_vg(da ? 3 : 2))
             : t.U32(0);
  if (op == 0x24) {
    const Id lod_addr = t.SelectB(is_1d_img, addr_vgf(da ? 2 : 1),
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
  if (sample) {
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
    y = t.SelectB(is_1d_img, t.U32(0), y);
  }
  // gfx10.3 swizzle mode 0 is the linear one, and it has no pow2-pad bit.
  const Id linear_general = t.Eq(field(3, 20, 0x1F), t.U32(t.rdna_sources ? 0 : 31));
  const Id pow2_pad =
      t.rdna_sources ? t.m.ConstBool(false) : t.IsNonZero(field(3, 25, 1));
  const Id stored_height = t.SelectB(pow2_pad, BitCeil(t, height), height);
  const Id pitch = LinearMipPitch(t, base_pitch, stored_height, physical_mip,
                                  linear_general, pow2_pad);
  const Id base_array =
      t.rdna_sources ? field(4, 16, 0x1FFF) : field(5, 0, 0x1FFF);
  const Id last_array = t.rdna_sources ? t.Add(descriptor_layers, t.U32(~0u))
                                       : field(5, 13, 0x1FFF);
  const Id view_layer =
      da ? t.SelectB(is_1d_img, addr_vg(1), addr_vg(2)) : t.U32(0);
  const Id physical_layer = t.Add(base_array, view_layer);
  const Id padded_layers =
      t.SelectB(pow2_pad, BitCeil(t, descriptor_layers), descriptor_layers);
  const Id layers = t.SelectB(is_array, padded_layers, t.U32(1));
  Id array_ok =
      t.LAnd(t.Ult(base_array, layers), t.Uge(last_array, base_array));
  array_ok = t.LAnd(array_ok, t.Ule(view_layer, t.Sub(last_array, base_array)));
  array_ok = t.LAnd(array_ok, t.Ult(physical_layer, layers));
  const Id layer_ok = t.m.Emit(spv::Op::OpSelect, t.t_bool,
                               {is_array, array_ok, t.m.ConstBool(true)});
  Id valid = t.LAnd(t.Ult(x, width), t.Ult(y, height));
  valid = t.LAnd(valid, layer_ok);
  valid = t.LAnd(valid, t.Uge(last_mip, base_mip));
  valid = t.LAnd(valid, supported_type);
  valid = t.LAnd(valid, supported_format);
  if (load || sample) {  // default loaded components to 0 for invalid accesses
    uint32_t out = 0;
    for (int i = 0; i < 4; i++)
      if (dmask & (1 << i))
        t.SetVg(vdata + out++, t.U32(0));
  }

  const Id access_blk = t.m.NewBlock(), merge_blk = t.m.NewBlock();
  t.m.SelectionMerge(merge_blk);
  t.m.BranchConditional(valid, access_blk, merge_blk);
  t.m.OpenBlock(access_blk);

  const Id layer = t.SelectB(is_array, physical_layer, t.U32(0));
  Id mip_off = t.U32(0);
  for (uint32_t mip = 0; mip < 16; mip++) {
    const Id level = t.U32(mip);
    const Id level_height = Max1(t, t.Shr(base_height, level));
    const Id level_stored =
        t.SelectB(pow2_pad, BitCeil(t, level_height), level_height);
    const Id level_pitch = LinearMipPitch(t, base_pitch, level_stored, level,
                                          linear_general, pow2_pad);
    const Id level_size = t.Mul(t.Mul(level_pitch, level_stored), layers);
    mip_off = t.Add(
        mip_off, t.SelectB(t.Ult(level, physical_mip), level_size, t.U32(0)));
  }
  const Id layer_off = t.Mul(layer, t.Mul(pitch, stored_height));
  const Id texel_idx =
      t.Add(mip_off, t.Add(layer_off, t.Add(t.Mul(y, pitch), x)));
  Id dword_idx = t.SelectB(is_rgba16f, t.Mul(texel_idx, t.U32(2)), texel_idx);
  dword_idx = t.SelectB(is_r11g11b10f, t.Mul(texel_idx, t.U32(4)), dword_idx);

  if (load) {
    const Id raw = CsSsboLoad(t, sc, binding, dword_idx);
    const Id has_second =
        t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_rgba16f, is_r11g11b10f});
    const Id raw_hi = CsSsboLoad(
        t, sc, binding,
        t.SelectB(has_second, t.Add(dword_idx, t.U32(1)), dword_idx));
    const Id raw_2 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(is_r11g11b10f, t.Add(dword_idx, t.U32(2)), dword_idx));
    const Id raw_3 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(is_r11g11b10f, t.Add(dword_idx, t.U32(3)), dword_idx));
    const Id halfs = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {raw});
    const Id halfs_hi = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {raw_hi});
    const Id float_component[4] = {raw, raw_hi, raw_2, raw_3};
    uint32_t out = 0;
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
      t.SetVg(vdata + out++, value);
    }
  } else if (sample) {
    const Id has_second =
        t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_rgba16f, is_r11g11b10f});
    // Storage-buffer index of texel (xx,yy) in the current mip/layer, scaled
    // for the multi-dword formats (matches the single-texel index above).
    const auto texel_at = [&](Id xx, Id yy) {
      Id ti = t.Add(mip_off, t.Add(layer_off, t.Add(t.Mul(yy, pitch), xx)));
      ti = t.SelectB(is_rgba16f, t.Mul(ti, t.U32(2)), ti);
      ti = t.SelectB(is_r11g11b10f, t.Mul(ti, t.U32(4)), ti);
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
    const auto mix = [&](Id a, Id b, Id w) {
      return t.FAdd(a, t.FMul(t.FSub(b, a), w));
    };
    uint32_t out = 0;
    for (int i = 0; i < 4; i++) {
      if (!(dmask & (1 << i)))
        continue;
      const Id top = mix(c00[i], c10[i], wx);
      const Id bot = mix(c01[i], c11[i], wx);
      t.SetVgF(vdata + out++, mix(top, bot, wy));
    }
  } else {
    const auto store_byte = [&](uint32_t reg) {
      const Id value = t.Vg(reg);
      Id normalized = t.FClamp01(t.m.Bitcast(t.t_f, value));
      normalized = t.m.ExtInst(t.t_f, GLSLstd450RoundEven,
                               {t.FMul(normalized, t.F32(255.0f))});
      const Id unorm = t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {normalized});
      return t.And(t.SelectB(is_unorm, unorm, value), t.U32(0xFF));
    };
    const Id old_raw = CsSsboLoad(t, sc, binding, dword_idx);
    const Id has_second =
        t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_rgba16f, is_r11g11b10f});
    const Id old_raw_hi = CsSsboLoad(
        t, sc, binding,
        t.SelectB(has_second, t.Add(dword_idx, t.U32(1)), dword_idx));
    const Id old_raw_2 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(is_r11g11b10f, t.Add(dword_idx, t.U32(2)), dword_idx));
    const Id old_raw_3 = CsSsboLoad(
        t, sc, binding,
        t.SelectB(is_r11g11b10f, t.Add(dword_idx, t.U32(3)), dword_idx));
    Id packed;
    if (dmask == 0xF) {
      packed = store_byte(vdata);
      packed = t.Or(packed, t.Shl(store_byte(vdata + 1), t.U32(8)));
      packed = t.Or(packed, t.Shl(store_byte(vdata + 2), t.U32(16)));
      packed = t.Or(packed, t.Shl(store_byte(vdata + 3), t.U32(24)));
    } else {
      packed = old_raw;  // read-modify-write
      uint32_t comp = 0;
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
    uint32_t half_reg = 0;
    for (uint32_t i = 0; i < 2; i++)
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
    uint32_t wide_reg = 0;
    for (uint32_t i = 0; i < 4; i++)
      if (dmask & (1u << i))
        wide_half[i] = t.m.Bitcast(t.t_f, t.Vg(vdata + wide_reg++));
    const Id packed_rgba16f_lo = t.m.ExtInst(
        t.t_u, GLSLstd450PackHalf2x16,
        {t.m.CompositeConstruct(t.t_v2, {wide_half[0], wide_half[1]})});
    const Id packed_rgba16f_hi = t.m.ExtInst(
        t.t_u, GLSLstd450PackHalf2x16,
        {t.m.CompositeConstruct(t.t_v2, {wide_half[2], wide_half[3]})});
    Id packed_float[4] = {old_raw, old_raw_hi, old_raw_2, old_raw_3};
    uint32_t packed_float_reg = 0;
    for (uint32_t i = 0; i < 4; i++) {
      if (!(dmask & (1u << i)))
        continue;
      packed_float[i] = t.Vg(vdata + packed_float_reg);
      packed_float_reg++;
    }
    Id packed_rg8 = old_raw;
    uint32_t rg8_reg = 0;
    for (uint32_t i = 0; i < 4; i++) {
      if (!(dmask & (1u << i)))
        continue;
      if (i < 2) {
        const Id keep = t.And(packed_rg8, t.U32(~(0xFFu << (i * 8))));
        packed_rg8 =
            t.Or(keep, t.Shl(store_byte(vdata + rg8_reg), t.U32(i * 8)));
      }
      rg8_reg++;
    }
    packed = t.SelectB(is_rg16f, packed_rg16f, packed);
    packed = t.SelectB(is_r16f, packed_r16f, packed);
    packed = t.SelectB(is_rg8, packed_rg8, packed);
    packed = t.SelectB(is_rgba16f, packed_rgba16f_lo, packed);
    packed = t.SelectB(is_r11g11b10f, packed_float[0], packed);
    CsSsboStore(t, sc, binding, dword_idx,
                t.SelectB(is_r32, t.Vg(vdata), packed));
    const Id wide_store = t.m.NewBlock(), store_done = t.m.NewBlock();
    t.m.SelectionMerge(store_done);
    t.m.BranchConditional(is_rgba16f, wide_store, store_done);
    t.m.OpenBlock(wide_store);
    CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(1)), packed_rgba16f_hi);
    t.m.Branch(store_done);
    t.m.OpenBlock(store_done);
    const Id packed_store = t.m.NewBlock(), packed_done = t.m.NewBlock();
    t.m.SelectionMerge(packed_done);
    t.m.BranchConditional(is_r11g11b10f, packed_store, packed_done);
    t.m.OpenBlock(packed_store);
    for (uint32_t i = 1; i < 4; i++)
      CsSsboStore(t, sc, binding, t.Add(dword_idx, t.U32(i)), packed_float[i]);
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
void EmitDs(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  if (op != 0x35 && (!sc.lds_var || !sc.lds_dwords)) {
    sc.cs_unsupported = true;
    return;
  }
  const uint32_t offset0 = w & 0xFF, offset1 = (w >> 8) & 0xFF;
  const uint32_t offset16 = w & 0xFFFF;
  const uint32_t addr_reg = w1 & 0xFF, data0 = (w1 >> 8) & 0xFF;
  const uint32_t data1 = (w1 >> 16) & 0xFF, vdst = (w1 >> 24) & 0xFF;

  const Id p_lds = t.m.TypePointer(spv::StorageClass::Workgroup, t.t_u);
  const auto lds_at = [&](Id byte_addr) {
    const Id idx = t.UMin(t.Shr(byte_addr, t.U32(2)), t.U32(sc.lds_dwords - 1));
    return t.m.AccessChain(p_lds, sc.lds_var, {idx});
  };
  const Id addr = t.Vg(addr_reg);
  const auto single_addr = [&] { return t.Add(addr, t.U32(offset16)); };
  const auto pair_addr = [&](uint32_t which, uint32_t elem_bytes, bool st64) {
    const uint32_t off =
        (which == 0 ? offset0 : offset1) * elem_bytes * (st64 ? 64u : 1u);
    return t.Add(addr, t.U32(off));
  };

  switch (op) {
    case 32: {  // ds_add_rtn_u32
      const Id old = t.m.Emit(
          spv::Op::OpAtomicIAdd, t.t_u,
          {lds_at(single_addr()),
           t.U32(static_cast<uint32_t>(spv::Scope::Workgroup)),
           t.U32(
               static_cast<uint32_t>(spv::MemorySemanticsMask::AcquireRelease) |
               static_cast<uint32_t>(
                   spv::MemorySemanticsMask::WorkgroupMemory)),
           t.Vg(data0)});
      t.SetVg(vdst, old);
      break;
    }
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
        const uint32_t and_mask = offset16 & 0x1f;
        const uint32_t or_mask = (offset16 >> 5) & 0x1f;
        const uint32_t xor_mask = (offset16 >> 10) & 0x1f;
        Id low = t.And(lane, t.U32(and_mask));
        low = t.Or(low, t.U32(or_mask));
        low = t.Xor(low, t.U32(xor_mask));
        source_lane = t.Or(t.And(lane, t.U32(~31u)), low);
      }
      const Id scope = t.U32(static_cast<uint32_t>(spv::Scope::Subgroup));
      const Id value = t.m.Emit(spv::Op::OpGroupNonUniformShuffle, t.t_u,
                                {scope, t.Vg(addr_reg), source_lane});
      const Id source_exec = t.m.Emit(spv::Op::OpGroupNonUniformShuffle, t.t_u,
                                      {scope, t.Exec(), source_lane});
      t.SetVg(vdst, t.SelectB(t.IsNonZero(source_exec), value, t.U32(0)));
      break;
    }
    case 13:  // ds_write_b32
      t.m.Store(lds_at(single_addr()), t.Vg(data0));
      break;
    case 14:
    case 15: {  // ds_write2_b32 / ds_write2st64_b32
      const bool st64 = op == 15;
      t.m.Store(lds_at(pair_addr(0, 4, st64)), t.Vg(data0));
      t.m.Store(lds_at(pair_addr(1, 4, st64)), t.Vg(data1));
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
      t.m.Store(lds_at(a), t.Vg(data0));
      t.m.Store(lds_at(t.Add(a, t.U32(4))), t.Vg(data0 + 1));
      break;
    }
    case 118: {  // ds_read_b64
      const Id a = single_addr();
      t.SetVg(vdst, t.m.Load(t.t_u, lds_at(a)));
      t.SetVg(vdst + 1, t.m.Load(t.t_u, lds_at(t.Add(a, t.U32(4)))));
      break;
    }
    case 119: {  // ds_read2_b64
      for (uint32_t i = 0; i < 2; i++) {
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
