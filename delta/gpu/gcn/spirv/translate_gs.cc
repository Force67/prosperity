/*
 * PS4Delta: the legacy ES -> GS pipeline. The ES writes its outputs into the
 * ESGS ring, the GS reads them back per input vertex and writes the GSVS ring,
 * and the copy shader in the VS slot moves GSVS data into exports. Vulkan
 * passes vertex data between stages itself, so the rings become the stage
 * interface and the copy shader is only read for its export map.
 */
#ifdef DELTA_HAVE_SPIRV_BACKEND
#include <algorithm>
#include <array>

#include "gpu/gcn/spirv/translator.h"

namespace gpu::gcn {
namespace {

constexpr u32 kBufferLoadDword = 0x0c;
constexpr u32 kBufferStoreDword = 0x1c;
constexpr u32 kExpPos0 = 12, kExpPos1 = 13, kExpParam0 = 32;

// Byte offset of a ring access within one lane's item. The ring base the
// hardware passes in an SGPR is seeded as zero, so the offsets the compiler
// adds on top are all that is left. A component is 64 lanes of dwords apart,
// and a GS vertex offset VGPR selects the lane.
Id RingAddress(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  Id address =
      t.Add(t.SrcRaw((w1 >> 24) & 0xFF, inst.literal), t.U32(w & 0xFFF));
  if ((w >> 12) & 1)
    address = t.Add(address, t.Vg(w1 & 0xFF));
  return address;
}

bool RingDwordOp(const Inst& inst, u32 op) {
  const u32 w = inst.raw[0];
  return ((w >> 18) & 0x7F) == op && !((w >> 13) & 1);  // no IDXEN
}

}  // namespace

bool ParseCopyShader(const Program& program,
                     u32 max_vert_out,
                     std::vector<GsCopyExport>& out) {
  struct Value {
    enum Kind : u8 { kUnknown, kConst, kComp } kind = kUnknown;
    u32 bits = 0;
  };
  Value sgpr[128], vgpr[256];
  const u32 comp_bytes = max_vert_out * 64;
  if (!comp_bytes)
    return false;
  const auto constant = [&](u32 field, u32 literal, u32& bits) {
    static constexpr u32 kFloats[8] = {0x3f000000, 0xbf000000, 0x3f800000,
                                       0xbf800000, 0x40000000, 0xc0000000,
                                       0x40800000, 0xc0800000};
    if (field < 128) {
      bits = sgpr[field].bits;
      return sgpr[field].kind == Value::kConst;
    }
    if (field <= 192)
      bits = field - 128;
    else if (field <= 208)
      bits = static_cast<u32>(-static_cast<int>(field - 192));
    else if (field >= 240 && field <= 247)
      bits = kFloats[field - 240];
    else if (field == 255)
      bits = literal;
    else
      return false;
    return true;
  };
  for (const Inst& inst : program) {
    const u32 w = inst.raw[0], w1 = inst.raw[1];
    switch (inst.enc) {
      case Enc::kSopp:
        if (inst.opcode == 0x01)  // s_endpgm
          return true;
        if (inst.opcode >= 0x02 && inst.opcode <= 0x09)
          return false;  // a branch: more than one stream
        break;
      case Enc::kSopk: {
        const u32 sdst = (w >> 16) & 0x7F;
        if (sdst < 128)
          sgpr[sdst] = inst.opcode == 0x00  // s_movk_i32
                           ? Value{Value::kConst, static_cast<u32>(
                                                      static_cast<i16>(w))}
                           : Value{};
        break;
      }
      case Enc::kSop1: {
        const u32 sdst = (w >> 16) & 0x7F;
        Value v;
        if (inst.opcode == 0x03 && constant(w & 0xFF, inst.literal, v.bits))
          v.kind = Value::kConst;  // s_mov_b32
        if (sdst < 128)
          sgpr[sdst] = v;
        break;
      }
      case Enc::kSop2: {
        const u32 sdst = (w >> 16) & 0x7F;
        if (sdst < 128)
          sgpr[sdst] = {};
        break;
      }
      case Enc::kSmrd: {
        const u32 sdst = (w >> 15) & 0x7F;
        for (u32 i = 0; i < SmrdDwordCount(inst.opcode) && sdst + i < 128; i++)
          sgpr[sdst + i] = {};
        break;
      }
      case Enc::kMubuf: {
        u32 soffset = 0;
        if (!RingDwordOp(inst, kBufferLoadDword) ||
            !constant((w1 >> 24) & 0xFF, 0, soffset))
          return false;
        const u32 offset = soffset + (w & 0xFFF);
        if (offset % comp_bytes)
          return false;
        vgpr[(w1 >> 8) & 0xFF] = {Value::kComp, offset / comp_bytes};
        break;
      }
      case Enc::kVop1: {
        const u32 vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
        Value v;
        if (inst.opcode == 0x01) {  // v_mov_b32
          if (src0 >= 256)
            v = vgpr[src0 - 256];
          else if (constant(src0, inst.literal, v.bits))
            v.kind = Value::kConst;
        }
        vgpr[vdst] = v;
        break;
      }
      case Enc::kVop2:
        vgpr[(w >> 17) & 0xFF] = {};
        break;
      case Enc::kVop3:
        vgpr[w & 0xFF] = {};
        break;
      case Enc::kExp: {
        const u32 en = w & 0xF, target = (w >> 4) & 0x3F;
        if ((w >> 10) & 1)
          return false;  // COMPR
        const bool pos = target >= kExpPos0 && target <= kExpPos0 + 3;
        const bool param = target >= kExpParam0 && target < kExpParam0 + 32;
        if (!pos && !param)
          break;
        for (u32 c = 0; c < 4; c++) {
          if (!((en >> c) & 1))
            continue;
          const Value& v = vgpr[(w1 >> (8 * c)) & 0xFF];
          if (v.kind == Value::kUnknown)
            return false;
          out.push_back({target, c,
                         v.kind == Value::kComp ? static_cast<int>(v.bits) : -1,
                         v.bits});
        }
        break;
      }
      default:
        return false;
    }
  }
  return true;
}

void EmitEsRingStore(Translator& t, const Inst& inst, StageContext& sc) {
  if (!RingDwordOp(inst, kBufferStoreDword)) {
    WarnUnsupported("mubuf.es-ring", (inst.raw[0] >> 18) & 0x7F, inst.raw[0],
                    inst.raw[1]);
    return;
  }
  const Id comp = t.UMin(t.Shr(RingAddress(t, inst), t.U32(2)),
                         t.U32(sc.es_ring_vec4s * 4 - 1));
  const Id ptr = t.m.AccessChain(
      t.m.TypePointer(spv::StorageClass::Output, t.t_u), sc.es_ring,
      {t.Shr(comp, t.U32(2)), t.And(comp, t.U32(3))});
  t.m.Store(ptr, t.Vg((inst.raw[1] >> 8) & 0xFF));
}

void EmitGsRingAccess(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 vdata = (inst.raw[1] >> 8) & 0xFF;
  const Id address = RingAddress(t, inst);
  if (RingDwordOp(inst, kBufferLoadDword)) {
    const Id vertex = t.UMin(t.And(t.Shr(address, t.U32(2)), t.U32(63)),
                             t.U32(sc.gs_input_verts - 1));
    const Id comp = t.UMin(t.Shr(address, t.U32(8)),
                           t.U32(sc.es_ring_vec4s * 4 - 1));
    const Id ptr = t.m.AccessChain(
        t.m.TypePointer(spv::StorageClass::Input, t.t_u), sc.es_ring,
        {vertex, t.Shr(comp, t.U32(2)), t.And(comp, t.U32(3))});
    t.SetVg(vdata, t.m.Load(t.t_u, ptr));
  } else if (RingDwordOp(inst, kBufferStoreDword)) {
    const Id ptr = t.m.AccessChain(
        t.p_priv_u, sc.gsvs,
        {t.UMin(t.Shr(address, t.U32(2)), t.U32(sc.gsvs_dwords - 1))});
    t.StorePrivate(ptr, t.SelectB(t.LaneActive(t.Exec()), t.Vg(vdata),
                                  t.m.Load(t.t_u, ptr)));
  } else {
    WarnUnsupported("mubuf.gs-ring", (inst.raw[0] >> 18) & 0x7F, inst.raw[0],
                    inst.raw[1]);
  }
}

namespace {

// Write GSVS vertex `vertex` to the stage outputs the copy shader would have
// exported it through.
void WriteGsVertex(Translator& t, StageContext& sc, Id vertex) {
  Id pos[4] = {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(1.f)};
  bool wrote_pos = false;
  std::unordered_map<u32, std::array<Id, 4>> params;
  Id layer = 0;
  for (const GsCopyExport& e : *sc.gs_exports) {
    Id value;
    if (e.comp < 0) {
      value = t.U32(e.bits);
    } else {
      const Id index = t.UMin(
          t.Add(t.U32(static_cast<u32>(e.comp) * sc.gs_max_vert_out), vertex),
          t.U32(sc.gsvs_dwords - 1));
      value = t.m.Load(t.t_u, t.m.AccessChain(t.p_priv_u, sc.gsvs, {index}));
    }
    if (e.target == kExpPos0) {
      pos[e.chan] = t.m.Bitcast(t.t_f, value);
      wrote_pos = true;
    } else if (e.target == kExpPos1 && e.chan == 2) {
      layer = value;
    } else if (e.target >= kExpParam0) {
      auto [it, fresh] = params.try_emplace(e.target - kExpParam0);
      if (fresh)
        it->second.fill(t.F32(0.f));
      it->second[e.chan] = t.m.Bitcast(t.t_f, value);
    }
  }
  if (wrote_pos) {
    if (sc.gl_clip)
      pos[2] = t.FMul(t.FAdd(pos[2], pos[3]), t.F32(0.5f));
    t.m.Store(sc.pos_out, t.m.CompositeConstruct(t.t_v4, {pos[0], pos[1],
                                                          pos[2], pos[3]}));
  }
  for (const auto& [p, c] : params) {
    sc.max_param = std::max(sc.max_param, p + 1);
    t.m.Store(VsParamOut(t, sc, p),
              t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  }
  if (layer) {
    if (!sc.layer_out) {
      sc.layer_out = t.m.Variable(
          t.m.TypePointer(spv::StorageClass::Output, t.t_i),
          spv::StorageClass::Output);
      t.m.Decorate(sc.layer_out, spv::Decoration::BuiltIn,
                   {static_cast<u32>(spv::BuiltIn::Layer)});
      sc.iface->push_back(sc.layer_out);
      sc.r->writes_layer = true;
    }
    t.m.Store(sc.layer_out, t.m.Bitcast(t.t_i, layer));
  }
}

}  // namespace

void EmitGsMessage(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 simm = inst.raw[0] & 0xFFFF;
  const u32 msg = simm & 0xF, op = (simm >> 4) & 3, stream = (simm >> 8) & 3;
  if (msg == 3)
    return;  // GS_DONE
  if (msg != 2 || !sc.gs_exports) {
    WarnUnsupported("sendmsg", simm);
    return;
  }
  if (stream != 0 || op == 0)
    return;  // only stream 0 reaches the rasterizer
  // The hardware counts a vertex for each lane EXEC has on at the message.
  const Id emitted = t.m.Load(t.t_u, sc.gs_emitted);
  Id go = t.LaneActive(t.Exec());
  if (op & 2)
    go = t.LAnd(go, t.Ult(emitted, t.U32(sc.gs_max_vert_out)));
  const Id body = t.m.NewBlock(), merge = t.m.NewBlock();
  t.m.SelectionMerge(merge);
  t.m.BranchConditional(go, body, merge);
  t.m.OpenBlock(body);
  if (op & 2) {
    WriteGsVertex(t, sc, emitted);
    t.m.EmitVoid(spv::Op::OpEmitVertex, {});
    t.m.Store(sc.gs_emitted, t.Add(emitted, t.U32(1)));
  }
  if (op & 1)
    t.m.EmitVoid(spv::Op::OpEndPrimitive, {});
  t.m.Branch(merge);
  t.m.OpenBlock(merge);
}

}  // namespace gpu::gcn
#endif
