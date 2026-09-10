/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 IMAGE_BVH[64]_INTERSECT_RAY: one node test, not a traversal.
 * The guest continues traversal using the four returned child pointers.
 *
 * Box/triangle arithmetic adapted from AMD GPURT IntersectCommon.hlsl,
 * revision 7b226d48b46b7e92fec3b9ecc5712e5bf2bf3dd9. Node vertex mapping
 * follows AMD Radeon Raytracing Analyzer's TriangleNode::GetTriangle,
 * revision 28d0a0e5c93d46006cb4c10316aab97bfcf5b2d6.
 *
 * Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND
#include <array>
#include <limits>

#include "gpu/ps5/rdna/rdna_emit.h"

namespace gpu::rdna {
using gpu::gcn::Id;
using gpu::gcn::Translator;
namespace {
using Vec3 = std::array<Id, 3>;
Id FLe(Translator& t, Id a, Id b) {
  return t.m.Emit(spv::Op::OpFOrdLessThanEqual, t.t_bool, {a, b});
}
Vec3 Sub(Translator& t, const Vec3& a, const Vec3& b) {
  return {t.FSub(a[0], b[0]), t.FSub(a[1], b[1]), t.FSub(a[2], b[2])};
}
Vec3 Cross(Translator& t, const Vec3& a, const Vec3& b) {
  Vec3 out;
  for (u32 i = 0; i < 3; i++)
    out[i] = t.FSub(t.FMul(a[(i + 1) % 3], b[(i + 2) % 3]),
                    t.FMul(a[(i + 2) % 3], b[(i + 1) % 3]));
  return out;
}
Id Dot(Translator& t, const Vec3& a, const Vec3& b) {
  return t.FAdd(t.FAdd(t.FMul(a[0], b[0]), t.FMul(a[1], b[1])),
                t.FMul(a[2], b[2]));
}
}  // namespace

void EmitBvh(Translator& t,
             const gpu::gcn::Inst& inst,
             gpu::gcn::StageContext& sc) {
  const int binding = gpu::gcn::CsBindingFor(sc, inst.pc);
  if (binding < 0)
    return;
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 nsa = (w >> 1) & 3, srsrc = ((w1 >> 16) & 31) * 4;
  const u32 dest = (w1 >> 8) & 255, wide = inst.opcode == 0xe7;
  const bool a16 = (w1 >> 30) & 1;
  // Read ALL addresses before overwriting VDATA: Astro aliases vaddr/vdata.
  std::array<Id, 12> a{};
  const u32 count = (a16 ? 8 : 11) + wide;
  for (u32 i = 0; i < count; i++) {
    const u32 reg =
        nsa && i ? (inst.raw[2 + (i - 1) / 4] >> (((i - 1) % 4) * 8)) & 255
                 : (w1 & 255) + i;
    a[i] = t.Vg(reg);
  }
  const auto as_float = [&](Id v) { return t.m.Bitcast(t.t_f, v); };
  const Id extent = as_float(a[1 + wide]);
  Vec3 origin, direction, inverse;
  for (u32 i = 0; i < 3; i++)
    origin[i] = as_float(a[2 + wide + i]);
  std::array<Id, 6> directions;
  if (a16) {
    for (u32 i = 0; i < 3; i++) {
      const Id pair = t.m.ExtInst(t.m.TypeVec(t.t_f, 2),
                                  GLSLstd450UnpackHalf2x16, {a[5 + wide + i]});
      for (u32 j = 0; j < 2; j++)
        directions[2 * i + j] =
            t.m.Emit(spv::Op::OpCompositeExtract, t.t_f, {pair, j});
    }
  } else {
    for (u32 i = 0; i < 6; i++)
      directions[i] = as_float(a[5 + wide + i]);
  }
  for (u32 i = 0; i < 3; i++) {
    direction[i] = directions[i];
    inverse[i] = directions[3 + i];
  }
  const Id type = t.And(a[0], t.U32(7));
  const Id box16 = t.Eq(type, t.U32(4)), box32 = t.Eq(type, t.U32(5));
  const Id triangle = t.Ult(type, t.U32(4));
  const Id desc1 = t.Sg(srsrc + 1), desc2 = t.Sg(srsrc + 2);
  const Id desc3 = t.Sg(srsrc + 3);
  const Id index = t.Shr(a[0], t.U32(3));
  const Id words = t.SelectB(box32, t.U32(32), t.U32(16));
  const bool runtime = sc.cs_runtime_resources.count(binding);
  // Runtime descriptors supply their node extent; the address map separately
  // checks that the entire node has readable backing before any load.
  const Id bound = runtime ? t.U32(0xffffffff) : gpu::gcn::CsSsboBound(t, sc, binding);
  Id physical = 0;
  if (runtime) {
    const Id wide_type = t.m.TypeInt(64, false);
    const Id offset64 = t.m.Emit(spv::Op::OpShiftLeftLogical, wide_type,
        {t.m.Emit(spv::Op::OpUConvert, wide_type, {index}), t.U32(6)});
    physical = gpu::gcn::CsGuestAddress(t, sc,
        t.m.Emit(spv::Op::OpIAdd, wide_type,
                  {gpu::gcn::CsGuestBase(t, sc, binding), offset64}), t.Mul(words, t.U32(4)));
  }
  // Check the entire node before any SSBO load. Generic SSBO loads clamp,
  // which would turn a bad pointer into fabricated geometry here.
  Id valid = t.LAnd(t.Uge(bound, words),
                    t.Ule(index, t.Shr(t.Sub(bound, words), t.U32(4))));
  valid = t.LAnd(valid, t.Ule(t.Add(index, t.SelectB(box32, t.U32(1), t.U32(0))), desc2));
  valid = t.LAnd(valid, t.Eq(t.Shr(desc3, t.U32(28)), t.U32(8)));
  valid = t.LAnd(valid, t.Ult(type, t.U32(6)));
  if (wide)
    valid = t.LAnd(valid, t.IsZero(a[1]));
  if (runtime)
    valid = t.LAnd(valid, t.m.Emit(spv::Op::OpINotEqual, t.t_bool,
        {physical, t.m.ConstNull(t.m.TypeInt(64, false))}));
  const Id invalid = t.U32(0xffffffff);
  for (u32 i = 0; i < 4; i++)
    t.SetVg(dest + i, invalid);
  const Id node_block = t.m.NewBlock(), done = t.m.NewBlock();
  t.m.SelectionMerge(done);
  t.m.BranchConditional(valid, node_block, done);
  t.m.OpenBlock(node_block);
  const Id offset = t.Shl(index, t.U32(4));
  const auto load = [&](Id word) {
    if (runtime) {
      const Id wide_type = t.m.TypeInt(64, false);
      const Id bytes = t.m.Emit(spv::Op::OpUConvert, wide_type, {t.Mul(word, t.U32(4))});
      return gpu::gcn::CsPhysicalLoad(t, t.m.Emit(spv::Op::OpIAdd, wide_type, {physical, bytes}));
    }
    return gpu::gcn::CsSsboLoad(t, sc, binding, t.Add(offset, word));
  };
  const Id tri_block = t.m.NewBlock(), box_block = t.m.NewBlock();
  const Id node_done = t.m.NewBlock();
  t.m.SelectionMerge(node_done);
  t.m.BranchConditional(triangle, tri_block, box_block);
  t.m.OpenBlock(tri_block);
  // Four compressed triangles share five vertex slots in a 64-byte node.
  const Id mapping = t.SelectB(
      t.Eq(type, t.U32(0)), t.U32(0x210),
      t.SelectB(t.Eq(type, t.U32(1)), t.U32(0x231),
                t.SelectB(t.Eq(type, t.U32(2)), t.U32(0x432), t.U32(0x042))));
  std::array<Vec3, 3> vertices;
  for (u32 v = 0; v < 3; v++) {
    const Id slot =
        t.Mul(t.And(t.Shr(mapping, t.U32(v * 4)), t.U32(15)), t.U32(3));
    for (u32 c = 0; c < 3; c++)
      vertices[v][c] = as_float(load(t.Add(slot, t.U32(c))));
  }
  const Vec3 e1 = Sub(t, vertices[1], vertices[0]);
  const Vec3 e2 = Sub(t, vertices[2], vertices[0]);
  const Vec3 e3 = Sub(t, origin, vertices[0]);
  const Vec3 s1 = Cross(t, direction, e2), s2 = Cross(t, e3, e1);
  const Id tn = Dot(t, e2, s2), den = Dot(t, s1, e1);
  const Id in = Dot(t, e3, s1), jn = Dot(t, direction, s2);
  const Id distance = t.FDiv(tn, den), u = t.FDiv(in, den), v = t.FDiv(jn, den);
  const Id zero = t.F32(0), one = t.F32(1);
  // Ordered tests also reject zero-area/NaN triangles. Triangle distance is
  // deliberately not clipped to extent: the instruction clips boxes only.
  Id hit = t.LAnd(FLe(t, zero, u), FLe(t, u, one));
  hit = t.LAnd(hit, t.LAnd(FLe(t, zero, v), FLe(t, t.FAdd(u, v), one)));
  hit = t.LAnd(hit, FLe(t, zero, distance));
  hit = t.LAnd(hit, t.m.Emit(spv::Op::OpFOrdNotEqual, t.t_bool, {den, zero}));
  const Id infinity = t.F32(std::numeric_limits<float>::infinity());
  const Id result_den = t.SelectF(hit, den, one);
  const Id tri_id = load(t.U32(15));
  const Id selectors = t.Shr(tri_id, t.Mul(type, t.U32(8)));
  const Id bary0 = t.FSub(t.FSub(result_den, in), jn);
  const auto bary = [&](Id selector) {
    return t.SelectF(t.Eq(selector, t.U32(0)), bary0,
                     t.SelectF(t.Eq(selector, t.U32(1)), in, jn));
  };
  const Id bary_mode = t.IsNonZero(t.And(desc3, t.U32(1u << 24)));
  t.SetVgF(dest, t.SelectF(hit, tn, infinity));
  t.SetVgF(dest + 1, result_den);
  t.SetVg(
      dest + 2,
      t.SelectB(bary_mode, t.m.Bitcast(t.t_u, bary(t.And(selectors, t.U32(3)))),
                tri_id));
  t.SetVg(
      dest + 3,
      t.SelectB(
          bary_mode,
          t.m.Bitcast(t.t_u, bary(t.And(t.Shr(selectors, t.U32(2)), t.U32(3)))),
          t.SelectB(hit, t.U32(1), t.U32(0))));
  t.m.Branch(node_done);
  t.m.OpenBlock(box_block);
  std::array<Id, 4> children, distances;
  const Id grow =
      t.FAdd(one, t.FMul(t.m.Emit(spv::Op::OpConvertUToF, t.t_f,
                                  {t.And(t.Shr(desc1, t.U32(23)), t.U32(255))}),
                         t.F32(0x1p-24f)));
  for (u32 child = 0; child < 4; child++) {
    Id near = zero, far = extent, box_valid = FLe(t, zero, extent);
    for (u32 axis = 0; axis < 3; axis++)
      for (Id value : {origin[axis], inverse[axis]})
        box_valid = t.LAnd(box_valid, t.m.Emit(spv::Op::OpFOrdEqual, t.t_bool,
                                               {value, value}));
    for (u32 axis = 0; axis < 3; axis++) {
      std::array<Id, 2> limits;
      for (u32 side = 0; side < 2; side++) {
        const u32 component = axis + 3 * side;
        const Id word =
            load(t.SelectB(box16, t.U32(4 + child * 3 + component / 2),
                           t.U32(4 + child * 6 + component)));
        const Id pair = t.m.ExtInst(t.m.TypeVec(t.t_f, 2),
                                    GLSLstd450UnpackHalf2x16, {word});
        const Id half =
            t.m.Emit(spv::Op::OpCompositeExtract, t.t_f, {pair, component % 2});
        limits[side] = t.SelectF(box16, half, as_float(word));
      }
      box_valid = t.LAnd(box_valid, FLe(t, limits[0], limits[1]));
      // Box tests consume the supplied inverse direction; the direction
      // operands are used only by the triangle test.
      const Id parallel = t.IsInf(inverse[axis]);
      const Id inside = t.LAnd(FLe(t, limits[0], origin[axis]),
                               FLe(t, origin[axis], limits[1]));
      box_valid = t.LAnd(
          box_valid,
          t.m.Emit(
              spv::Op::OpLogicalOr, t.t_bool,
              {t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {parallel}), inside}));
      const Id lo = t.FMul(t.FSub(limits[0], origin[axis]), inverse[axis]);
      const Id hi = t.FMul(t.FSub(limits[1], origin[axis]), inverse[axis]);
      // Parallel rays on a slab boundary must not produce 0*infinity NaNs.
      const Id axis_near =
          t.SelectF(parallel, t.FNeg(infinity),
                    t.m.ExtInst(t.t_f, GLSLstd450FMin, {lo, hi}));
      const Id axis_far = t.SelectF(
          parallel, infinity, t.m.ExtInst(t.t_f, GLSLstd450FMax, {lo, hi}));
      near = t.m.ExtInst(t.t_f, GLSLstd450FMax, {near, axis_near});
      far = t.m.ExtInst(t.t_f, GLSLstd450FMin, {far, axis_far});
    }
    box_valid = t.LAnd(box_valid, FLe(t, near, t.FMul(far, grow)));
    children[child] = t.SelectB(box_valid, load(t.U32(child)), invalid);
    distances[child] = near;
  }
  const Id sort = t.IsNonZero(t.And(desc1, t.U32(0x80000000)));
  constexpr u32 pairs[5][2] = {{0, 2}, {1, 3}, {0, 1}, {2, 3}, {1, 2}};
  for (const auto& pair : pairs) {
    const u32 a = pair[0], b = pair[1];
    Id swap = t.LAnd(t.IsNonZero(t.Not(children[b])),
                     t.FLt(distances[b], distances[a]));
    swap = t.m.Emit(spv::Op::OpLogicalOr, t.t_bool,
                    {swap, t.Eq(children[a], invalid)});
    swap = t.LAnd(sort, swap);
    const Id ca = children[a], da = distances[a];
    children[a] = t.SelectB(swap, children[b], ca);
    children[b] = t.SelectB(swap, ca, children[b]);
    distances[a] = t.SelectF(swap, distances[b], da);
    distances[b] = t.SelectF(swap, da, distances[b]);
  }
  for (u32 i = 0; i < 4; i++)
    t.SetVg(dest + i, children[i]);
  t.m.Branch(node_done);
  t.m.OpenBlock(node_done);
  t.m.Branch(done);
  t.m.OpenBlock(done);
}
}  // namespace gpu::rdna
#endif
