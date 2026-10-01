#include "gpu/gcn/gcn_translate.h"

#ifdef DELTA_HAVE_SPIRV_BACKEND
#include <spirv/unified1/GLSL.std.450.h>
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"

namespace gpu::gcn {

base::Vector<u32> BuildImageTilingShader() {
  using namespace spirv;
  using spv::Op;
  using spv::StorageClass;
  Module m;
  const Id u = m.TypeInt();
  const Id array = m.TypeRuntimeArray(u);
  m.Decorate(array, spv::Decoration::ArrayStride, {4});
  const Id block = m.TypeStruct({array});
  m.Decorate(block, spv::Decoration::Block);
  m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
  const Id ptr = m.TypePointer(StorageClass::StorageBuffer, u);
  Id buffers[3];
  for (u32 i = 0; i < 3; i++) {
    buffers[i] = m.Variable(m.TypePointer(StorageClass::StorageBuffer, block),
                            StorageClass::StorageBuffer);
    m.Decorate(buffers[i], spv::Decoration::DescriptorSet, {0});
    m.Decorate(buffers[i], spv::Decoration::Binding, {i});
  }
  const Id pc_type =
      m.TypeStruct(base::Vector<Id>(sizeof(ImageTilingParams) / 4, u));
  m.Decorate(pc_type, spv::Decoration::Block);
  for (u32 i = 0; i < sizeof(ImageTilingParams) / 4; i++)
    m.MemberDecorate(pc_type, i, spv::Decoration::Offset, {i * 4});
  const Id pc = m.Variable(m.TypePointer(StorageClass::PushConstant, pc_type),
                           StorageClass::PushConstant);
  const Id vec = m.TypeVec(u, 3);
  const Id gid =
      m.Variable(m.TypePointer(StorageClass::Input, vec), StorageClass::Input);
  m.Decorate(gid, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::GlobalInvocationId)});
  const Id fn = m.BeginFunction(m.TypeVoid(), m.TypeFunction(m.TypeVoid()));
  m.EntryPoint(spv::ExecutionModel::GLCompute, fn, "main", {gid});
  m.ExecMode(fn, spv::ExecutionMode::LocalSize, {64, 1, 1});
  const auto c = [&](u32 v) { return m.ConstU32(v); };
  const auto op = [&](Op code, Id a, Id b) { return m.Emit(code, u, {a, b}); };
  const auto add = [&](Id a, Id b) { return op(Op::OpIAdd, a, b); };
  const auto mul = [&](Id a, Id b) { return op(Op::OpIMul, a, b); };
  const auto param = [&](u32 i) {
    return m.Load(u, m.AccessChain(m.TypePointer(StorageClass::PushConstant, u),
                                   pc, {c(i)}));
  };
  const auto at = [&](u32 binding, Id index) {
    return m.AccessChain(ptr, buffers[binding], {c(0), index});
  };
  const Id global = m.Load(vec, gid);
  const Id xw = m.CompositeExtract(u, global, 0);
  const Id y = m.CompositeExtract(u, global, 1);
  const Id z = m.CompositeExtract(u, global, 2);
  const Id width = param(0), words = param(2);
  const Id valid =
      m.Emit(Op::OpULessThan, m.TypeBool(), {xw, mul(width, words)});
  const Id body = m.NewBlock(), end = m.NewBlock();
  m.SelectionMerge(end);
  m.BranchConditional(valid, body, end);
  m.OpenBlock(body);
  const Id x = op(Op::OpUDiv, xw, words);
  const Id component = op(Op::OpUMod, xw, words);
  const Id table = param(3), mask = param(4);
  const Id xt = m.Load(u, at(2, add(table, x)));
  const Id yt = m.Load(u, at(2, add(add(table, width), y)));
  const Id zt = m.Load(u, at(2, add(add(add(table, width), param(1)), z)));
  const Id low =
      op(Op::OpBitwiseAnd,
         op(Op::OpBitwiseXor, op(Op::OpBitwiseXor, xt, yt), zt), mask);
  const Id high_mask = m.Emit(Op::OpNot, u, {mask});
  const Id high = add(op(Op::OpBitwiseAnd, xt, high_mask),
                      op(Op::OpBitwiseAnd, yt, high_mask));
  const Id tiled_byte = add(add(param(5), mul(z, param(6))),
                            add(add(high, low), mul(component, c(4))));
  const Id tiled = op(Op::OpShiftRightLogical, tiled_byte, c(2));
  const Id shift = mul(op(Op::OpBitwiseAnd, tiled_byte, c(3)), c(8));
  const Id elem = param(11);
  const Id narrow = m.Emit(Op::OpULessThan, m.TypeBool(), {elem, c(4)});
  const Id bits = m.Emit(Op::OpSelect, u, {narrow, mul(elem, c(8)), c(32)});
  const Id value_mask =
      op(Op::OpShiftRightLogical, c(UINT32_MAX), op(Op::OpISub, c(32), bits));
  const Id linear_dw = add(add(param(7), mul(z, param(9))),
                           add(mul(y, mul(param(8), words)), xw));
  // Packed narrow texels on the linear side (param 12): byte addressing with
  // the offset and stride given in bytes, and a lane shift inside the dword.
  const Id is_packed = m.Emit(Op::OpINotEqual, m.TypeBool(), {param(12), c(0)});
  const Id lin_byte =
      add(add(param(7), mul(z, param(9))), mul(add(mul(y, param(8)), x), elem));
  const Id linear = m.Emit(
      Op::OpSelect, u,
      {is_packed, op(Op::OpShiftRightLogical, lin_byte, c(2)), linear_dw});
  const Id lshift = m.Emit(
      Op::OpSelect, u,
      {is_packed, mul(op(Op::OpBitwiseAnd, lin_byte, c(3)), c(8)), c(0)});
  // R11G11B10F from three floats, rounded to nearest with ties away, as the
  // host's PackUnsignedFloat: NaN keeps a mantissa bit, negatives are zero.
  const Id t_bool = m.TypeBool();
  const auto sel = [&](Id cond, Id a, Id b) {
    return m.Emit(Op::OpSelect, u, {cond, a, b});
  };
  const auto eq = [&](Id a, Id b) { return m.Emit(Op::OpIEqual, t_bool, {a, b}); };
  const auto ult = [&](Id a, Id b) {
    return m.Emit(Op::OpULessThan, t_bool, {a, b});
  };
  const auto pack_unsigned_float = [&](Id bits, u32 mantissa_bits) {
    const u32 drop = 23 - mantissa_bits;
    const Id inf = c(0x1Fu << mantissa_bits);
    const Id exp32 =
        op(Op::OpBitwiseAnd, op(Op::OpShiftRightLogical, bits, c(23)), c(0xFF));
    const Id mant23 = op(Op::OpBitwiseAnd, bits, c(0x7FFFFF));
    const Id negative = m.Emit(Op::OpINotEqual, t_bool,
                               {op(Op::OpShiftRightLogical, bits, c(31)), c(0)});
    const Id special =
        sel(m.Emit(Op::OpINotEqual, t_bool, {mant23, c(0)}),
            op(Op::OpBitwiseOr, inf, c(1)), sel(negative, c(0), inf));
    const Id normal = m.ExtInst(
        u, GLSLstd450UMin,
        {op(Op::OpShiftRightLogical,
            add(op(Op::OpBitwiseOr,
                   op(Op::OpShiftLeftLogical, op(Op::OpISub, exp32, c(112)),
                      c(23)),
                   mant23),
                c(1u << (drop - 1))),
            c(drop)),
         inf});
    const Id shift = op(Op::OpISub, c(drop + 113),
                        m.ExtInst(u, GLSLstd450UMax, {exp32, c(1)}));
    const Id held = m.ExtInst(u, GLSLstd450UMin, {shift, c(31)});
    const Id mant = sel(eq(exp32, c(0)), mant23,
                        op(Op::OpBitwiseOr, mant23, c(0x800000)));
    const Id denormal = sel(
        ult(shift, c(32)),
        op(Op::OpShiftRightLogical,
           add(mant, op(Op::OpShiftLeftLogical, c(1),
                        op(Op::OpISub, held, c(1)))),
           held),
        c(0));
    return sel(eq(exp32, c(0xFF)), special,
               sel(negative, c(0),
                   sel(m.Emit(Op::OpUGreaterThan, t_bool, {exp32, c(112)}),
                       normal, denormal)));
  };
  const Id is_pack11 = m.Emit(Op::OpINotEqual, t_bool, {param(14), c(0)});
  // Every lane loads: a lane not packing reads its own dword three times.
  const Id texel4 = add(add(param(7), mul(z, param(9))),
                        mul(add(mul(y, param(8)), xw), c(4)));
  const Id detile = m.NewBlock(), retile = m.NewBlock(), copied = m.NewBlock();
  const Id direction = m.Emit(Op::OpINotEqual, m.TypeBool(), {param(10), c(0)});
  const Id f = m.TypeFloat();
  const Id is_depth16 =
      m.Emit(Op::OpINotEqual, m.TypeBool(), {param(13), c(0)});
  m.SelectionMerge(copied);
  m.BranchConditional(direction, detile, retile);
  m.OpenBlock(detile);
  const Id raw_texel = op(
      Op::OpBitwiseAnd,
      op(Op::OpShiftRightLogical, m.Load(u, at(0, tiled)), shift), value_mask);
  const Id unorm_to_float =
      m.Emit(Op::OpBitcast, u,
             {m.Emit(Op::OpFMul, f,
                     {m.Emit(Op::OpConvertUToF, f, {raw_texel}),
                      m.ConstF32(1.0f / 65535.0f)})});
  const Id texel =
      m.Emit(Op::OpSelect, u, {is_depth16, unorm_to_float, raw_texel});
  const Id pk_store = m.NewBlock(), plain_store = m.NewBlock(),
           det_done = m.NewBlock();
  // RGBA32F out of R11G11B10F, as the host's UnpackUnsignedFloat.
  const auto unpack_unsigned_float = [&](Id packed, u32 mantissa_bits) {
    const Id mant = op(Op::OpBitwiseAnd, packed, c((1u << mantissa_bits) - 1));
    const Id exponent = op(Op::OpBitwiseAnd,
                           op(Op::OpShiftRightLogical, packed, c(mantissa_bits)),
                           c(0x1F));
    const Id denormal = m.Emit(
        Op::OpBitcast, u,
        {m.Emit(Op::OpFMul, f,
                {m.Emit(Op::OpConvertUToF, f, {mant}),
                 m.Emit(Op::OpBitcast, f,
                        {c((127u - 14u - mantissa_bits) << 23)})})});
    const Id special = sel(m.Emit(Op::OpINotEqual, t_bool, {mant, c(0)}),
                           c(0x7FC00000), c(0x7F800000));
    const Id normal = op(
        Op::OpBitwiseOr,
        op(Op::OpShiftLeftLogical, add(exponent, c(112)), c(23)),
        op(Op::OpShiftLeftLogical, mant, c(23 - mantissa_bits)));
    return sel(eq(exponent, c(0)), denormal,
               sel(eq(exponent, c(0x1F)), special, normal));
  };
  const Id unpack_store = m.NewBlock(), narrow_or_plain = m.NewBlock();
  m.SelectionMerge(det_done);
  m.BranchConditional(is_pack11, unpack_store, narrow_or_plain);
  m.OpenBlock(unpack_store);
  {
    const Id word = m.Load(u, at(0, tiled));
    m.Store(at(1, texel4), unpack_unsigned_float(word, 6));
    m.Store(at(1, add(texel4, c(1))),
            unpack_unsigned_float(op(Op::OpShiftRightLogical, word, c(11)), 6));
    m.Store(at(1, add(texel4, c(2))),
            unpack_unsigned_float(op(Op::OpShiftRightLogical, word, c(22)), 5));
    m.Store(at(1, add(texel4, c(3))), c(0x3F800000));
  }
  m.Branch(det_done);
  m.OpenBlock(narrow_or_plain);
  const Id narrow_done = m.NewBlock();
  m.SelectionMerge(narrow_done);
  m.BranchConditional(is_packed, pk_store, plain_store);
  m.OpenBlock(pk_store);
  m.Emit(
      Op::OpAtomicAnd, u,
      {at(1, linear), c(1), c(0),
       m.Emit(Op::OpNot, u, {op(Op::OpShiftLeftLogical, value_mask, lshift)})});
  m.Emit(
      Op::OpAtomicOr, u,
      {at(1, linear), c(1), c(0), op(Op::OpShiftLeftLogical, texel, lshift)});
  m.Branch(narrow_done);
  m.OpenBlock(plain_store);
  m.Store(at(1, linear), texel);
  m.Branch(narrow_done);
  m.OpenBlock(narrow_done);
  m.Branch(det_done);
  m.OpenBlock(det_done);
  m.Branch(copied);
  m.OpenBlock(retile);
  const Id raw_value =
      op(Op::OpShiftRightLogical, m.Load(u, at(1, linear)), lshift);
  const Id depth = m.Emit(Op::OpBitcast, f, {raw_value});
  const Id clamped =
      m.ExtInst(f, GLSLstd450FClamp, {depth, m.ConstF32(0.f), m.ConstF32(1.f)});
  const Id float_to_unorm =
      m.Emit(Op::OpConvertFToU, u,
             {m.Emit(Op::OpFAdd, f,
                     {m.Emit(Op::OpFMul, f, {clamped, m.ConstF32(65535.f)}),
                      m.ConstF32(0.5f)})});
  const auto channel = [&](u32 k) {
    return m.Load(u, at(1, sel(is_pack11, add(texel4, c(k)), linear)));
  };
  const Id pack11 = op(
      Op::OpBitwiseOr,
      op(Op::OpBitwiseOr, pack_unsigned_float(channel(0), 6),
         op(Op::OpShiftLeftLogical, pack_unsigned_float(channel(1), 6), c(11))),
      op(Op::OpShiftLeftLogical, pack_unsigned_float(channel(2), 5), c(22)));
  const Id value = sel(
      is_pack11, pack11,
      op(Op::OpBitwiseAnd,
         m.Emit(Op::OpSelect, u, {is_depth16, float_to_unorm, raw_value}),
         value_mask));
  const Id packed = m.NewBlock(), direct = m.NewBlock(), stored = m.NewBlock();
  m.SelectionMerge(stored);
  m.BranchConditional(narrow, packed, direct);
  m.OpenBlock(packed);
  // Neighbouring narrow texels share a dword: clear this texel's lanes, then
  // set them. Bytes no texel maps to (padding) are never touched.
  m.Emit(
      Op::OpAtomicAnd, u,
      {at(0, tiled), c(1), c(0),
       m.Emit(Op::OpNot, u, {op(Op::OpShiftLeftLogical, value_mask, shift)})});
  m.Emit(Op::OpAtomicOr, u,
         {at(0, tiled), c(1), c(0),
          op(Op::OpShiftLeftLogical, op(Op::OpBitwiseAnd, value, value_mask),
             shift)});
  m.Branch(stored);
  m.OpenBlock(direct);
  m.Store(at(0, tiled), value);
  m.Branch(stored);
  m.OpenBlock(stored);
  m.Branch(copied);
  m.OpenBlock(copied);
  m.Branch(end);
  m.OpenBlock(end);
  m.ReturnVoid();
  m.EndFunction();
  auto code = m.Assemble();
  base::String error;
  if (!Validate(code, &error)) {
    BASE_LOGI("gpuvk", "tiling shader invalid: {}", error.c_str());
    return {};
  }
  return code;
}

}  // namespace gpu::gcn
#endif
