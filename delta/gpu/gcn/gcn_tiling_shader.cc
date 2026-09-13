#include "gpu/gcn/gcn_translate.h"

#ifdef DELTA_HAVE_SPIRV_BACKEND
#include <base/logging.h>
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"

namespace gpu::gcn {

std::vector<u32> BuildImageTilingShader() {
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
      m.TypeStruct(std::vector<Id>(sizeof(ImageTilingParams) / 4, u));
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
  const Id lin_byte = add(add(param(7), mul(z, param(9))),
                          mul(add(mul(y, param(8)), x), elem));
  const Id linear = m.Emit(
      Op::OpSelect, u,
      {is_packed, op(Op::OpShiftRightLogical, lin_byte, c(2)), linear_dw});
  const Id lshift = m.Emit(
      Op::OpSelect, u,
      {is_packed, mul(op(Op::OpBitwiseAnd, lin_byte, c(3)), c(8)), c(0)});
  const Id detile = m.NewBlock(), retile = m.NewBlock(), copied = m.NewBlock();
  const Id direction = m.Emit(Op::OpINotEqual, m.TypeBool(), {param(10), c(0)});
  m.SelectionMerge(copied);
  m.BranchConditional(direction, detile, retile);
  m.OpenBlock(detile);
  const Id texel =
      op(Op::OpBitwiseAnd,
         op(Op::OpShiftRightLogical, m.Load(u, at(0, tiled)), shift),
         value_mask);
  const Id pk_store = m.NewBlock(), plain_store = m.NewBlock(),
           det_done = m.NewBlock();
  m.SelectionMerge(det_done);
  m.BranchConditional(is_packed, pk_store, plain_store);
  m.OpenBlock(pk_store);
  m.Emit(Op::OpAtomicAnd, u,
         {at(1, linear), c(1), c(0),
          m.Emit(Op::OpNot, u,
                 {op(Op::OpShiftLeftLogical, value_mask, lshift)})});
  m.Emit(Op::OpAtomicOr, u,
         {at(1, linear), c(1), c(0), op(Op::OpShiftLeftLogical, texel, lshift)});
  m.Branch(det_done);
  m.OpenBlock(plain_store);
  m.Store(at(1, linear), texel);
  m.Branch(det_done);
  m.OpenBlock(det_done);
  m.Branch(copied);
  m.OpenBlock(retile);
  const Id value =
      op(Op::OpBitwiseAnd,
         op(Op::OpShiftRightLogical, m.Load(u, at(1, linear)), lshift),
         value_mask);
  const Id packed = m.NewBlock(), direct = m.NewBlock(), stored = m.NewBlock();
  m.SelectionMerge(stored);
  m.BranchConditional(narrow, packed, direct);
  m.OpenBlock(packed);
  // Neighbouring narrow texels share a dword: clear this texel's lanes, then
  // set them. Bytes no texel maps to (padding) are never touched.
  m.Emit(Op::OpAtomicAnd, u,
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
  std::string error;
  if (!Validate(code, &error)) {
    BASE_LOGI("gpuvk", "tiling shader invalid: {}", error.c_str());
    return {};
  }
  return code;
}

}  // namespace gpu::gcn
#endif
