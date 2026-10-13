#include "gpu/gcn/gcn_translate.h"

#ifdef DELTA_HAVE_SPIRV_BACKEND
#include "base/logging.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"

namespace gpu::gcn {

base::Vector<u32> BuildMeshIndirectShader() {
  using namespace spirv;
  using spv::Op;
  using spv::StorageClass;
  Module m;
  const Id u = m.TypeInt(), b = m.TypeBool();
  const Id array = m.TypeRuntimeArray(u);
  m.Decorate(array, spv::Decoration::ArrayStride, {4});
  const Id block = m.TypeStruct({array});
  m.Decorate(block, spv::Decoration::Block);
  m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
  Id buffers[2];
  for (u32 i = 0; i < 2; ++i) {
    buffers[i] = m.Variable(m.TypePointer(StorageClass::StorageBuffer, block),
                            StorageClass::StorageBuffer);
    m.Decorate(buffers[i], spv::Decoration::DescriptorSet, {0});
    m.Decorate(buffers[i], spv::Decoration::Binding, {i});
  }
  const Id params = m.TypeStruct({u, u, u, u, u, u});
  m.Decorate(params, spv::Decoration::Block);
  for (u32 i = 0; i < 6; ++i)
    m.MemberDecorate(params, i, spv::Decoration::Offset, {i * 4});
  const Id push = m.Variable(m.TypePointer(StorageClass::PushConstant, params),
                             StorageClass::PushConstant);
  const Id fn = m.BeginFunction(m.TypeVoid(), m.TypeFunction(m.TypeVoid()));
  const auto c = [&](u32 v) { return m.ConstU32(v); };
  const auto op = [&](Op code, Id a, Id d) { return m.Emit(code, u, {a, d}); };
  const auto param = [&](u32 i) {
    return m.Load(u, m.AccessChain(m.TypePointer(StorageClass::PushConstant, u),
                                   push, {c(i)}));
  };
  const auto at = [&](u32 binding, Id index) {
    return m.AccessChain(m.TypePointer(StorageClass::StorageBuffer, u),
                         buffers[binding], {c(0), index});
  };
  Id args[4];
  for (u32 i = 0; i < 4; ++i) {
    args[i] = m.Load(u, at(0, op(Op::OpIAdd, param(0), c(i))));
    m.Store(at(1, op(Op::OpIAdd, param(1), c(i))), args[i]);
  }
  const Id extra = m.Emit(
      Op::OpSelect, u,
      {m.Emit(Op::OpINotEqual, b, {op(Op::OpUMod, args[0], param(2)), c(0)}),
       c(1), c(0)});
  const Id x = op(Op::OpIAdd, op(Op::OpUDiv, args[0], param(2)), extra);
  const Id divisor =
      m.Emit(Op::OpSelect, u,
             {m.Emit(Op::OpINotEqual, b, {args[1], c(0)}), args[1], c(1)});
  Id valid = m.Emit(Op::OpULessThanEqual, b, {x, param(3)});
  valid = m.Emit(Op::OpLogicalAnd, b,
                 {valid, m.Emit(Op::OpULessThanEqual, b, {args[1], param(4)})});
  valid = m.Emit(Op::OpLogicalAnd, b,
                 {valid, m.Emit(Op::OpULessThanEqual, b,
                                {x, op(Op::OpUDiv, param(5), divisor)})});
  const Id groups[] = {x, args[1], c(1)};
  for (u32 i = 0; i < 3; ++i)
    m.Store(at(1, op(Op::OpIAdd, param(1), c(4 + i))),
            m.Emit(Op::OpSelect, u, {valid, groups[i], c(0)}));
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::GLCompute, fn, "main", {});
  m.ExecMode(fn, spv::ExecutionMode::LocalSize, {1, 1, 1});
  base::Vector<u32> result;
  base::String error;
  if (!Finalize(m.Assemble(), &result, &error))
    BASE_LOGE("gpu", "mesh argument shader: {}", error.c_str());
  return result;
}

}  // namespace gpu::gcn
#endif
