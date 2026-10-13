#include "gpu/gcn/gcn_translate.h"

#ifdef DELTA_HAVE_SPIRV_BACKEND
#include "base/logging.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"

namespace gpu::gcn {

base::Vector<u32> BuildIndexConversionShader(u32 index_type,
                                             bool quads,
                                             IndexConversionMode mode) {
  const bool indirect = mode != IndexConversionMode::kDirect;
  const bool strided = mode == IndexConversionMode::kIndirectStrided;
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
  for (u32 i = 0; i < (indirect ? 3u : 2u); i++) {
    buffers[i] = m.Variable(m.TypePointer(StorageClass::StorageBuffer, block),
                            StorageClass::StorageBuffer);
    m.Decorate(buffers[i], spv::Decoration::DescriptorSet, {0});
    m.Decorate(buffers[i], spv::Decoration::Binding, {i});
  }
  const Id pc_type =
      indirect ? m.TypeStruct({u, u, u, u, u}) : m.TypeStruct({u, u, u});
  m.Decorate(pc_type, spv::Decoration::Block);
  for (u32 i = 0; i < (indirect ? 5u : 3u); i++)
    m.MemberDecorate(pc_type, i, spv::Decoration::Offset, {i * 4});
  const Id pc = m.Variable(m.TypePointer(StorageClass::PushConstant, pc_type),
                           StorageClass::PushConstant);
  const Id vec = m.TypeVec(u, 3);
  const Id gid =
      m.Variable(m.TypePointer(StorageClass::Input, vec), StorageClass::Input);
  m.Decorate(gid, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::GlobalInvocationId)});
  Id groups = 0;
  if (strided) {
    groups = m.Variable(m.TypePointer(StorageClass::Input, vec),
                        StorageClass::Input);
    m.Decorate(groups, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::NumWorkgroups)});
  }
  const Id fn = m.BeginFunction(m.TypeVoid(), m.TypeFunction(m.TypeVoid()));
  const Id cursor =
      strided ? m.Emit(Op::OpVariable, m.TypePointer(StorageClass::Function, u),
                       {static_cast<u32>(StorageClass::Function)})
              : 0;
  m.EntryPoint(spv::ExecutionModel::GLCompute, fn, "main",
               strided ? base::Vector<Id>{gid, groups} : base::Vector<Id>{gid});
  m.ExecMode(fn, spv::ExecutionMode::LocalSize, {64, 1, 1});
  const auto c = [&](u32 v) { return m.ConstU32(v); };
  const auto op = [&](Op code, Id a, Id b) { return m.Emit(code, u, {a, b}); };
  const auto param = [&](u32 i) {
    return m.Load(u, m.AccessChain(m.TypePointer(StorageClass::PushConstant, u),
                                   pc, {c(i)}));
  };
  const auto at = [&](u32 binding, Id index) {
    return m.AccessChain(ptr, buffers[binding], {c(0), index});
  };
  Id index = m.CompositeExtract(u, m.Load(vec, gid), 0);
  Id count = param(0), first = c(0);
  if (indirect) {
    const auto argument = [&](u32 i) {
      return m.Load(u, at(2, op(Op::OpIAdd, param(3), c(i))));
    };
    first = argument(2);
    const Id available =
        m.Emit(Op::OpSelect, u,
               {m.Emit(Op::OpULessThan, m.TypeBool(), {first, count}),
                op(Op::OpISub, count, first), c(0)});
    const Id requested = argument(0);
    count =
        m.Emit(Op::OpSelect, u,
               {m.Emit(Op::OpULessThan, m.TypeBool(), {requested, available}),
                requested, available});
    count = op(Op::OpIMul, op(Op::OpUDiv, count, c(4)), c(6));
    const Id write_args = m.NewBlock(), args_done = m.NewBlock();
    const Id first_invocation =
        m.Emit(Op::OpIEqual, m.TypeBool(), {index, c(0)});
    m.SelectionMerge(args_done);
    m.BranchConditional(first_invocation, write_args, args_done);
    m.OpenBlock(write_args);
    for (u32 i = 0; i < 5; i++)
      m.Store(at(1, op(Op::OpIAdd, param(4), c(i))), i == 0   ? count
                                                     : i == 2 ? c(0)
                                                              : argument(i));
    m.Branch(args_done);
    m.OpenBlock(args_done);
  }
  Id header = 0, next = 0;
  Id stride = 0;
  if (strided) {
    stride =
        op(Op::OpIMul, m.CompositeExtract(u, m.Load(vec, groups), 0), c(64));
    m.Store(cursor, index);
    header = m.NewBlock();
    next = m.NewBlock();
    m.Branch(header);
    m.OpenBlock(header);
    index = m.Load(u, cursor);
  }
  const Id body = m.NewBlock(), end = m.NewBlock();
  const Id valid = m.Emit(Op::OpULessThan, m.TypeBool(), {index, count});
  if (strided)
    m.LoopMerge(end, next);
  else
    m.SelectionMerge(end);
  m.BranchConditional(valid, body, end);
  m.OpenBlock(body);
  Id source = index;
  if (quads) {
    const Id corner = op(Op::OpUMod, index, c(6));
    const Id tail = m.Emit(Op::OpUGreaterThan, m.TypeBool(), {corner, c(3)});
    const Id repeat = m.Emit(Op::OpIEqual, m.TypeBool(), {corner, c(3)});
    const Id vertex =
        m.Emit(Op::OpSelect, u,
               {repeat, c(0),
                m.Emit(Op::OpSelect, u,
                       {tail, op(Op::OpISub, corner, c(2)), corner})});
    source = op(Op::OpIAdd, op(Op::OpIMul, op(Op::OpUDiv, index, c(6)), c(4)),
                vertex);
  }
  source = op(Op::OpIAdd, source, first);
  const u32 bytes = index_type == 1 ? 4 : index_type == 2 ? 1 : 2;
  const Id address = op(Op::OpIAdd, op(Op::OpIMul, source, c(bytes)), param(2));
  Id value = m.Load(u, at(0, op(Op::OpShiftRightLogical, address, c(2))));
  if (bytes != 4) {
    const Id shift = op(Op::OpIMul, op(Op::OpBitwiseAnd, address, c(3)), c(8));
    value = op(Op::OpBitwiseAnd, op(Op::OpShiftRightLogical, value, shift),
               c(bytes == 1 ? 0xff : 0xffff));
  }
  m.Store(at(1, op(Op::OpIAdd, index, param(1))), value);
  if (strided) {
    m.Branch(next);
    m.OpenBlock(next);
    m.Store(cursor, op(Op::OpIAdd, index, stride));
    m.Branch(header);
  } else {
    m.Branch(end);
  }
  m.OpenBlock(end);
  m.ReturnVoid();
  m.EndFunction();
  auto code = m.Assemble();
  if (strided)
    code = Optimize(code);
  base::String error;
  if (!Validate(code, &error)) {
    BASE_LOGI("gpuvk", "index conversion shader invalid: {}", error.c_str());
    return {};
  }
  return code;
}

}  // namespace gpu::gcn
#endif
