#include "base/arch.h"

#include <gtest/gtest.h>

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_opt.h"

namespace {

using gpu::gcn::spirv::Id;
using gpu::gcn::spirv::Module;
using spv::Op;
using spv::StorageClass;

// Instructions of a binary module, each as its full word list.
base::Vector<base::Vector<u32>> Insts(const base::Vector<u32>& spv) {
  base::Vector<base::Vector<u32>> out;
  for (size_t i = 5; i < spv.size();) {
    const u32 wc = spv[i] >> 16;
    if (!wc || i + wc > spv.size())
      break;
    out.push_back(base::Vector<u32>(spv.data() + i, spv.data() + i + wc));
    i += wc;
  }
  return out;
}

size_t Count(const base::Vector<u32>& spv, Op op) {
  size_t n = 0;
  for (const auto& in : Insts(spv))
    n += (in[0] & 0xffff) == static_cast<u32>(op);
  return n;
}

size_t CountVariables(const base::Vector<u32>& spv, StorageClass sc) {
  size_t n = 0;
  for (const auto& in : Insts(spv))
    n += (in[0] & 0xffff) == static_cast<u32>(Op::OpVariable) &&
         in[3] == static_cast<u32>(sc);
  return n;
}

// The value the single store to `ptr` writes, 0 if there is none.
u32 StoredValue(const base::Vector<u32>& spv, Id ptr) {
  for (const auto& in : Insts(spv))
    if ((in[0] & 0xffff) == static_cast<u32>(Op::OpStore) && in[1] == ptr)
      return in[2];
  return 0;
}

base::Vector<u32> Find(const base::Vector<u32>& spv, Id result) {
  for (const auto& in : Insts(spv))
    if (in.size() > 2 && in[2] == result)
      return in;
  return {};
}

// A compute entry point reading one uint input and writing one uint output.
struct Shader {
  Module m;
  Id t_void, t_u, t_bool, in, out, fn, entry;

  Shader() {
    t_void = m.TypeVoid();
    t_u = m.TypeInt(32, false);
    t_bool = m.TypeBool();
    in = m.Variable(m.TypePointer(StorageClass::Input, t_u),
                    StorageClass::Input);
    out = m.Variable(m.TypePointer(StorageClass::Output, t_u),
                     StorageClass::Output);
    fn = m.BeginFunction(t_void, m.TypeFunction(t_void));
    entry = m.CurrentBlock();
  }
  Id Register() {
    return m.Variable(m.TypePointer(StorageClass::Private, t_u),
                      StorageClass::Private, m.ConstNull(t_u));
  }
  Id Input() { return m.Load(t_u, in); }
  base::Vector<u32> Finish() {
    m.ReturnVoid();
    m.EndFunction();
    m.EntryPoint(spv::ExecutionModel::GLCompute, fn, "main", {});
    m.ExecMode(fn, spv::ExecutionMode::LocalSize, {1, 1, 1});
    base::Vector<u32> opt;
    base::String err;
    EXPECT_TRUE(gpu::gcn::spirv::OptimizeModule(m.Assemble(), &opt, 2, &err))
        << err.c_str();
    return opt;
  }
};

TEST(SpvOpt, MergePhiPairsEachValueWithItsPredecessor) {
  Shader s;
  const Id r = s.Register();
  const Id x = s.Input();
  s.m.Store(r, x);
  const Id then = s.m.NewBlock(), merge = s.m.NewBlock();
  s.m.SelectionMerge(merge);
  s.m.BranchConditional(
      s.m.Emit(Op::OpINotEqual, s.t_bool, {x, s.m.ConstU32(0)}), then, merge);
  s.m.OpenBlock(then);
  const Id y = s.m.Emit(Op::OpIMul, s.t_u, {x, x});
  s.m.Store(r, y);
  s.m.Branch(merge);
  s.m.OpenBlock(merge);
  s.m.Store(s.out, s.m.Load(s.t_u, r));
  const auto spv = s.Finish();

  EXPECT_EQ(CountVariables(spv, StorageClass::Private), 0u);
  EXPECT_EQ(Count(spv, Op::OpLoad), 1u);  // only the input
  const auto phi = Find(spv, StoredValue(spv, s.out));
  ASSERT_EQ(phi.size(), 7u);
  ASSERT_EQ(phi[0] & 0xffff, static_cast<u32>(Op::OpPhi));
  for (u32 k = 3; k < 7; k += 2)
    EXPECT_EQ(phi[k], phi[k + 1] == then ? y : x);
}

TEST(SpvOpt, UnwrittenRegisterNeedsNoPhiAtMerges) {
  Shader s;
  const Id r = s.Register();
  const Id x = s.Input();
  s.m.Store(r, x);
  for (int i = 0; i < 3; i++) {
    const Id then = s.m.NewBlock(), merge = s.m.NewBlock();
    s.m.SelectionMerge(merge);
    s.m.BranchConditional(
        s.m.Emit(Op::OpUGreaterThan, s.t_bool, {x, s.m.ConstU32(i)}), then,
        merge);
    s.m.OpenBlock(then);
    s.m.Branch(merge);
    s.m.OpenBlock(merge);
  }
  s.m.Store(s.out, s.m.Load(s.t_u, r));
  const auto spv = s.Finish();

  EXPECT_EQ(Count(spv, Op::OpPhi), 0u);
  EXPECT_EQ(StoredValue(spv, s.out), x);
}

TEST(SpvOpt, LoopCounterBecomesHeaderPhi) {
  Shader s;
  const Id i = s.Register();
  s.m.Store(i, s.m.ConstU32(0));
  const Id header = s.m.NewBlock(), body = s.m.NewBlock(),
           cont = s.m.NewBlock(), merge = s.m.NewBlock();
  s.m.Branch(header);
  s.m.OpenBlock(header);
  s.m.LoopMerge(merge, cont);
  s.m.Branch(body);
  s.m.OpenBlock(body);
  const Id more = s.m.Emit(Op::OpULessThan, s.t_bool,
                           {s.m.Load(s.t_u, i), s.m.ConstU32(10)});
  s.m.BranchConditional(more, cont, merge);
  s.m.OpenBlock(cont);
  const Id next =
      s.m.Emit(Op::OpIAdd, s.t_u, {s.m.Load(s.t_u, i), s.m.ConstU32(1)});
  s.m.Store(i, next);
  s.m.Branch(header);
  s.m.OpenBlock(merge);
  s.m.Store(s.out, s.m.Load(s.t_u, i));
  const auto spv = s.Finish();

  EXPECT_EQ(Count(spv, Op::OpPhi), 1u);
  const auto phi = Find(spv, StoredValue(spv, s.out));
  ASSERT_EQ(phi.size(), 7u);
  ASSERT_EQ(phi[0] & 0xffff, static_cast<u32>(Op::OpPhi));
  for (u32 k = 3; k < 7; k += 2) {
    if (phi[k + 1] == s.entry)
      EXPECT_EQ(phi[k], s.m.ConstU32(0));
    else
      EXPECT_EQ(phi[k + 1], cont);
  }
  // The increment reads the phi.
  const auto add = Find(spv, next);
  ASSERT_EQ(add.size(), 5u);
  EXPECT_EQ(add[3], phi[2]);
}

TEST(SpvOpt, DynamicallyIndexedRegistersStayInMemory) {
  Shader s;
  const Id t_arr = s.m.TypeArray(s.t_u, 4);
  const Id arr =
      s.m.Variable(s.m.TypePointer(StorageClass::Private, t_arr),
                   StorageClass::Private, s.m.ConstNull(t_arr));
  const Id p_u = s.m.TypePointer(StorageClass::Private, s.t_u);
  const Id x = s.Input();
  s.m.Store(s.m.AccessChain(p_u, arr, {x}), x);
  s.m.Store(s.out, s.m.Load(s.t_u, s.m.AccessChain(p_u, arr, {s.m.ConstU32(1)})));
  const auto spv = s.Finish();

  EXPECT_EQ(CountVariables(spv, StorageClass::Private), 1u);
  EXPECT_EQ(Count(spv, Op::OpStore), 2u);
}

TEST(SpvOpt, ValueNumberingReusesDominatingResults) {
  Shader s;
  const Id x = s.Input();
  const Id y = s.m.Emit(Op::OpBitwiseXor, s.t_u, {x, s.m.ConstU32(5)});
  const Id first = s.m.Emit(Op::OpIMul, s.t_u, {x, y});
  const Id next = s.m.NewBlock();
  s.m.Branch(next);
  s.m.OpenBlock(next);
  const Id again = s.m.Emit(Op::OpIMul, s.t_u, {y, x});
  s.m.Store(s.out, s.m.Emit(Op::OpIAdd, s.t_u, {again, s.m.ConstU32(0)}));
  const auto spv = s.Finish();

  EXPECT_EQ(Count(spv, Op::OpIMul), 1u);
  EXPECT_EQ(Count(spv, Op::OpIAdd), 0u);
  EXPECT_EQ(StoredValue(spv, s.out), first);
}

TEST(SpvOpt, NestedSelectOnTheSameConditionFolds) {
  Shader s;
  const Id x = s.Input();
  const Id c = s.m.Emit(Op::OpINotEqual, s.t_bool, {x, s.m.ConstU32(0)});
  const Id y = s.m.Emit(Op::OpNot, s.t_u, {x});
  const Id inner = s.m.Emit(Op::OpSelect, s.t_u, {c, y, s.m.ConstU32(0)});
  const Id next = s.m.NewBlock();
  s.m.Branch(next);
  s.m.OpenBlock(next);
  s.m.Store(s.out,
            s.m.Emit(Op::OpSelect, s.t_u, {c, inner, s.m.ConstU32(0)}));
  const auto spv = s.Finish();

  EXPECT_EQ(Count(spv, Op::OpSelect), 1u);
  EXPECT_EQ(StoredValue(spv, s.out), inner);
}

TEST(SpvOpt, ExpressionInBothArmsMovesToTheirDominator) {
  Shader s;
  const Id r = s.Register();
  const Id x = s.Input();
  const Id then = s.m.NewBlock(), other = s.m.NewBlock(),
           merge = s.m.NewBlock();
  s.m.SelectionMerge(merge);
  s.m.BranchConditional(
      s.m.Emit(Op::OpINotEqual, s.t_bool, {x, s.m.ConstU32(0)}), then, other);
  s.m.OpenBlock(then);
  s.m.Store(r, s.m.Emit(Op::OpIMul, s.t_u, {x, x}));
  s.m.Branch(merge);
  s.m.OpenBlock(other);
  const Id y = s.m.Emit(Op::OpIMul, s.t_u, {x, x});
  s.m.Store(r, s.m.Emit(Op::OpIAdd, s.t_u, {y, s.m.ConstU32(1)}));
  s.m.Branch(merge);
  s.m.OpenBlock(merge);
  s.m.Store(s.out, s.m.Load(s.t_u, r));
  const auto spv = s.Finish();

  EXPECT_EQ(Count(spv, Op::OpIMul), 1u);
  // The product now precedes the branch.
  bool seen_mul = false;
  for (const auto& in : Insts(spv)) {
    const u32 op = in[0] & 0xffff;
    seen_mul |= op == static_cast<u32>(Op::OpIMul);
    if (op == static_cast<u32>(Op::OpBranchConditional))
      EXPECT_TRUE(seen_mul);
  }
  EXPECT_TRUE(gpu::gcn::spirv::CheckModule(spv, nullptr));
}

TEST(SpvOpt, ConstantBranchDropsTheDeadArm) {
  Shader s;
  const Id r = s.Register();
  const Id x = s.Input();
  s.m.Store(r, x);
  const Id then = s.m.NewBlock(), other = s.m.NewBlock(),
           merge = s.m.NewBlock();
  s.m.SelectionMerge(merge);
  s.m.BranchConditional(
      s.m.Emit(Op::OpIEqual, s.t_bool, {s.m.ConstU32(1), s.m.ConstU32(1)}),
      then, other);
  s.m.OpenBlock(then);
  s.m.Store(r, s.m.Emit(Op::OpIMul, s.t_u, {x, x}));
  s.m.Branch(merge);
  s.m.OpenBlock(other);
  s.m.Store(r, s.m.Emit(Op::OpBitwiseXor, s.t_u, {x, s.m.ConstU32(7)}));
  s.m.Branch(merge);
  s.m.OpenBlock(merge);
  s.m.Store(s.out, s.m.Load(s.t_u, r));
  const auto spv = s.Finish();

  EXPECT_EQ(Count(spv, Op::OpBranchConditional), 0u);
  EXPECT_EQ(Count(spv, Op::OpSelectionMerge), 0u);
  EXPECT_EQ(Count(spv, Op::OpBitwiseXor), 0u);
  EXPECT_EQ(Count(spv, Op::OpPhi), 0u);
  const auto mul = Find(spv, StoredValue(spv, s.out));
  ASSERT_FALSE(mul.empty());
  EXPECT_EQ(mul[0] & 0xffff, static_cast<u32>(Op::OpIMul));
  EXPECT_TRUE(gpu::gcn::spirv::CheckModule(spv, nullptr));
}

TEST(SpvOpt, CheckRejectsUndefinedIds) {
  Shader s;
  s.m.Store(s.out, s.m.Alloc());
  s.m.ReturnVoid();
  s.m.EndFunction();
  s.m.EntryPoint(spv::ExecutionModel::GLCompute, s.fn, "main", {});
  base::String err;
  EXPECT_FALSE(gpu::gcn::spirv::CheckModule(s.m.Assemble(), &err));
  EXPECT_FALSE(err.empty());
}

}  // namespace
