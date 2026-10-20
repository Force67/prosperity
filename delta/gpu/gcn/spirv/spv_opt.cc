/*
 * PS4Delta : PS4 emulation and research project
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>

#include "gpu/gcn/spirv/spv_opt.h"

#include <cstring>

#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/xstring.h"

namespace gpu::gcn::spirv {
namespace {

using spv::Op;
constexpr u32 kNone = ~0u;

bool AllIdOperands(Op op) {
  using enum Op;
  switch (op) {
    case OpNop:
    case OpUndef:
    case OpLabel:
    case OpFunctionParameter:
    case OpFunctionEnd:
    case OpReturn:
    case OpKill:
    case OpUnreachable:
    case OpTerminateInvocation:
    case OpDemoteToHelperInvocation:
    case OpIsHelperInvocationEXT:
    case OpEmitVertex:
    case OpEndPrimitive:
    case OpBeginInvocationInterlockEXT:
    case OpEndInvocationInterlockEXT:
    case OpConstantTrue:
    case OpConstantFalse:
    case OpConstantNull:
    case OpConstantComposite:
    case OpTypeVoid:
    case OpTypeBool:
    case OpTypeSampler:
    case OpTypeStruct:
    case OpTypeFunction:
    case OpTypeArray:
    case OpTypeRuntimeArray:
    case OpTypeSampledImage:
    case OpPhi:
    case OpFunctionCall:
    case OpReturnValue:
    case OpBranch:
    case OpAccessChain:
    case OpInBoundsAccessChain:
    case OpPtrAccessChain:
    case OpCopyObject:
    case OpCopyLogical:
    case OpCompositeConstruct:
    case OpVectorExtractDynamic:
    case OpVectorInsertDynamic:
    case OpVectorTimesScalar:
    case OpMatrixTimesScalar:
    case OpVectorTimesMatrix:
    case OpMatrixTimesVector:
    case OpMatrixTimesMatrix:
    case OpOuterProduct:
    case OpDot:
    case OpTranspose:
    case OpSNegate:
    case OpFNegate:
    case OpIAdd:
    case OpFAdd:
    case OpISub:
    case OpFSub:
    case OpIMul:
    case OpFMul:
    case OpUDiv:
    case OpSDiv:
    case OpFDiv:
    case OpUMod:
    case OpSRem:
    case OpSMod:
    case OpFRem:
    case OpFMod:
    case OpIAddCarry:
    case OpISubBorrow:
    case OpUMulExtended:
    case OpSMulExtended:
    case OpShiftRightLogical:
    case OpShiftRightArithmetic:
    case OpShiftLeftLogical:
    case OpBitwiseOr:
    case OpBitwiseXor:
    case OpBitwiseAnd:
    case OpNot:
    case OpBitFieldInsert:
    case OpBitFieldSExtract:
    case OpBitFieldUExtract:
    case OpBitReverse:
    case OpBitCount:
    case OpAny:
    case OpAll:
    case OpIsNan:
    case OpIsInf:
    case OpLogicalEqual:
    case OpLogicalNotEqual:
    case OpLogicalOr:
    case OpLogicalAnd:
    case OpLogicalNot:
    case OpSelect:
    case OpIEqual:
    case OpINotEqual:
    case OpUGreaterThan:
    case OpSGreaterThan:
    case OpUGreaterThanEqual:
    case OpSGreaterThanEqual:
    case OpULessThan:
    case OpSLessThan:
    case OpULessThanEqual:
    case OpSLessThanEqual:
    case OpFOrdEqual:
    case OpFUnordEqual:
    case OpFOrdNotEqual:
    case OpFUnordNotEqual:
    case OpFOrdLessThan:
    case OpFUnordLessThan:
    case OpFOrdGreaterThan:
    case OpFUnordGreaterThan:
    case OpFOrdLessThanEqual:
    case OpFUnordLessThanEqual:
    case OpFOrdGreaterThanEqual:
    case OpFUnordGreaterThanEqual:
    case OpConvertFToU:
    case OpConvertFToS:
    case OpConvertSToF:
    case OpConvertUToF:
    case OpUConvert:
    case OpSConvert:
    case OpFConvert:
    case OpQuantizeToF16:
    case OpConvertPtrToU:
    case OpConvertUToPtr:
    case OpBitcast:
    case OpDPdx:
    case OpDPdy:
    case OpFwidth:
    case OpDPdxFine:
    case OpDPdyFine:
    case OpFwidthFine:
    case OpDPdxCoarse:
    case OpDPdyCoarse:
    case OpFwidthCoarse:
    case OpImage:
    case OpSampledImage:
    case OpImageQuerySizeLod:
    case OpImageQuerySize:
    case OpImageQueryLod:
    case OpImageQueryLevels:
    case OpImageQuerySamples:
    case OpImageTexelPointer:
    case OpControlBarrier:
    case OpMemoryBarrier:
    case OpAtomicLoad:
    case OpAtomicStore:
    case OpAtomicExchange:
    case OpAtomicCompareExchange:
    case OpAtomicIIncrement:
    case OpAtomicIDecrement:
    case OpAtomicIAdd:
    case OpAtomicISub:
    case OpAtomicSMin:
    case OpAtomicUMin:
    case OpAtomicSMax:
    case OpAtomicUMax:
    case OpAtomicAnd:
    case OpAtomicOr:
    case OpAtomicXor:
    case OpAtomicFAddEXT:
    case OpAtomicFMinEXT:
    case OpAtomicFMaxEXT:
    case OpGroupNonUniformElect:
    case OpGroupNonUniformAll:
    case OpGroupNonUniformAny:
    case OpGroupNonUniformAllEqual:
    case OpGroupNonUniformBroadcast:
    case OpGroupNonUniformBroadcastFirst:
    case OpGroupNonUniformBallot:
    case OpGroupNonUniformInverseBallot:
    case OpGroupNonUniformBallotBitExtract:
    case OpGroupNonUniformBallotFindLSB:
    case OpGroupNonUniformBallotFindMSB:
    case OpGroupNonUniformShuffle:
    case OpGroupNonUniformShuffleXor:
    case OpGroupNonUniformShuffleUp:
    case OpGroupNonUniformShuffleDown:
    case OpGroupNonUniformQuadBroadcast:
    case OpGroupNonUniformQuadSwap:
    case OpReadClockKHR:
    case OpSetMeshOutputsEXT:
    case OpEmitMeshTasksEXT:
      return true;
    default:
      return false;
  }
}

u32 StringWords(const u32* w, u32 n) {
  for (u32 i = 0; i < n; i++)
    if ((w[i] & 0xff000000u) == 0 || (w[i] & 0xff0000u) == 0 ||
        (w[i] & 0xff00u) == 0 || (w[i] & 0xffu) == 0)
      return i + 1;
  return n;
}

// Calls f(word&) for every id among an instruction's operands (the words after
// its result type and id). False when the layout is not modeled.
template <typename F>
bool ForEachId(Op op, u32* w, u32 n, F&& f) {
  using enum Op;
  const auto ids = [&](u32 from, u32 to) {
    for (u32 i = from; i < to && i < n; i++)
      f(w[i]);
  };
  const auto memory_access = [&](u32 at) {
    if (n <= at)
      return true;
    const u32 mask = w[at];
    if (mask & ~0x3fu)
      return false;
    u32 i = at + 1;
    if (mask & 0x2)
      i++;
    if (mask & 0x8)
      ids(i, i + 1), i++;
    if (mask & 0x10)
      ids(i, i + 1), i++;
    return i == n;
  };
  const auto image = [&](u32 fixed) {
    ids(0, fixed);
    ids(fixed + 1, n);
    return true;
  };
  switch (op) {
    case OpCompositeExtract:
    case OpArrayLength:
    case OpSelectionMerge:
    case OpLine:
    case OpExecutionMode:
    case OpName:
    case OpMemberName:
    case OpDecorate:
    case OpMemberDecorate:
    case OpDecorateString:
    case OpMemberDecorateString:
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeImage:
    case OpTypeForwardPointer:
      ids(0, 1);
      return true;
    case OpCompositeInsert:
    case OpVectorShuffle:
    case OpLoopMerge:
      ids(0, 2);
      return true;
    case OpVariable:
    case OpTypePointer:
      ids(1, 2);
      return true;
    case OpExtInst:
      ids(0, 1);
      ids(2, n);
      return true;
    case OpLoad:
      ids(0, 1);
      return memory_access(1);
    case OpStore:
      ids(0, 2);
      return memory_access(2);
    case OpBranchConditional:
      ids(0, 3);
      return true;
    case OpSwitch:
      ids(0, 2);
      for (u32 i = 3; i < n; i += 2)
        f(w[i]);
      return (n & 1) == 0;
    case OpEntryPoint:
      ids(1, 2);
      ids(2 + StringWords(w + 2, n > 2 ? n - 2 : 0), n);
      return true;
    case OpImageSampleImplicitLod:
    case OpImageSampleExplicitLod:
    case OpImageSampleProjImplicitLod:
    case OpImageSampleProjExplicitLod:
    case OpImageFetch:
    case OpImageRead:
    case OpImageSparseSampleImplicitLod:
    case OpImageSparseSampleExplicitLod:
    case OpImageSparseFetch:
    case OpImageSparseRead:
      return image(2);
    case OpImageSampleDrefImplicitLod:
    case OpImageSampleDrefExplicitLod:
    case OpImageSampleProjDrefImplicitLod:
    case OpImageSampleProjDrefExplicitLod:
    case OpImageGather:
    case OpImageDrefGather:
    case OpImageWrite:
    case OpImageSparseSampleDrefImplicitLod:
    case OpImageSparseSampleDrefExplicitLod:
    case OpImageSparseGather:
    case OpImageSparseDrefGather:
      return image(3);
    case OpCapability:
    case OpExtension:
    case OpExtInstImport:
    case OpMemoryModel:
    case OpString:
    case OpSource:
    case OpSourceExtension:
    case OpModuleProcessed:
    case OpNoLine:
    case OpTypeInt:
    case OpTypeFloat:
    case OpConstant:
    case OpFunction:
      if (op == OpFunction)
        ids(1, 2);
      return true;
    default:
      if (op == OpGroupNonUniformBallotBitCount ||
          (op >= OpGroupNonUniformIAdd && op <= OpGroupNonUniformLogicalXor)) {
        ids(0, 1);
        ids(2, n);
        return true;
      }
      if (AllIdOperands(op)) {
        ids(0, n);
        return true;
      }
      return false;
  }
}

bool IsType(Op op) {
  return op >= Op::OpTypeVoid && op < Op::OpTypeForwardPointer;
}

bool IsShift(Op op) {
  return op == Op::OpShiftLeftLogical || op == Op::OpShiftRightLogical ||
         op == Op::OpShiftRightArithmetic;
}

bool IsTerminator(Op op) {
  using enum Op;
  switch (op) {
    case OpBranch:
    case OpBranchConditional:
    case OpSwitch:
    case OpReturn:
    case OpReturnValue:
    case OpKill:
    case OpUnreachable:
    case OpTerminateInvocation:
    case OpEmitMeshTasksEXT:
      return true;
    default:
      return false;
  }
}

// Body instructions DCE must keep even when nothing reads their result.
bool HasSideEffects(Op op) {
  using enum Op;
  if (IsTerminator(op))
    return true;
  switch (op) {
    case OpStore:
    case OpCopyMemory:
    case OpFunctionCall:
    case OpImageWrite:
    case OpControlBarrier:
    case OpMemoryBarrier:
    case OpEmitVertex:
    case OpEndPrimitive:
    case OpDemoteToHelperInvocation:
    case OpSelectionMerge:
    case OpLoopMerge:
    case OpLine:
    case OpSetMeshOutputsEXT:
    case OpBeginInvocationInterlockEXT:
    case OpEndInvocationInterlockEXT:
    case OpAtomicLoad:
    case OpAtomicStore:
    case OpAtomicExchange:
    case OpAtomicCompareExchange:
    case OpAtomicIIncrement:
    case OpAtomicIDecrement:
    case OpAtomicIAdd:
    case OpAtomicISub:
    case OpAtomicSMin:
    case OpAtomicUMin:
    case OpAtomicSMax:
    case OpAtomicUMax:
    case OpAtomicAnd:
    case OpAtomicOr:
    case OpAtomicXor:
    case OpAtomicFAddEXT:
    case OpAtomicFMinEXT:
    case OpAtomicFMaxEXT:
      return true;
    default:
      return false;
  }
}

// Pure and position independent: an identical instruction that dominates this
// one computes the same value.
bool Numberable(Op op) {
  using enum Op;
  switch (op) {
    case OpPhi:
    case OpAccessChain:
    case OpInBoundsAccessChain:
    case OpPtrAccessChain:
    case OpCompositeConstruct:
    case OpCompositeExtract:
    case OpCompositeInsert:
    case OpVectorShuffle:
    case OpVectorExtractDynamic:
    case OpVectorInsertDynamic:
    case OpVectorTimesScalar:
    case OpDot:
    case OpSNegate:
    case OpFNegate:
    case OpIAdd:
    case OpFAdd:
    case OpISub:
    case OpFSub:
    case OpIMul:
    case OpFMul:
    case OpUDiv:
    case OpSDiv:
    case OpFDiv:
    case OpUMod:
    case OpSRem:
    case OpSMod:
    case OpFRem:
    case OpFMod:
    case OpIAddCarry:
    case OpISubBorrow:
    case OpUMulExtended:
    case OpSMulExtended:
    case OpShiftRightLogical:
    case OpShiftRightArithmetic:
    case OpShiftLeftLogical:
    case OpBitwiseOr:
    case OpBitwiseXor:
    case OpBitwiseAnd:
    case OpNot:
    case OpBitFieldInsert:
    case OpBitFieldSExtract:
    case OpBitFieldUExtract:
    case OpBitReverse:
    case OpBitCount:
    case OpAny:
    case OpAll:
    case OpIsNan:
    case OpIsInf:
    case OpLogicalEqual:
    case OpLogicalNotEqual:
    case OpLogicalOr:
    case OpLogicalAnd:
    case OpLogicalNot:
    case OpSelect:
    case OpIEqual:
    case OpINotEqual:
    case OpUGreaterThan:
    case OpSGreaterThan:
    case OpUGreaterThanEqual:
    case OpSGreaterThanEqual:
    case OpULessThan:
    case OpSLessThan:
    case OpULessThanEqual:
    case OpSLessThanEqual:
    case OpFOrdEqual:
    case OpFUnordEqual:
    case OpFOrdNotEqual:
    case OpFUnordNotEqual:
    case OpFOrdLessThan:
    case OpFUnordLessThan:
    case OpFOrdGreaterThan:
    case OpFUnordGreaterThan:
    case OpFOrdLessThanEqual:
    case OpFUnordLessThanEqual:
    case OpFOrdGreaterThanEqual:
    case OpFUnordGreaterThanEqual:
    case OpConvertFToU:
    case OpConvertFToS:
    case OpConvertSToF:
    case OpConvertUToF:
    case OpUConvert:
    case OpSConvert:
    case OpFConvert:
    case OpQuantizeToF16:
    case OpConvertUToPtr:
    case OpBitcast:
    case OpArrayLength:
      return true;
    default:
      return false;
  }
}

bool Commutative(Op op) {
  using enum Op;
  switch (op) {
    case OpIAdd:
    case OpFAdd:
    case OpIMul:
    case OpFMul:
    case OpBitwiseOr:
    case OpBitwiseXor:
    case OpBitwiseAnd:
    case OpLogicalEqual:
    case OpLogicalNotEqual:
    case OpLogicalOr:
    case OpLogicalAnd:
    case OpIEqual:
    case OpINotEqual:
    case OpIAddCarry:
    case OpUMulExtended:
    case OpSMulExtended:
      return true;
    default:
      return false;
  }
}

// u64 -> u32 open addressing map; no erase.
class FlatMap {
 public:
  void Clear(size_t expect) {
    u32 bits = 4;
    while ((size_t{1} << bits) < expect * 2)
      bits++;
    Reset(bits);
  }
  const u32* Find(u64 key) const {
    if (keys_.empty())
      return nullptr;
    for (u32 i = Slot(key);; i = (i + 1) & mask_) {
      if (keys_[i] == key)
        return &vals_[i];
      if (keys_[i] == kEmpty)
        return nullptr;
    }
  }
  void Set(u64 key, u32 value) {
    if (keys_.empty() || (size_ + 1) * 2 > keys_.size())
      Grow();
    u32 i = Slot(key);
    while (keys_[i] != kEmpty && keys_[i] != key)
      i = (i + 1) & mask_;
    if (keys_[i] == kEmpty) {
      keys_[i] = key;
      size_++;
    }
    vals_[i] = value;
  }

 private:
  static constexpr u64 kEmpty = ~0ull;
  u32 Slot(u64 key) const {
    return static_cast<u32>((key * 0x9E3779B97F4A7C15ull) >> shift_);
  }
  void Reset(u32 bits) {
    keys_.assign(size_t{1} << bits, kEmpty);
    vals_.resize(size_t{1} << bits);
    mask_ = (1u << bits) - 1;
    shift_ = 64 - bits;
    size_ = 0;
  }
  void Grow() {
    base::Vector<u64> keys;
    base::Vector<u32> vals;
    keys.swap(keys_);
    vals.swap(vals_);
    u32 bits = 4;
    while ((size_t{1} << bits) < (keys.size() * 2 > 16 ? keys.size() * 2 : 16))
      bits++;
    Reset(bits);
    for (size_t i = 0; i < keys.size(); i++)
      if (keys[i] != kEmpty)
        Set(keys[i], vals[i]);
  }

  base::Vector<u64> keys_;
  base::Vector<u32> vals_;
  u32 mask_ = 0, shift_ = 64;
  size_t size_ = 0;
};

// (type, bits) -> constant id.
class ConstMap {
 public:
  const u32* Find(u32 type, u64 bits) const {
    if (slots_.empty())
      return nullptr;
    for (u32 i = Slot(type, bits);; i = (i + 1) & mask_) {
      const Entry& e = slots_[i];
      if (e.id == 0)
        return nullptr;
      if (e.type == type && e.bits == bits)
        return &e.id;
    }
  }
  void Set(u32 type, u64 bits, u32 id) {
    if ((size_ + 1) * 2 > slots_.size()) {
      base::Vector<Entry> old;
      old.swap(slots_);
      slots_.resize(old.empty() ? 64 : old.size() * 2);
      mask_ = static_cast<u32>(slots_.size() - 1);
      size_ = 0;
      for (const Entry& e : old)
        if (e.id)
          Set(e.type, e.bits, e.id);
    }
    u32 i = Slot(type, bits);
    while (slots_[i].id && !(slots_[i].type == type && slots_[i].bits == bits))
      i = (i + 1) & mask_;
    if (!slots_[i].id)
      size_++;
    slots_[i] = {bits, type, id};
  }

 private:
  struct Entry {
    u64 bits = 0;
    u32 type = 0, id = 0;
  };
  u32 Slot(u32 type, u64 bits) const {
    return static_cast<u32>(((bits ^ (u64{type} << 40)) * 0x9E3779B97F4A7C15ull) >>
                            32) &
           mask_;
  }
  base::Vector<Entry> slots_;
  u32 mask_ = 0;
  size_t size_ = 0;
};

u64 Key(u32 a, u32 b) {
  return (static_cast<u64>(a) << 32) | b;
}

struct Inst {
  Op op;
  u32 type;
  u32 result;
  u32 begin;  // first operand word in Ir::words_
  u32 count;
};

struct Block {
  u32 label_inst = 0;
  u32 label = 0;
  base::Vector<u32> body;  // phis first, terminator last
  base::Vector<u32> preds, succs;
  base::Vector<u32> phis;  // made by promotion, written before the body
  bool loop_header = false;
};

struct Function {
  u32 inst = kNone, end = kNone;
  base::Vector<u32> params;
  base::Vector<Block> blocks;
};

enum Kind : u8 { kOther, kInt32, kInt64, kFloat32, kBool };

// A promotable pointer: a candidate variable, or a constant access chain into
// one ending on a scalar or vector.
struct Pointer {
  u32 var;
  u32 flat;  // leaf index within the variable
  u32 leaf;  // leaf type
  u32 chain;  // access chain instruction, kNone for the variable itself
  u32 slot;   // promotion slot in the owning function, kNone until used
};

struct Var {
  u32 id;
  u32 owner;  // function index, kNone while unused
  bool bad;
};

struct Phi {
  u32 inst;
  u32 block;
  u32 slot;
};

struct Slot {
  u32 type;
  u32 init;
};

class Ir {
 public:
  bool Parse(const base::Vector<u32>& in, base::String* err);
  bool Check(base::String* err);
  bool Promote();
  void Simplify();
  void Sweep();
  void Write(base::Vector<u32>* out) const;

 private:
  u32* Ops(const Inst& inst) { return words_.data() + inst.begin; }
  const u32* Ops(const Inst& inst) const { return words_.data() + inst.begin; }
  const Inst* Def(u32 id) const {
    return id < def_.size() && def_[id] != kNone ? &insts_[def_[id]] : nullptr;
  }
  u32 TypeOf(u32 id) const {
    const Inst* d = Def(id);
    return d ? d->type : 0;
  }
  Kind KindOf(u32 type) const {
    return type < kind_.size() ? static_cast<Kind>(kind_[type]) : kOther;
  }
  bool Known(u32 id) const { return known_[id] != 0; }
  u32 Find(u32 id);
  u32 NewId();
  u32 AddInst(Op op, u32 type, u32 result, const u32* ops, u32 count);
  u32 AddGlobal(Op op, u32 type, const u32* ops, u32 count);
  u32 Constant(u32 type, u64 bits);
  u32 Null(u32 type);
  u32 Undef(u32 type);
  void NoteConstant(u32 idx);

  bool BuildCfg(Function& f);
  bool ClassifyPointers();
  void PromoteFunction(u32 fn);
  u32 SlotOf(Pointer& p);
  u32 SlotInit(const Pointer& p, u32 leaf);
  u32 Read(u32 slot, u32 block);
  u32 Cur(u32 b, u32 slot) const {
    if (!cur_dense_.empty())
      return cur_dense_[b * slots_.size() + slot];
    const u32* v = cur_.Find(Key(b, slot));
    return v ? *v : 0;
  }
  void SetCur(u32 b, u32 slot, u32 v) {
    if (!cur_dense_.empty())
      cur_dense_[b * slots_.size() + slot] = v;
    else
      cur_.Set(Key(b, slot), v);
  }
  bool Walk(u32 slot, u32* block, u32* value);
  u32 NewPhi(u32 slot, u32 block, const u32* ops, u32 count);
  void Seal(u32 block);
  void Drain();
  void RemoveTrivialPhis(u32 first_phi);

  void SimplifyFunction(Function& f);
  bool Pure(Inst& in);
  void FoldBranches(Function& f);
  void Hoist(Function& f,
             const base::Vector<u32>& order,
             const base::Vector<u32>& rpo,
             const base::Vector<u32>& idom);
  u32 Fold(u32 idx);
  u32 FoldInt(Inst& in, u32* o);
  u32 Number(u32 idx);

  bool IsAggregate(u32 type) const;
  bool Implies(u32 a, u32 b) const;
  bool ReadOnly(u32 pointer) const;
  u32 LeafCount(u32 type);

  u32 header_[5] = {};
  u32 bound_ = 0;
  base::Vector<Inst> insts_;
  base::Vector<u32> words_;
  base::Vector<u32> globals_;
  base::Vector<Function> fns_;
  base::Vector<u32> def_, repl_, block_of_, ptr_;
  base::Vector<u64> bits_;
  base::Vector<u8> kind_, known_, dead_, live_;
  base::Vector<Pointer> pointers_;
  base::Vector<Var> vars_;
  FlatMap nulls_, undefs_, leaf_counts_, read_only_, buffer_blocks_;
  ConstMap consts_;
  u32 glsl_ = 0;
  u32 entry_fn_ = kNone;

  // Promotion state, per function.
  Function* fn_ = nullptr;
  FlatMap slot_of_, cur_;
  base::Vector<u32> cur_dense_;
  base::Vector<Slot> slots_;
  // Each slot's value inside the block being filled; stamp is block + 1.
  struct Local {
    u32 value, stamp, written;
  };
  base::Vector<Local> local_;
  base::Vector<u32> written_;
  base::Vector<Phi> phis_;
  base::Vector<u8> sealed_, filled_;
  base::Vector<u32> walk_mark_;
  u32 walk_gen_ = 0;
  base::Vector<base::Vector<u32>> incomplete_;
  base::Vector<u32> fill_, path_, scratch_;
  struct Frame {
    u32 block, next, value, ops, path;
    bool mixed;
  };
  base::Vector<Frame> frames_;
  base::Vector<u32> ops_;
  base::Vector<u8> busy_;

  // Value numbering, per function.
  struct Entry {
    u32 inst, hash, next;
  };
  base::Vector<u32> buckets_;
  base::Vector<Entry> entries_;
  u32 bucket_mask_ = 0;
};

u32 Ir::Find(u32 id) {
  u32 root = id;
  while (repl_[root])
    root = repl_[root];
  while (repl_[id] && repl_[id] != root) {
    const u32 next = repl_[id];
    repl_[id] = root;
    id = next;
  }
  return root;
}

u32 Ir::NewId() {
  const u32 id = bound_++;
  def_.push_back(kNone);
  repl_.push_back(0);
  bits_.push_back(0);
  known_.push_back(0);
  kind_.push_back(kOther);
  ptr_.push_back(0);
  return id;
}

u32 Ir::AddInst(Op op, u32 type, u32 result, const u32* ops, u32 count) {
  const u32 idx = static_cast<u32>(insts_.size());
  insts_.push_back({op, type, result, static_cast<u32>(words_.size()), count});
  words_.insert(words_.end(), ops, ops + count);
  dead_.push_back(0);
  if (result)
    def_[result] = idx;
  return idx;
}

u32 Ir::AddGlobal(Op op, u32 type, const u32* ops, u32 count) {
  const u32 id = NewId();
  const u32 idx = AddInst(op, type, id, ops, count);
  globals_.push_back(idx);
  NoteConstant(idx);
  return id;
}

void Ir::NoteConstant(u32 idx) {
  using enum Op;
  const Inst& in = insts_[idx];
  const Kind kind = KindOf(in.type);
  u64 bits;
  if (in.op == OpConstant && in.count == 1 && kind != kOther && kind != kInt64)
    bits = Ops(in)[0];
  else if (in.op == OpConstant && in.count == 2 && kind == kInt64)
    bits = Ops(in)[0] | u64{Ops(in)[1]} << 32;
  else if (in.op == OpConstantTrue || in.op == OpConstantFalse)
    bits = in.op == OpConstantTrue;
  else if (in.op == OpConstantNull && kind != kOther)
    bits = 0;
  else
    return;
  known_[in.result] = 1;
  bits_[in.result] = bits;
  if (!consts_.Find(in.type, bits))
    consts_.Set(in.type, bits, in.result);
}

u32 Ir::Constant(u32 type, u64 bits) {
  if (const u32* id = consts_.Find(type, bits))
    return *id;
  if (KindOf(type) == kBool)
    return AddGlobal(bits ? Op::OpConstantTrue : Op::OpConstantFalse, type,
                     nullptr, 0);
  const u32 words[2] = {static_cast<u32>(bits), static_cast<u32>(bits >> 32)};
  return AddGlobal(Op::OpConstant, type, words, KindOf(type) == kInt64 ? 2 : 1);
}

u32 Ir::Null(u32 type) {
  if (KindOf(type) != kOther)
    return Constant(type, 0);
  if (const u32* id = nulls_.Find(type))
    return *id;
  const u32 id = AddGlobal(Op::OpConstantNull, type, nullptr, 0);
  nulls_.Set(type, id);
  return id;
}

u32 Ir::Undef(u32 type) {
  if (const u32* id = undefs_.Find(type))
    return *id;
  const u32 id = AddGlobal(Op::OpUndef, type, nullptr, 0);
  undefs_.Set(type, id);
  return id;
}

bool Ir::Parse(const base::Vector<u32>& in, base::String* err) {
  using enum Op;
  const auto fail = [&](const char* why) {
    if (err)
      *err = why;
    return false;
  };
  if (in.size() < 5 || in[0] != spv::MagicNumber)
    return fail("bad header");
  for (u32 i = 0; i < 5; i++)
    header_[i] = in[i];
  bound_ = in[3];
  if (bound_ == 0 || bound_ > (1u << 26))
    return fail("bad id bound");
  def_.assign(bound_, kNone);
  repl_.assign(bound_, 0);
  bits_.assign(bound_, 0);
  known_.assign(bound_, 0);
  kind_.assign(bound_, kOther);
  ptr_.assign(bound_, 0);
  insts_.reserve(in.size() / 3);
  words_.reserve(in.size() + in.size() / 4);
  words_.assign(in.begin(), in.end());
  dead_.reserve(in.size() / 3);
  const u32* w = in.data();
  const size_t n = in.size();
  bool in_fn = false;
  for (size_t i = 5; i < n;) {
    const u32 wc = w[i] >> 16;
    const Op op = static_cast<Op>(w[i] & 0xffff);
    if (wc == 0 || i + wc > n)
      return fail("bad instruction length");
    bool has_result, has_type;
    spv::HasResultAndType(op, &has_result, &has_type);
    u32 k = 1, type = 0, result = 0;
    if (has_type + has_result >= wc)
      return fail("truncated instruction");
    if (has_type)
      type = w[i + k++];
    if (has_result) {
      result = w[i + k++];
      if (result == 0 || result >= bound_)
        return fail("result id out of bound");
      if (def_[result] != kNone)
        return fail("id defined twice");
    }
    const u32 idx = static_cast<u32>(insts_.size());
    insts_.push_back({op, type, result, static_cast<u32>(i + k), wc - k});
    dead_.push_back(0);
    if (result)
      def_[result] = idx;
    i += wc;
    switch (op) {
      case OpTypeInt:
        if (Ops(insts_[idx])[0] == 32)
          kind_[result] = kInt32;
        else if (Ops(insts_[idx])[0] == 64)
          kind_[result] = kInt64;
        globals_.push_back(idx);
        continue;
      case OpTypeFloat:
        if (Ops(insts_[idx])[0] == 32)
          kind_[result] = kFloat32;
        globals_.push_back(idx);
        continue;
      case OpTypeBool:
        kind_[result] = kBool;
        globals_.push_back(idx);
        continue;
      case OpExtInstImport:
        if (insts_[idx].count >= 3 &&
            std::memcmp(Ops(insts_[idx]), "GLSL.std.450", 12) == 0)
          glsl_ = result;
        globals_.push_back(idx);
        continue;
      case OpFunction:
        if (in_fn)
          return fail("nested function");
        fns_.emplace_back().inst = idx;
        in_fn = true;
        continue;
      case OpFunctionParameter:
        if (!in_fn || !fns_.back().blocks.empty())
          return fail("stray function parameter");
        fns_.back().params.push_back(idx);
        continue;
      case OpLabel: {
        if (!in_fn)
          return fail("label outside a function");
        Block& b = fns_.back().blocks.emplace_back();
        b.label_inst = idx;
        b.label = result;
        continue;
      }
      case OpFunctionEnd:
        if (!in_fn)
          return fail("stray function end");
        fns_.back().end = idx;
        in_fn = false;
        continue;
      case OpLine:
      case OpNoLine:
        if (!in_fn)
          globals_.push_back(idx);
        continue;
      default:
        break;
    }
    if (in_fn) {
      if (fns_.back().blocks.empty())
        return fail("instruction outside a block");
      fns_.back().blocks.back().body.push_back(idx);
    } else {
      globals_.push_back(idx);
      NoteConstant(idx);
    }
  }
  if (in_fn)
    return fail("unterminated function");
  for (const u32 g : globals_) {
    const Inst& e = insts_[g];
    if (e.op != OpEntryPoint || e.count < 2)
      continue;
    if (entry_fn_ != kNone) {
      entry_fn_ = kNone;
      break;
    }
    for (u32 f = 0; f < fns_.size(); f++)
      if (insts_[fns_[f].inst].result == Ops(e)[1])
        entry_fn_ = f;
  }
  for (const u32 g : globals_) {
    const Inst& d = insts_[g];
    if (d.op == OpDecorate && d.count >= 2 &&
        Ops(d)[1] == static_cast<u32>(spv::Decoration::NonWritable))
      read_only_.Set(Ops(d)[0], 1);
    if (d.op == OpDecorate && d.count >= 2 &&
        Ops(d)[1] == static_cast<u32>(spv::Decoration::BufferBlock))
      buffer_blocks_.Set(Ops(d)[0], 1);
  }
  block_of_.assign(bound_, kNone);
  for (Function& f : fns_)
    if (!BuildCfg(f))
      return fail("malformed control flow");
  return true;
}

bool Ir::BuildCfg(Function& f) {
  using enum Op;
  if (f.blocks.empty())
    return false;
  for (u32 b = 0; b < f.blocks.size(); b++)
    block_of_[f.blocks[b].label] = b;
  for (u32 b = 0; b < f.blocks.size(); b++) {
    Block& blk = f.blocks[b];
    if (blk.body.empty())
      return false;
    const Inst& t = insts_[blk.body.back()];
    if (!IsTerminator(t.op))
      return false;
    blk.loop_header = blk.body.size() > 1 &&
                      insts_[blk.body[blk.body.size() - 2]].op == OpLoopMerge;
    const u32* o = Ops(t);
    const auto edge = [&](u32 label) {
      if (label >= bound_ || block_of_[label] == kNone)
        return false;
      const u32 s = block_of_[label];
      if (s >= f.blocks.size() || f.blocks[s].label != label)
        return false;
      for (const u32 e : blk.succs)
        if (e == s)
          return true;
      blk.succs.push_back(s);
      f.blocks[s].preds.push_back(b);
      return true;
    };
    bool ok = true;
    if (t.op == OpBranch)
      ok = t.count == 1 && edge(o[0]);
    else if (t.op == OpBranchConditional)
      ok = t.count >= 3 && edge(o[1]) && edge(o[2]);
    else if (t.op == OpSwitch) {
      ok = t.count >= 2 && edge(o[1]);
      for (u32 i = 3; ok && i < t.count; i += 2)
        ok = edge(o[i]);
    }
    if (!ok)
      return false;
  }
  return true;
}

bool Ir::Check(base::String* err) {
  using enum Op;
  const auto fail = [&](const char* why) {
    if (err)
      *err = why;
    return false;
  };
  bool defined = true;
  const auto use = [&](u32& id) {
    if (id == 0 || id >= bound_ || def_[id] == kNone)
      defined = false;
  };
  for (const u32 g : globals_) {
    Inst& in = insts_[g];
    if (in.type)
      use(in.type);
    ForEachId(in.op, Ops(in), in.count, use);
    if (!defined)
      return fail("global uses an undefined id");
  }
  for (Function& f : fns_) {
    Inst& fi = insts_[f.inst];
    ForEachId(fi.op, Ops(fi), fi.count, use);
    for (const Block& b : f.blocks) {
      bool phis = true;
      for (u32 i = 0; i < b.body.size(); i++) {
        Inst& in = insts_[b.body[i]];
        if (in.op == OpPhi && !phis)
          return fail("phi after a non-phi instruction");
        phis = in.op == OpPhi;
        if (IsTerminator(in.op) != (i + 1 == b.body.size()))
          return fail("terminator not at the end of its block");
        if (in.type)
          use(in.type);
        ForEachId(in.op, Ops(in), in.count, use);
        if (!defined)
          return fail("instruction uses an undefined id");
      }
    }
  }
  return true;
}

// Memory no invocation writes: loads from it can be value-numbered.
bool Ir::ReadOnly(u32 pointer) const {
  using enum Op;
  const Inst* d = Def(pointer);
  while (d && (d->op == OpAccessChain || d->op == OpInBoundsAccessChain))
    d = Def(Ops(*d)[0]);
  if (!d || d->op != OpVariable)
    return false;
  switch (static_cast<spv::StorageClass>(Ops(*d)[0])) {
    case spv::StorageClass::UniformConstant:
    case spv::StorageClass::PushConstant:
    case spv::StorageClass::Input:
      return true;
    case spv::StorageClass::Uniform: {
      // Before StorageBuffer, a BufferBlock struct here was a writable SSBO.
      const Inst* p = Def(d->type);
      return p && !buffer_blocks_.Find(Ops(*p)[1]);
    }
    case spv::StorageClass::StorageBuffer:
      return read_only_.Find(d->result) != nullptr;
    default:
      return false;
  }
}

// Whether a being true makes b true, seen through a LogicalAnd chain.
bool Ir::Implies(u32 a, u32 b) const {
  for (u32 depth = 0; depth < 8; depth++) {
    if (a == b)
      return true;
    const Inst* d = Def(a);
    if (!d || d->op != Op::OpLogicalAnd)
      return false;
    if (Ops(*d)[0] == b || Ops(*d)[1] == b)
      return true;
    const Inst* l = Def(Ops(*d)[0]);
    a = l && l->op == Op::OpLogicalAnd ? Ops(*d)[0] : Ops(*d)[1];
  }
  return false;
}

bool Ir::IsAggregate(u32 type) const {
  const Inst* t = Def(type);
  return t && (t->op == Op::OpTypeArray || t->op == Op::OpTypeStruct ||
               t->op == Op::OpTypeRuntimeArray);
}

u32 Ir::LeafCount(u32 type) {
  if (const u32* n = leaf_counts_.Find(type))
    return *n;
  const Inst* t = Def(type);
  u64 n = 1;
  if (t && t->op == Op::OpTypeArray) {
    const u32 len = Ops(*t)[1];
    n = Known(len) && bits_[len] < (1u << 24)
            ? bits_[len] * LeafCount(Ops(*t)[0])
            : kNone;
  } else if (t && t->op == Op::OpTypeStruct) {
    n = 0;
    for (u32 m = 0; m < t->count; m++)
      n += LeafCount(Ops(*t)[m]);
  } else if (t && t->op == Op::OpTypeRuntimeArray) {
    n = kNone;
  }
  const u32 r = n >= (1u << 24) ? kNone : static_cast<u32>(n);
  leaf_counts_.Set(type, r);
  return r;
}

// Finds the variables the promotion can turn into SSA values: Function
// variables, and Private ones only the entry point touches, read and written
// solely through loads, stores and constant access chains.
bool Ir::ClassifyPointers() {
  using enum Op;
  for (const u32 g : globals_) {
    Inst& in = insts_[g];
    if (!ForEachId(in.op, Ops(in), in.count, [](u32&) {}))
      return false;
  }
  const auto pointee = [&](u32 ptr_type) {
    const Inst* p = Def(ptr_type);
    return p && p->op == OpTypePointer ? Ops(*p)[1] : 0;
  };
  const auto add_var = [&](const Inst& in, u32 owner) {
    const u32 sc = Ops(in)[0];
    if (sc != static_cast<u32>(spv::StorageClass::Private) &&
        sc != static_cast<u32>(spv::StorageClass::Function))
      return;
    if (sc == static_cast<u32>(spv::StorageClass::Private) && owner != kNone)
      return;
    const u32 t = pointee(in.type);
    vars_.push_back({in.result, owner, t == 0 || LeafCount(t) == kNone});
    pointers_.push_back(
        {static_cast<u32>(vars_.size() - 1), 0, t, kNone, kNone});
    ptr_[in.result] = static_cast<u32>(pointers_.size());
  };
  for (const u32 g : globals_)
    if (insts_[g].op == OpVariable)
      add_var(insts_[g], kNone);
  for (u32 f = 0; f < fns_.size(); f++)
    for (const u32 idx : fns_[f].blocks[0].body)
      if (insts_[idx].op == OpVariable)
        add_var(insts_[idx], f);
  if (vars_.empty())
    return true;

  const auto bad = [&](u32 id) {
    if (id < bound_ && ptr_[id])
      vars_[pointers_[ptr_[id] - 1].var].bad = true;
  };
  for (u32 f = 0; f < fns_.size(); f++) {
    for (const Block& b : fns_[f].blocks) {
      for (const u32 idx : b.body) {
        Inst& in = insts_[idx];
        u32* o = Ops(in);
        if (in.op == OpVariable)
          continue;
        if ((in.op == OpLoad || in.op == OpStore) && in.count >= 1 &&
            o[0] < bound_ && ptr_[o[0]]) {
          const Pointer& p = pointers_[ptr_[o[0]] - 1];
          Var& v = vars_[p.var];
          if (v.owner == kNone)
            v.owner = f;
          if (v.owner != f || IsAggregate(p.leaf) ||
              in.count != (in.op == OpLoad ? 1u : 2u))
            v.bad = true;
          if (in.op == OpStore)
            bad(o[1]);
          continue;
        }
        if ((in.op == OpAccessChain || in.op == OpInBoundsAccessChain) &&
            in.count >= 1 && o[0] < bound_ && ptr_[o[0]]) {
          const Pointer base_ptr = pointers_[ptr_[o[0]] - 1];
          Var& v = vars_[base_ptr.var];
          if (base_ptr.chain != kNone) {
            v.bad = true;
            continue;
          }
          u32 t = base_ptr.leaf, flat = 0;
          bool ok = true;
          for (u32 i = 1; ok && i < in.count; i++) {
            const Inst* ti = Def(t);
            if (o[i] >= bound_ || !Known(o[i]) || !ti) {
              ok = false;
              break;
            }
            const u64 index = bits_[o[i]];
            if (ti->op == OpTypeArray && index < bits_[Ops(*ti)[1]]) {
              t = Ops(*ti)[0];
              flat += static_cast<u32>(index) * LeafCount(t);
            } else if (ti->op == OpTypeStruct && index < ti->count) {
              for (u32 m = 0; m < index; m++)
                flat += LeafCount(Ops(*ti)[m]);
              t = Ops(*ti)[index];
            } else {
              ok = false;
            }
          }
          if (!ok || IsAggregate(t)) {
            v.bad = true;
            continue;
          }
          pointers_.push_back({base_ptr.var, flat, t, idx, kNone});
          ptr_[in.result] = static_cast<u32>(pointers_.size());
          continue;
        }
        if (!ForEachId(in.op, o, in.count, [&](u32& id) { bad(id); }))
          return false;
      }
    }
  }
  for (Var& v : vars_) {
    const Inst& in = insts_[def_[v.id]];
    if (Ops(in)[0] == static_cast<u32>(spv::StorageClass::Private) &&
        (v.owner != entry_fn_ || entry_fn_ == kNone))
      v.bad = v.bad || v.owner != kNone;
  }
  return true;
}

u32 Ir::SlotInit(const Pointer& p, u32 leaf) {
  using enum Op;
  const Inst& var = insts_[def_[vars_[p.var].id]];
  u32 c = var.count > 1 ? Ops(var)[1] : 0;
  if (!c)
    return Undef(leaf);
  if (p.chain != kNone) {
    const Inst& chain = insts_[p.chain];
    for (u32 i = 1; i < chain.count; i++) {
      const Inst* d = Def(c);
      if (!d || d->op != OpConstantComposite)
        break;
      const u64 index = bits_[Ops(chain)[i]];
      if (index >= d->count)
        return Undef(leaf);
      c = Ops(*d)[index];
    }
  }
  const Inst* d = Def(c);
  if (!d || d->op == OpConstantNull || d->type != leaf)
    return Null(leaf);
  return c;
}

u32 Ir::SlotOf(Pointer& p) {
  if (p.slot != kNone)
    return p.slot;
  const u64 key = Key(p.var, p.flat);
  if (const u32* s = slot_of_.Find(key))
    return p.slot = *s;
  const u32 s = static_cast<u32>(slots_.size());
  slots_.push_back({p.leaf, SlotInit(p, p.leaf)});
  local_.push_back({0, 0, 0});
  slot_of_.Set(key, s);
  return p.slot = s;
}

u32 Ir::NewPhi(u32 slot, u32 b, const u32* ops, u32 count) {
  const u32 id = NewId();
  const u32 idx = AddInst(Op::OpPhi, slots_[slot].type, id, ops, count);
  fn_->blocks[b].phis.push_back(idx);
  phis_.push_back({idx, b, slot});
  return id;
}

// Follows single-predecessor chains from *b. True with *v once the value is
// known; false when the walk stops at a merge block (*b) whose predecessors
// decide it. Blocks passed on the way go to path_ and get the value later.
bool Ir::Walk(u32 slot, u32* b, u32* v) {
  walk_gen_++;
  for (;;) {
    const u32 at = *b;
    if (const u32 c = Cur(at, slot)) {
      *v = c;
      return true;
    }
    const Block& blk = fn_->blocks[at];
    if (busy_[at] || blk.preds.empty() || walk_mark_[at] == walk_gen_) {
      // Only unreachable or malformed flow comes back around without a loop
      // header in between.
      *v = slots_[slot].init;
      return true;
    }
    if (!sealed_[at] || (blk.loop_header && blk.preds.size() > 1)) {
      // The phi is the value right away; its operands come later.
      *v = NewPhi(slot, at, nullptr, 0);
      const u32 phi = static_cast<u32>(phis_.size() - 1);
      if (sealed_[at])
        fill_.push_back(phi);
      else
        incomplete_[at].push_back(phi);
      SetCur(at, slot, *v);
      return true;
    }
    if (blk.preds.size() > 1)
      return false;
    walk_mark_[at] = walk_gen_;
    path_.push_back(at);
    *b = blk.preds[0];
  }
}

// The value of slot at the end of block b. A merge block that is not a loop
// header gets a phi only when its predecessors disagree; every cycle in
// structured control flow passes a loop header, whose phi stops the search.
u32 Ir::Read(u32 slot, u32 b) {
  const auto settle = [&](u32 from, u32 v) {
    for (u32 i = from; i < path_.size(); i++)
      SetCur(path_[i], slot, v);
    path_.resize(from);
  };
  u32 v;
  if (Walk(slot, &b, &v)) {
    settle(0, v);
    return v;
  }
  frames_.push_back({b, 0, 0, static_cast<u32>(ops_.size()), 0, false});
  busy_[b] = 1;
  for (;;) {
    Frame& f = frames_.back();
    const Block& blk = fn_->blocks[f.block];
    if (f.next < blk.preds.size()) {
      u32 at = blk.preds[f.next++];
      const u32 from = static_cast<u32>(path_.size());
      if (!Walk(slot, &at, &v)) {
        frames_.push_back({at, 0, 0, static_cast<u32>(ops_.size()), from, false});
        busy_[at] = 1;
        continue;
      }
      settle(from, v);
    } else {
      v = f.mixed ? NewPhi(slot, f.block, ops_.data() + f.ops,
                           static_cast<u32>(ops_.size()) - f.ops)
                  : f.value;
      SetCur(f.block, slot, v);
      busy_[f.block] = 0;
      settle(f.path, v);
      ops_.resize(f.ops);
      frames_.pop_back();
      if (frames_.empty())
        return v;
    }
    Frame& top = frames_.back();
    const Block& merge = fn_->blocks[top.block];
    if (ops_.size() == top.ops)
      top.value = v;
    else if (v != top.value)
      top.mixed = true;
    ops_.push_back(v);
    ops_.push_back(fn_->blocks[merge.preds[top.next - 1]].label);
  }
}

void Ir::Seal(u32 b) {
  sealed_[b] = 1;
  for (const u32 phi : incomplete_[b])
    fill_.push_back(phi);
  incomplete_[b].clear();
  Drain();
}

void Ir::Drain() {
  while (!fill_.empty()) {
    const Phi phi = phis_[fill_.back()];
    fill_.pop_back();
    scratch_.clear();
    for (u32 i = 0; i < fn_->blocks[phi.block].preds.size(); i++) {
      const u32 pred = fn_->blocks[phi.block].preds[i];
      const u32 v = Read(phi.slot, pred);
      scratch_.push_back(v);
      scratch_.push_back(fn_->blocks[pred].label);
    }
    Inst& in = insts_[phi.inst];
    in.begin = static_cast<u32>(words_.size());
    in.count = static_cast<u32>(scratch_.size());
    words_.insert(words_.end(), scratch_.begin(), scratch_.end());
  }
}

void Ir::RemoveTrivialPhis(u32 first) {
  const u32 count = static_cast<u32>(phis_.size()) - first;
  if (!count)
    return;
  FlatMap index;
  index.Clear(count);
  for (u32 i = first; i < phis_.size(); i++)
    index.Set(insts_[phis_[i].inst].result, i - first);
  for (u32 i = first; i < phis_.size(); i++) {
    const Inst& in = insts_[phis_[i].inst];
    u32* o = Ops(in);
    for (u32 k = 0; k < in.count; k += 2)
      o[k] = Find(o[k]);
  }
  // Users of each phi among the phis, as lists a replaced phi hands on to
  // the phi replacing it.
  struct User {
    u32 phi, next;
  };
  base::Vector<u32> head(count, kNone);
  base::Vector<User> users;
  for (u32 i = 0; i < count; i++) {
    const Inst& in = insts_[phis_[first + i].inst];
    const u32* o = Ops(in);
    for (u32 k = 0; k < in.count; k += 2)
      if (const u32* j = index.Find(o[k])) {
        users.push_back({i, head[*j]});
        head[*j] = static_cast<u32>(users.size() - 1);
      }
  }

  base::Vector<u32> work;
  work.reserve(count);
  for (u32 i = count; i-- > 0;)
    work.push_back(i);
  while (!work.empty()) {
    const u32 i = work.back();
    work.pop_back();
    const u32 idx = phis_[first + i].inst;
    if (dead_[idx])
      continue;
    Inst& in = insts_[idx];
    u32* o = Ops(in);
    u32 same = 0;
    bool trivial = true;
    for (u32 k = 0; k < in.count; k += 2) {
      const u32 v = Find(o[k]);
      o[k] = v;
      if (v == in.result || v == same)
        continue;
      if (same) {
        trivial = false;
        break;
      }
      same = v;
    }
    if (!trivial)
      continue;
    const u32 result = in.result;
    const u32 value = same ? same : Undef(in.type);
    repl_[result] = value;
    dead_[idx] = 1;
    u32 last = kNone;
    for (u32 u = head[i]; u != kNone; u = users[u].next) {
      last = u;
      if (!dead_[phis_[first + users[u].phi].inst])
        work.push_back(users[u].phi);
    }
    if (const u32* j = index.Find(value); j && last != kNone) {
      users[last].next = head[*j];
      head[*j] = head[i];
      head[i] = kNone;
    }
  }
}

void Ir::PromoteFunction(u32 f) {
  using enum Op;
  fn_ = &fns_[f];
  const u32 nblocks = static_cast<u32>(fn_->blocks.size());
  slot_of_.Clear(64);
  slots_.clear();
  local_.clear();
  for (Pointer& p : pointers_) {
    const Var& v = vars_[p.var];
    if (!v.bad && v.owner == f && !IsAggregate(p.leaf))
      SlotOf(p);
  }
  // Each block's value per slot: a dense table unless that gets large.
  const u64 cells = u64{nblocks} * slots_.size();
  cur_dense_.clear();
  cur_.Clear(256);
  if (cells <= (1u << 22))
    cur_dense_.resize(cells, 0);
  sealed_.assign(nblocks, 0);
  filled_.assign(nblocks, 0);
  walk_mark_.assign(nblocks, 0);
  busy_.assign(nblocks, 0);
  walk_gen_ = 0;
  incomplete_.resize(nblocks);
  for (auto& v : incomplete_)
    v.clear();
  const u32 first_phi = static_cast<u32>(phis_.size());

  const auto promoted = [&](u32 id) -> Pointer* {
    if (id >= ptr_.size() || !ptr_[id])
      return nullptr;
    Pointer& p = pointers_[ptr_[id] - 1];
    return vars_[p.var].bad ? nullptr : &p;
  };
  const auto ready = [&](u32 b) {
    for (const u32 p : fn_->blocks[b].preds)
      if (!filled_[p])
        return false;
    return true;
  };
  for (u32 b = 0; b < nblocks; b++) {
    if (!sealed_[b] && ready(b))
      Seal(b);
    const u32 stamp = b + 1;
    // Indexing, not references: Read appends instructions.
    for (u32 i = 0; i < fn_->blocks[b].body.size(); i++) {
      const u32 idx = fn_->blocks[b].body[i];
      const Op op = insts_[idx].op;
      if (op == OpLoad) {
        const Inst& in = insts_[idx];
        if (Pointer* p = promoted(Ops(in)[0])) {
          const u32 result = in.result;
          const u32 slot = SlotOf(*p);
          if (local_[slot].stamp != stamp) {
            const u32 v = Read(slot, b);
            Drain();
            local_[slot].value = v;
            local_[slot].stamp = stamp;
          }
          repl_[result] = local_[slot].value;
          dead_[idx] = 1;
        }
      } else if (op == OpStore) {
        const u32 ptr = Ops(insts_[idx])[0];
        if (Pointer* p = promoted(ptr)) {
          const u32 value = Ops(insts_[idx])[1];
          Local& l = local_[SlotOf(*p)];
          l.value = value;
          l.stamp = stamp;
          if (l.written != stamp) {
            l.written = stamp;
            written_.push_back(p->slot);
          }
          dead_[idx] = 1;
        }
      } else if (op == OpAccessChain || op == OpInBoundsAccessChain ||
                 op == OpVariable) {
        if (promoted(insts_[idx].result))
          dead_[idx] = 1;
      }
    }
    for (const u32 slot : written_)
      SetCur(b, slot, local_[slot].value);
    written_.clear();
    filled_[b] = 1;
    for (u32 i = 0; i < fn_->blocks[b].succs.size(); i++) {
      const u32 s = fn_->blocks[b].succs[i];
      if (!sealed_[s] && ready(s))
        Seal(s);
    }
  }
  for (u32 b = 0; b < nblocks; b++)
    if (!sealed_[b])
      Seal(b);
  RemoveTrivialPhis(first_phi);
  fn_ = nullptr;
}

bool Ir::Promote() {
  if (!ClassifyPointers())
    return false;
  for (const Var& v : vars_)
    if (!v.bad)
      dead_[def_[v.id]] = 1;
  for (u32 f = 0; f < fns_.size(); f++) {
    bool any = false;
    for (const Var& v : vars_)
      any = any || (!v.bad && v.owner == f);
    if (any)
      PromoteFunction(f);
  }
  return true;
}

u32 Ir::FoldInt(Inst& in, u32* o) {
  using enum Op;
  const u32 t0 = in.count > 0 ? TypeOf(o[0]) : 0;
  const u32 t1 = in.count > 1 ? TypeOf(o[1]) : 0;
  const Kind k = KindOf(t0);
  if ((k != kInt32 && k != kInt64) || (in.count > 1 && KindOf(t1) != k &&
                                       !IsShift(in.op)))
    return 0;
  const u32 width = k == kInt64 ? 64 : 32;
  const u64 mask = width == 64 ? ~0ull : 0xffffffffull;
  const auto sext = [&](u64 v) {
    return width == 64 ? static_cast<i64>(v)
                       : static_cast<i64>(static_cast<i32>(v));
  };
  const bool k0 = Known(o[0]), k1 = in.count > 1 && Known(o[1]);
  const u64 x = bits_[o[0]], y = in.count > 1 ? bits_[o[1]] : 0;
  const i64 sx = sext(x), sy = sext(y);
  if (KindOf(in.type) == kBool) {
    if (in.count != 2)
      return 0;
    if (o[0] == o[1]) {
      switch (in.op) {
        case OpIEqual:
        case OpUGreaterThanEqual:
        case OpSGreaterThanEqual:
        case OpULessThanEqual:
        case OpSLessThanEqual:
          return Constant(in.type, 1);
        default:
          return Constant(in.type, 0);
      }
    }
    if (!k0 || !k1)
      return 0;
    bool r;
    switch (in.op) {
      case OpIEqual: r = x == y; break;
      case OpINotEqual: r = x != y; break;
      case OpULessThan: r = x < y; break;
      case OpULessThanEqual: r = x <= y; break;
      case OpUGreaterThan: r = x > y; break;
      case OpUGreaterThanEqual: r = x >= y; break;
      case OpSLessThan: r = sx < sy; break;
      case OpSLessThanEqual: r = sx <= sy; break;
      case OpSGreaterThan: r = sx > sy; break;
      case OpSGreaterThanEqual: r = sx >= sy; break;
      default: return 0;
    }
    return Constant(in.type, r);
  }
  if (KindOf(in.type) != k)
    return 0;
  const auto make = [&](u64 v) { return Constant(in.type, v & mask); };
  if (in.count == 1) {
    if (in.op == OpNot && k0)
      return make(~x);
    if (in.op == OpSNegate && k0)
      return make(0 - x);
    if (in.op == OpNot) {
      const Inst* d = Def(o[0]);
      if (d && d->op == OpNot && TypeOf(Ops(*d)[0]) == in.type)
        return Ops(*d)[0];
    }
    return 0;
  }
  if (in.count != 2)
    return 0;
  if (k0 && k1) {
    switch (in.op) {
      case OpIAdd: return make(x + y);
      case OpISub: return make(x - y);
      case OpIMul: return make(x * y);
      case OpBitwiseAnd: return make(x & y);
      case OpBitwiseOr: return make(x | y);
      case OpBitwiseXor: return make(x ^ y);
      case OpShiftLeftLogical: return y < width ? make(x << y) : 0;
      case OpShiftRightLogical: return y < width ? make(x >> y) : 0;
      case OpShiftRightArithmetic:
        return y < width ? make(static_cast<u64>(sx >> y)) : 0;
      case OpUDiv: return y ? make(x / y) : 0;
      case OpUMod: return y ? make(x % y) : 0;
      default: return 0;
    }
  }
  // Identities may only hand back an operand of the result's own type.
  const bool same0 = t0 == in.type, same1 = t1 == in.type;
  switch (in.op) {
    case OpIAdd:
    case OpBitwiseOr:
    case OpBitwiseXor:
      if (k1 && y == 0 && same0)
        return o[0];
      if (k0 && x == 0 && same1)
        return o[1];
      if (in.op == OpBitwiseOr && o[0] == o[1] && same0)
        return o[0];
      if (in.op == OpBitwiseXor && o[0] == o[1])
        return make(0);
      if (in.op == OpBitwiseOr && ((k0 && x == mask) || (k1 && y == mask)))
        return make(mask);
      return 0;
    case OpISub:
      if (k1 && y == 0 && same0)
        return o[0];
      if (o[0] == o[1])
        return make(0);
      return 0;
    case OpIMul:
      if ((k0 && x == 0) || (k1 && y == 0))
        return make(0);
      if (k1 && y == 1 && same0)
        return o[0];
      if (k0 && x == 1 && same1)
        return o[1];
      return 0;
    case OpBitwiseAnd:
      if ((k0 && x == 0) || (k1 && y == 0))
        return make(0);
      if (k1 && y == mask && same0)
        return o[0];
      if (k0 && x == mask && same1)
        return o[1];
      if (o[0] == o[1] && same0)
        return o[0];
      return 0;
    case OpShiftLeftLogical:
    case OpShiftRightLogical:
    case OpShiftRightArithmetic:
      if (k1 && y == 0 && same0)
        return o[0];
      if (k0 && x == 0)
        return make(0);
      return 0;
    default:
      return 0;
  }
}

u32 Ir::Fold(u32 idx) {
  using enum Op;
  Inst& in = insts_[idx];
  u32* o = Ops(in);
  switch (in.op) {
    case OpPhi: {
      u32 same = 0;
      for (u32 k = 0; k < in.count; k += 2) {
        if (o[k] == in.result || o[k] == same)
          continue;
        if (same)
          return 0;
        same = o[k];
      }
      return same;
    }
    case OpCopyObject:
      return o[0];
    case OpBitcast: {
      if (TypeOf(o[0]) == in.type)
        return o[0];
      const Inst* d = Def(o[0]);
      if (d && d->op == OpBitcast) {
        const u32 src = Ops(*d)[0];
        if (TypeOf(src) == in.type)
          return src;
        o[0] = src;
      }
      const Kind to = KindOf(in.type), from = KindOf(TypeOf(o[0]));
      if (Known(o[0]) && to != kOther && to != kBool && from != kOther &&
          from != kBool && (to == kInt64) == (from == kInt64))
        return Constant(in.type, bits_[o[0]]);
      return 0;
    }
    case OpSelect: {
      // select(c, select(c, a, x), b) == select(c, a, b), either arm.
      for (u32 k = 1; k < 3; k++) {
        const Inst* d = Def(o[k]);
        if (d && d->op == OpSelect && Ops(*d)[0] == o[0])
          o[k] = Ops(*d)[k];
      }
      if (o[1] == o[2])
        return o[1];
      if (Known(o[0]))
        return bits_[o[0]] ? o[1] : o[2];
      // select(c, select(c && e, a, x), x) == select(c && e, a, x): the
      // inner condition already implies the outer one.
      if (const Inst* d = Def(o[1]); d && d->op == OpSelect &&
                                     Ops(*d)[2] == o[2] &&
                                     Implies(Ops(*d)[0], o[0]))
        return o[1];
      if (KindOf(in.type) == kBool && Known(o[1]) && Known(o[2]) &&
          bits_[o[1]] && !bits_[o[2]])
        return o[0];
      return 0;
    }
    case OpLogicalNot: {
      if (Known(o[0]))
        return Constant(in.type, !bits_[o[0]]);
      const Inst* d = Def(o[0]);
      if (d && d->op == OpLogicalNot)
        return Ops(*d)[0];
      return 0;
    }
    case OpLogicalAnd:
    case OpLogicalOr: {
      if (KindOf(in.type) != kBool)
        return 0;
      if (o[0] == o[1])
        return o[0];
      const u32 absorbing = in.op == OpLogicalOr;
      for (u32 k = 0; k < 2; k++)
        if (Known(o[k]))
          return bits_[o[k]] == absorbing ? Constant(in.type, absorbing)
                                          : o[k ^ 1];
      return 0;
    }
    case OpLogicalEqual:
    case OpLogicalNotEqual: {
      if (KindOf(in.type) != kBool)
        return 0;
      const bool eq = in.op == OpLogicalEqual;
      if (o[0] == o[1])
        return Constant(in.type, eq);
      if (Known(o[0]) && Known(o[1]))
        return Constant(in.type, (bits_[o[0]] == bits_[o[1]]) == eq);
      return 0;
    }
    case OpVectorExtractDynamic: {
      const Inst* vt = Def(TypeOf(o[0]));
      if (!Known(o[1]) || !vt || vt->op != OpTypeVector ||
          bits_[o[1]] >= Ops(*vt)[1])
        return 0;
      in.op = OpCompositeExtract;
      o[1] = static_cast<u32>(bits_[o[1]]);
      [[fallthrough]];
    }
    case OpCompositeExtract: {
      if (in.count != 2)
        return 0;
      const Inst* d = Def(o[0]);
      if (!d)
        return 0;
      if (d->op == OpConstantNull)
        return Null(in.type);
      if (d->op != OpCompositeConstruct && d->op != OpConstantComposite)
        return 0;
      const Inst* ct = Def(d->type);
      if (ct && ct->op == OpTypeVector && Ops(*ct)[1] != d->count)
        return 0;
      return o[1] < d->count && TypeOf(Ops(*d)[o[1]]) == in.type
                 ? Ops(*d)[o[1]]
                 : 0;
    }
    case OpBitReverse:
    case OpBitCount: {
      if (KindOf(in.type) != kInt32 || !Known(o[0]) ||
          KindOf(TypeOf(o[0])) != kInt32)
        return 0;
      const u32 x = static_cast<u32>(bits_[o[0]]);
      u32 v = 0;
      for (u32 b = 0; b < 32; b++)
        if (x >> b & 1)
          v = in.op == OpBitCount ? v + 1 : v | 1u << (31 - b);
      return Constant(in.type, v);
    }
    case OpBitFieldUExtract:
    case OpBitFieldSExtract: {
      if (KindOf(in.type) != kInt32 || !Known(o[0]) || !Known(o[1]) ||
          !Known(o[2]))
        return 0;
      const u64 off = bits_[o[1]], cnt = bits_[o[2]];
      if (off > 32 || cnt > 32 || off + cnt > 32)
        return 0;
      if (cnt == 0)
        return Constant(in.type, 0);
      u32 v = static_cast<u32>(bits_[o[0]] >> off) &
              (cnt == 32 ? ~0u : (1u << cnt) - 1);
      if (in.op == OpBitFieldSExtract && cnt < 32 && (v >> (cnt - 1)) & 1)
        v |= ~0u << cnt;
      return Constant(in.type, v);
    }
    case OpUConvert:
    case OpSConvert: {
      const Kind to = KindOf(in.type), from = KindOf(TypeOf(o[0]));
      if (!Known(o[0]) || (to != kInt32 && to != kInt64) ||
          (from != kInt32 && from != kInt64))
        return 0;
      u64 v = bits_[o[0]];
      if (in.op == OpSConvert && from == kInt32)
        v = static_cast<u64>(static_cast<i64>(static_cast<i32>(v)));
      return Constant(in.type, to == kInt32 ? v & 0xffffffffull : v);
    }
    case OpConvertUToF:
    case OpConvertSToF: {
      if (KindOf(in.type) != kFloat32 || !Known(o[0]) ||
          KindOf(TypeOf(o[0])) != kInt32)
        return 0;
      const float f = in.op == OpConvertUToF
                          ? static_cast<float>(static_cast<u32>(bits_[o[0]]))
                          : static_cast<float>(static_cast<i32>(bits_[o[0]]));
      u32 b;
      std::memcpy(&b, &f, 4);
      return Constant(in.type, b);
    }
    case OpIAdd:
    case OpISub:
    case OpIMul:
    case OpUDiv:
    case OpUMod:
    case OpBitwiseAnd:
    case OpBitwiseOr:
    case OpBitwiseXor:
    case OpNot:
    case OpSNegate:
    case OpShiftLeftLogical:
    case OpShiftRightLogical:
    case OpShiftRightArithmetic:
    case OpIEqual:
    case OpINotEqual:
    case OpULessThan:
    case OpULessThanEqual:
    case OpUGreaterThan:
    case OpUGreaterThanEqual:
    case OpSLessThan:
    case OpSLessThanEqual:
    case OpSGreaterThan:
    case OpSGreaterThanEqual:
      return FoldInt(in, o);
    default:
      return 0;
  }
}

bool Ir::Pure(Inst& in) {
  using enum Op;
  if (in.op == OpExtInst) {
    const u32 ext = Ops(in)[1];
    // Modf/Frexp write through a pointer; InterpolateAt* read one.
    return Ops(in)[0] == glsl_ && ext != 35 && ext != 36 && ext != 51 &&
           ext != 52 && (ext < 76 || ext > 78) && ext <= 81;
  }
  if (in.op == OpLoad)
    return in.count == 1 && ReadOnly(Ops(in)[0]);
  return Numberable(in.op);
}

u32 Ir::Number(u32 idx) {
  using enum Op;
  Inst& in = insts_[idx];
  if (!Pure(in))
    return 0;
  u32* o = Ops(in);
  // Constants second, otherwise by id.
  if (Commutative(in.op) && in.count == 2 &&
      (Known(o[0]) != Known(o[1]) ? Known(o[0]) : o[0] > o[1])) {
    const u32 t = o[0];
    o[0] = o[1];
    o[1] = t;
  }
  u32 h = 2166136261u;
  h = (h ^ static_cast<u32>(in.op)) * 16777619u;
  h = (h ^ in.type) * 16777619u;
  for (u32 k = 0; k < in.count; k++)
    h = (h ^ o[k]) * 16777619u;
  for (u32 e = buckets_[h & bucket_mask_]; e != kNone; e = entries_[e].next) {
    if (entries_[e].hash != h)
      continue;
    const Inst& other = insts_[entries_[e].inst];
    if (other.op != in.op || other.type != in.type || other.count != in.count)
      continue;
    if (std::memcmp(Ops(other), o, in.count * 4) == 0)
      return other.result;
  }
  entries_.push_back({idx, h, buckets_[h & bucket_mask_]});
  buckets_[h & bucket_mask_] = static_cast<u32>(entries_.size() - 1);
  return 0;
}

void Ir::SimplifyFunction(Function& f) {
  const u32 n = static_cast<u32>(f.blocks.size());
  // Reverse postorder from the entry.
  base::Vector<u32> order, rpo(n, kNone), stack, cursor(n, 0);
  order.reserve(n);
  base::Vector<u8> seen(n, 0);
  stack.push_back(0);
  seen[0] = 1;
  while (!stack.empty()) {
    const u32 b = stack.back();
    if (cursor[b] < f.blocks[b].succs.size()) {
      const u32 s = f.blocks[b].succs[cursor[b]++];
      if (!seen[s]) {
        seen[s] = 1;
        stack.push_back(s);
      }
      continue;
    }
    order.push_back(b);
    stack.pop_back();
  }
  for (u32 i = 0; i < order.size() / 2; i++) {
    const u32 t = order[i];
    order[i] = order[order.size() - 1 - i];
    order[order.size() - 1 - i] = t;
  }
  for (u32 i = 0; i < order.size(); i++)
    rpo[order[i]] = i;
  // Cooper, Harvey and Kennedy.
  base::Vector<u32> idom(n, kNone);
  idom[0] = 0;
  for (bool changed = true; changed;) {
    changed = false;
    for (u32 i = 1; i < order.size(); i++) {
      const u32 b = order[i];
      u32 d = kNone;
      for (const u32 p : f.blocks[b].preds) {
        if (idom[p] == kNone)
          continue;
        if (d == kNone) {
          d = p;
          continue;
        }
        u32 x = p, y = d;
        while (x != y) {
          while (rpo[x] > rpo[y])
            x = idom[x];
          while (rpo[y] > rpo[x])
            y = idom[y];
        }
        d = x;
      }
      if (idom[b] != d) {
        idom[b] = d;
        changed = true;
      }
    }
  }
  base::Vector<u32> first(n + 1, 0), kids(order.size());
  for (u32 i = 1; i < order.size(); i++)
    first[idom[order[i]] + 1]++;
  for (u32 b = 0; b < n; b++)
    first[b + 1] += first[b];
  base::Vector<u32> at(first.begin(), first.end() - 1);
  for (u32 i = 1; i < order.size(); i++)
    kids[at[idom[order[i]]]++] = order[i];

  u32 insts = 0;
  for (const Block& b : f.blocks)
    insts += static_cast<u32>(b.body.size() + b.phis.size());
  u32 cap = 16;
  while (cap < insts)
    cap <<= 1;
  buckets_.assign(cap, kNone);
  bucket_mask_ = cap - 1;
  entries_.clear();

  const auto visit = [&](u32 idx) {
    if (dead_[idx])
      return;
    Inst& in = insts_[idx];
    ForEachId(in.op, Ops(in), in.count, [&](u32& id) { id = Find(id); });
    const u32 result = in.result;
    if (!result || !in.type)
      return;
    // Fold may add constants, moving insts_ and words_.
    u32 r = Fold(idx);
    if (!r)
      r = Number(idx);
    if (r) {
      repl_[result] = r;
      dead_[idx] = 1;
    }
  };
  struct Frame {
    u32 block, kid, mark;
  };
  base::Vector<Frame> walk;
  const auto enter = [&](u32 b) {
    walk.push_back({b, first[b], static_cast<u32>(entries_.size())});
    for (u32 i = 0; i < f.blocks[b].phis.size(); i++)
      visit(f.blocks[b].phis[i]);
    for (u32 i = 0; i < f.blocks[b].body.size(); i++)
      visit(f.blocks[b].body[i]);
  };
  enter(0);
  while (!walk.empty()) {
    Frame& top = walk.back();
    if (top.kid < first[top.block + 1]) {
      enter(kids[top.kid++]);
      continue;
    }
    while (entries_.size() > top.mark) {
      const Entry& e = entries_.back();
      buckets_[e.hash & bucket_mask_] = e.next;
      entries_.pop_back();
    }
    walk.pop_back();
  }
  Hoist(f, order, rpo, idom);
  FoldBranches(f);
}

// A conditional branch on a constant goes straight to the taken target.
// Blocks no longer reachable are dropped, except merge and continue targets
// a live construct still names: those keep a stub. Phis lose the incoming
// values of predecessors that are gone.
void Ir::FoldBranches(Function& f) {
  using enum Op;
  bool folded = false;
  for (Block& blk : f.blocks) {
    Inst& term = insts_[blk.body.back()];
    if (term.op != OpBranchConditional)
      continue;
    u32* o = Ops(term);
    const u32 cond = Find(o[0]);
    const u32 d = cond < def_.size() ? def_[cond] : kNone;
    u32 target;
    if (o[1] == o[2])
      target = o[1];
    else if (d != kNone && insts_[d].op == OpConstantTrue)
      target = o[1];
    else if (d != kNone && insts_[d].op == OpConstantFalse)
      target = o[2];
    else
      continue;
    term.op = OpBranch;
    term.count = 1;
    o[0] = target;
    const u32 n = static_cast<u32>(blk.body.size());
    if (n >= 2 && insts_[blk.body[n - 2]].op == OpSelectionMerge) {
      dead_[blk.body[n - 2]] = 1;
      blk.body.erase(blk.body.begin() + (n - 2));
    }
    folded = true;
  }
  if (!folded)
    return;

  base::Vector<u8> reach(f.blocks.size(), 0);
  base::Vector<u32> work{0};
  reach[0] = 1;
  while (!work.empty()) {
    const Inst& t = insts_[f.blocks[work.back()].body.back()];
    work.pop_back();
    const u32* o = Ops(t);
    const auto visit = [&](u32 label) {
      const u32 s = block_of_[label];
      if (!reach[s]) {
        reach[s] = 1;
        work.push_back(s);
      }
    };
    if (t.op == OpBranch)
      visit(o[0]);
    else if (t.op == OpBranchConditional) {
      visit(o[1]);
      visit(o[2]);
    } else if (t.op == OpSwitch) {
      visit(o[1]);
      for (u32 i = 3; i < t.count; i += 2)
        visit(o[i]);
    }
  }
  // Merge and continue targets named by reachable headers survive as stubs.
  base::Vector<u32> stub(f.blocks.size(), kNone);
  for (u32 b = 0; b < f.blocks.size(); b++) {
    if (!reach[b] || f.blocks[b].body.size() < 2)
      continue;
    const Inst& m = insts_[f.blocks[b].body[f.blocks[b].body.size() - 2]];
    if (m.op != OpSelectionMerge && m.op != OpLoopMerge)
      continue;
    const u32 merge = block_of_[Ops(m)[0]];
    if (!reach[merge])
      stub[merge] = kNone - 1;
    if (m.op == OpLoopMerge && !reach[block_of_[Ops(m)[1]]])
      stub[block_of_[Ops(m)[1]]] = b;
  }
  base::Vector<Block> kept;
  for (u32 b = 0; b < f.blocks.size(); b++) {
    Block& blk = f.blocks[b];
    if (!reach[b] && stub[b] == kNone) {
      for (const u32 idx : blk.body)
        dead_[idx] = 1;
      for (const u32 idx : blk.phis)
        dead_[idx] = 1;
      continue;
    }
    if (!reach[b]) {
      for (const u32 idx : blk.body)
        dead_[idx] = 1;
      for (const u32 idx : blk.phis)
        dead_[idx] = 1;
      blk.phis.clear();
      blk.body.clear();
      if (stub[b] == kNone - 1) {
        blk.body.push_back(AddInst(OpUnreachable, 0, 0, nullptr, 0));
      } else {
        const u32 header = f.blocks[stub[b]].label;
        blk.body.push_back(AddInst(OpBranch, 0, 0, &header, 1));
      }
    }
    kept.push_back(base::move(blk));
  }
  f.blocks = base::move(kept);
  for (Block& blk : f.blocks) {
    blk.preds.clear();
    blk.succs.clear();
  }
  BuildCfg(f);

  const auto trim = [&](u32 idx, const Block& blk) {
    Inst& in = insts_[idx];
    if (dead_[idx] || in.op != OpPhi)
      return;
    u32* o = Ops(in);
    u32 n = 0;
    for (u32 k = 0; k + 1 < in.count; k += 2) {
      const u32 from = block_of_[o[k + 1]];
      bool pred = false;
      for (const u32 p : blk.preds)
        pred |= p == from && f.blocks[p].label == o[k + 1];
      if (pred) {
        o[n] = o[k];
        o[n + 1] = o[k + 1];
        n += 2;
      }
    }
    in.count = n;
    if (n == 2) {
      repl_[in.result] = Find(o[0]);
      dead_[idx] = 1;
    }
  };
  for (Block& blk : f.blocks) {
    for (const u32 idx : blk.phis)
      trim(idx, blk);
    for (const u32 idx : blk.body)
      trim(idx, blk);
  }
}

// Code hoisting: an expression computed again where the first computation
// does not dominate (each in its own branch arm, say) moves to the nearest
// common dominator of the two and serves both. Its operands dominate both
// places, hence that block too. Only operations without side effects or
// undefined behaviour move, and only when a second copy exists. Nothing moves
// above a switch: in a dispatch loop that would run every case's share on
// every iteration.
void Ir::Hoist(Function& f,
               const base::Vector<u32>& order,
               const base::Vector<u32>& rpo,
               const base::Vector<u32>& idom) {
  using enum Op;
  const auto movable = [&](Inst& in) {
    switch (in.op) {
      case OpPhi:
      case OpUDiv:
      case OpSDiv:
      case OpUMod:
      case OpSRem:
      case OpSMod:
      case OpFRem:
      case OpFMod:
        return false;
      default:
        return Pure(in);
    }
  };
  base::Vector<u32> depth(f.blocks.size(), 0);
  for (u32 i = 1; i < order.size(); i++)
    depth[order[i]] = depth[idom[order[i]]] + 1;
  const auto ncd = [&](u32 a, u32 b) {
    while (a != b) {
      if (depth[a] > depth[b])
        a = idom[a];
      else
        b = idom[b];
    }
    return a;
  };
  const auto dominates = [&](u32 a, u32 b) {
    while (depth[b] > depth[a])
      b = idom[b];
    return a == b;
  };
  struct Seen {
    u32 inst, block, hash, next;
  };
  base::Vector<Seen> seen;
  base::Vector<u32> heads(buckets_.size(), kNone);
  const u32 mask = static_cast<u32>(heads.size() - 1);
  for (const u32 b : order) {
    Block& blk = f.blocks[b];
    for (u32 i = 0; i < blk.body.size(); i++) {
      const u32 idx = blk.body[i];
      if (dead_[idx])
        continue;
      Inst& in = insts_[idx];
      if (!in.result || !in.type || !movable(in))
        continue;
      u32* o = Ops(in);
      ForEachId(in.op, o, in.count, [&](u32& id) { id = Find(id); });
      u32 h = 2166136261u;
      h = (h ^ static_cast<u32>(in.op)) * 16777619u;
      h = (h ^ in.type) * 16777619u;
      for (u32 k = 0; k < in.count; k++)
        h = (h ^ o[k]) * 16777619u;
      u32 match = kNone;
      for (u32 e = heads[h & mask]; e != kNone; e = seen[e].next) {
        const Inst& other = insts_[seen[e].inst];
        if (seen[e].hash == h && other.op == in.op && other.type == in.type &&
            other.count == in.count && !dead_[seen[e].inst] &&
            std::memcmp(Ops(other), o, in.count * 4) == 0) {
          match = e;
          break;
        }
      }
      if (match == kNone) {
        seen.push_back({idx, b, h, heads[h & mask]});
        heads[h & mask] = static_cast<u32>(seen.size() - 1);
        continue;
      }
      Seen& first = seen[match];
      if (!dominates(first.block, b)) {
        const u32 to = ncd(first.block, b);
        if (insts_[f.blocks[to].body.back()].op == OpSwitch)
          continue;
        // Move the first copy to the end of the common dominator, ahead of
        // its merge instruction and terminator.
        base::Vector<u32>& from_body = f.blocks[first.block].body;
        for (u32 k = 0; k < from_body.size(); k++)
          if (from_body[k] == first.inst) {
            from_body.erase(from_body.begin() + k);
            if (first.block == b && k < i)
              i--;
            break;
          }
        base::Vector<u32>& to_body = f.blocks[to].body;
        u32 at = static_cast<u32>(to_body.size()) - 1;
        if (at > 0) {
          const Op prev = insts_[to_body[at - 1]].op;
          if (prev == OpSelectionMerge || prev == OpLoopMerge)
            at--;
        }
        to_body.insert(to_body.begin() + at, first.inst);
        if (to == b && at <= i)
          i++;
        first.block = to;
      }
      repl_[in.result] = insts_[first.inst].result;
      dead_[idx] = 1;
    }
  }
}

void Ir::Simplify() {
  for (Function& f : fns_)
    SimplifyFunction(f);
}

void Ir::Sweep() {
  using enum Op;
  live_.assign(insts_.size(), 0);
  base::Vector<u32> work;
  work.reserve(insts_.size() / 2);
  const auto keep = [&](u32 idx) {
    if (!live_[idx]) {
      live_[idx] = 1;
      work.push_back(idx);
    }
  };
  for (const u32 g : globals_) {
    const Inst& in = insts_[g];
    switch (in.op) {
      case OpVariable: {
        const u32 sc = Ops(in)[0];
        if (sc != static_cast<u32>(spv::StorageClass::Private) &&
            sc != static_cast<u32>(spv::StorageClass::Function))
          keep(g);
        break;
      }
      case OpConstant:
      case OpConstantTrue:
      case OpConstantFalse:
      case OpConstantNull:
      case OpConstantComposite:
      case OpUndef:
      case OpDecorate:
      case OpMemberDecorate:
      case OpDecorateString:
      case OpMemberDecorateString:
      case OpName:
      case OpMemberName:
        break;
      case OpEntryPoint:
        live_[g] = 1;
        break;
      default:
        if (!IsType(in.op))
          keep(g);
    }
  }
  for (const Function& f : fns_) {
    keep(f.inst);
    for (const u32 p : f.params)
      keep(p);
    keep(f.end);
    for (const Block& b : f.blocks) {
      live_[b.label_inst] = 1;
      for (const u32 idx : b.body)
        if (!dead_[idx] && HasSideEffects(insts_[idx].op))
          keep(idx);
    }
  }
  while (!work.empty()) {
    const u32 idx = work.back();
    work.pop_back();
    Inst& in = insts_[idx];
    if (in.type && def_[in.type] != kNone)
      keep(def_[in.type]);
    ForEachId(in.op, Ops(in), in.count, [&](u32& id) {
      id = Find(id);
      if (id < def_.size() && def_[id] != kNone)
        keep(def_[id]);
    });
  }
}

void Ir::Write(base::Vector<u32>* out) const {
  using enum Op;
  out->clear();
  out->reserve(words_.size() + insts_.size() * 3);
  out->push_back(header_[0]);
  out->push_back(header_[1]);
  out->push_back(header_[2]);
  out->push_back(bound_);
  out->push_back(header_[4]);
  const auto live_id = [&](u32 id) {
    return id < def_.size() && def_[id] != kNone && live_[def_[id]];
  };
  const auto put = [&](const Inst& in, const u32* ops, u32 count) {
    const u32 wc = 1 + (in.type != 0) + (in.result != 0) + count;
    out->push_back((wc << 16) | static_cast<u32>(in.op));
    if (in.type)
      out->push_back(in.type);
    if (in.result)
      out->push_back(in.result);
    out->insert(out->end(), ops, ops + count);
  };
  const auto emit = [&](u32 idx) {
    if (live_[idx])
      put(insts_[idx], Ops(insts_[idx]), insts_[idx].count);
  };
  for (const u32 g : globals_) {
    const Inst& in = insts_[g];
    switch (in.op) {
      case OpDecorate:
      case OpMemberDecorate:
      case OpDecorateString:
      case OpMemberDecorateString:
      case OpName:
      case OpMemberName:
        if (in.count && live_id(Ops(in)[0]))
          put(in, Ops(in), in.count);
        break;
      case OpEntryPoint: {
        const u32* o = Ops(in);
        const u32 fixed =
            2 + StringWords(o + 2, in.count > 2 ? in.count - 2 : 0);
        base::Vector<u32> ops;
        for (u32 k = 0; k < in.count; k++)
          if (k < fixed || live_id(o[k]))
            ops.push_back(o[k]);
        put(in, ops.data(), static_cast<u32>(ops.size()));
        break;
      }
      default:
        emit(g);
    }
  }
  for (const Function& f : fns_) {
    emit(f.inst);
    for (const u32 p : f.params)
      emit(p);
    for (const Block& b : f.blocks) {
      emit(b.label_inst);
      for (const u32 idx : b.phis)
        emit(idx);
      for (const u32 idx : b.body)
        emit(idx);
    }
    emit(f.end);
  }
}

}  // namespace

bool OptimizeModule(const base::Vector<u32>& in,
                    base::Vector<u32>* out,
                    u32 level,
                    base::String* err) {
  Ir ir;
  if (!ir.Parse(in, err) || (err && !ir.Check(err)))
    return false;
  if (!ir.Promote()) {
    BASE_LOGW("spv-opt", "module not optimized: unmodeled instruction");
    *out = in;
    return true;
  }
  if (level >= 2)
    ir.Simplify();
  ir.Sweep();
  ir.Write(out);
  return true;
}

bool CheckModule(const base::Vector<u32>& in, base::String* err) {
  Ir ir;
  return ir.Parse(in, err) && ir.Check(err);
}

}  // namespace gpu::gcn::spirv

#endif
