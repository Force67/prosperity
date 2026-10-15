/*
 * PS4Delta : PS4 emulation and research project
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "gpu/gcn/spirv/spv_emit.h"
#include "base/arch.h"

#include <cstring>
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/memory/move.h"
#include "base/strings/xstring.h"

namespace gpu::gcn::spirv {

void Module::PutString(base::Vector<u32>& sec, const base::String& s) {
  u32 w = 0;
  int b = 0;
  const auto push = [&](u8 c) {
    w |= static_cast<u32>(c) << (8 * b);
    if (++b == 4) {
      sec.push_back(w);
      w = 0;
      b = 0;
    }
  };
  for (char c : s)
    push(static_cast<u8>(c));
  push(0);
  if (b != 0)
    sec.push_back(w);
}

void Module::Instr(base::Vector<u32>& sec,
                   spv::Op op,
                   const base::Vector<u32>& ops) {
  const u32 wc = 1 + static_cast<u32>(ops.size());
  sec.push_back((wc << 16) | static_cast<u32>(op));
  sec.insert(sec.end(), ops.begin(), ops.end());
}

Module::Module() {
  Instr(caps_, spv::Op::OpCapability,
        {static_cast<u32>(spv::Capability::Shader)});
  glsl_ext_ = Alloc();
  base::Vector<u32> ops{glsl_ext_};
  PutString(ops, "GLSL.std.450");
  Instr(ext_imports_, spv::Op::OpExtInstImport, ops);
  Instr(mem_model_, spv::Op::OpMemoryModel,
        {static_cast<u32>(spv::AddressingModel::Logical),
         static_cast<u32>(spv::MemoryModel::GLSL450)});
}

Id Module::TypeVoid() {
  const u64 k = Key(CacheKind::kVoid);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeVoid, {id});
  return Cached(k, id);
}
Id Module::TypeBool() {
  const u64 k = Key(CacheKind::kBool);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeBool, {id});
  return Cached(k, id);
}
Id Module::TypeInt(u32 width, bool sign) {
  const u64 k = Key(CacheKind::kInt, width, sign ? 1 : 0);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeInt, {id, width, sign ? 1u : 0u});
  return Cached(k, id);
}
Id Module::TypeFloat(u32 width) {
  const u64 k = Key(CacheKind::kFloat, width);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeFloat, {id, width});
  return Cached(k, id);
}
Id Module::TypeVec(Id comp, u32 count) {
  const u64 k = Key(CacheKind::kVec, comp, count);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeVector, {id, comp, count});
  return Cached(k, id);
}
Id Module::TypeArray(Id elem, u32 len) {
  const Id len_id = ConstU32(len);
  const u64 k = Key(CacheKind::kArray, elem, len);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeArray, {id, elem, len_id});
  return Cached(k, id);
}
Id Module::TypeRuntimeArray(Id elem) {
  const u64 k = Key(CacheKind::kRuntimeArray, elem);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeRuntimeArray, {id, elem});
  return Cached(k, id);
}
Id Module::TypeStruct(const base::Vector<Id>& members) {
  auto it = struct_cache_.find(members);
  if (it != struct_cache_.end())
    return it->second;
  const Id id = Alloc();
  base::Vector<u32> ops{id};
  ops.insert(ops.end(), members.begin(), members.end());
  Instr(types_consts_, spv::Op::OpTypeStruct, ops);
  struct_cache_[members] = id;
  return id;
}
Id Module::TypePointer(spv::StorageClass sc, Id pointee) {
  const u64 k = Key(CacheKind::kPointer, pointee, static_cast<u64>(sc));
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypePointer,
        {id, static_cast<u32>(sc), pointee});
  return Cached(k, id);
}
Id Module::TypeFunction(Id ret, const base::Vector<Id>& params) {
  base::Vector<Id> key{ret};
  key.insert(key.end(), params.begin(), params.end());
  auto it = fn_type_cache_.find(key);
  if (it != fn_type_cache_.end())
    return it->second;
  const Id id = Alloc();
  base::Vector<u32> ops{id, ret};
  ops.insert(ops.end(), params.begin(), params.end());
  Instr(types_consts_, spv::Op::OpTypeFunction, ops);
  fn_type_cache_[key] = id;
  return id;
}
Id Module::TypeImage(Id sampled_type,
                     spv::Dim dim,
                     u32 depth,
                     u32 arrayed,
                     u32 ms,
                     u32 sampled,
                     spv::ImageFormat fmt) {
  const Id id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeImage,
        {id, sampled_type, static_cast<u32>(dim), depth, arrayed, ms, sampled,
         static_cast<u32>(fmt)});
  return id;
}
Id Module::TypeSampler() {
  if (!sampler_type_) {
    sampler_type_ = Alloc();
    Instr(types_consts_, spv::Op::OpTypeSampler, {sampler_type_});
  }
  return sampler_type_;
}

Id Module::TypeSampledImage(Id image_type) {
  const Id id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeSampledImage, {id, image_type});
  return id;
}

Id Module::ConstU32(u32 v) {
  const u64 k = Key(CacheKind::kConstU32, v);
  Id id;
  if (Lookup(k, id))
    return id;
  const Id t = TypeInt(32, false);
  id = Alloc();
  Instr(types_consts_, spv::Op::OpConstant, {t, id, v});
  NoteType(id, t);
  const_bits_[id] = v;
  return Cached(k, id);
}
Id Module::ConstI32(i32 v) {
  const u64 k = Key(CacheKind::kConstI32, static_cast<u32>(v));
  Id id;
  if (Lookup(k, id))
    return id;
  const Id t = TypeInt(32, true);
  id = Alloc();
  Instr(types_consts_, spv::Op::OpConstant, {t, id, static_cast<u32>(v)});
  NoteType(id, t);
  const_bits_[id] = static_cast<u32>(v);
  return Cached(k, id);
}
Id Module::ConstF32(float v) {
  u32 bits;
  std::memcpy(&bits, &v, 4);
  const u64 k = Key(CacheKind::kConstF32, bits);
  Id id;
  if (Lookup(k, id))
    return id;
  const Id t = TypeFloat(32);
  id = Alloc();
  Instr(types_consts_, spv::Op::OpConstant, {t, id, bits});
  NoteType(id, t);
  const_bits_[id] = bits;
  return Cached(k, id);
}
Id Module::ConstBool(bool v) {
  const u64 k = Key(CacheKind::kConstBool, v ? 1 : 0);
  Id id;
  if (Lookup(k, id))
    return id;
  const Id t = TypeBool();
  id = Alloc();
  Instr(types_consts_, v ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse,
        {t, id});
  NoteType(id, t);
  const_bools_[id] = v;
  return Cached(k, id);
}
Id Module::ConstComposite(Id type, const base::Vector<Id>& parts) {
  const Id id = Alloc();
  base::Vector<u32> ops{type, id};
  ops.insert(ops.end(), parts.begin(), parts.end());
  Instr(types_consts_, spv::Op::OpConstantComposite, ops);
  return id;
}
Id Module::ConstNull(Id type) {
  const u64 k = Key(CacheKind::kConstNull, type);
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpConstantNull, {type, id});
  return Cached(k, id);
}

Id Module::Variable(Id ptr_type, spv::StorageClass sc, Id init) {
  const Id id = Alloc();
  base::Vector<u32> ops{ptr_type, id, static_cast<u32>(sc)};
  if (init)
    ops.push_back(init);
  Instr(types_consts_, spv::Op::OpVariable, ops);
  globals_.push_back(id);
  return id;
}
void Module::Decorate(Id target,
                      spv::Decoration dec,
                      const base::Vector<u32>& operands) {
  if (operands.empty() &&
      !decorated_
           .insert((static_cast<u64>(target) << 32) | static_cast<u32>(dec))
           .second)
    return;
  base::Vector<u32> ops{target, static_cast<u32>(dec)};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(decos_, spv::Op::OpDecorate, ops);
}
void Module::MemberDecorate(Id struct_type,
                            u32 member,
                            spv::Decoration dec,
                            const base::Vector<u32>& operands) {
  base::Vector<u32> ops{struct_type, member, static_cast<u32>(dec)};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(decos_, spv::Op::OpMemberDecorate, ops);
}
void Module::Name(Id target, const base::String& n) {
  base::Vector<u32> ops{target};
  PutString(ops, n);
  Instr(debug_, spv::Op::OpName, ops);
}

Id Module::String(const base::String& s) {
  const Id id = Alloc();
  base::Vector<u32> ops{id};
  PutString(ops, s);
  Instr(strings_, spv::Op::OpString, ops);
  return id;
}

void Module::Line(Id file, u32 line) {
  Instr(fn_body_, spv::Op::OpLine, {file, line, 0});
}

void Module::EntryPoint(spv::ExecutionModel model,
                        Id fn,
                        const base::String& n,
                        const base::Vector<Id>& interface) {
  base::Vector<u32> ops{static_cast<u32>(model), fn};
  PutString(ops, n);
  (void)interface;
  ops.insert(ops.end(), globals_.begin(), globals_.end());
  Instr(entries_, spv::Op::OpEntryPoint, ops);
}
void Module::ExecMode(Id fn,
                      spv::ExecutionMode mode,
                      const base::Vector<u32>& operands) {
  base::Vector<u32> ops{fn, static_cast<u32>(mode)};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(exec_modes_, spv::Op::OpExecutionMode, ops);
}
void Module::Capability(spv::Capability cap) {
  Instr(caps_, spv::Op::OpCapability, {static_cast<u32>(cap)});
}

void Module::Extension(const base::String& name) {
  base::Vector<u32> ops;
  PutString(ops, name);
  Instr(exts_, spv::Op::OpExtension, ops);
}

void Module::PhysicalStorageBuffers() {
  Capability(spv::Capability::Int64);
  Capability(spv::Capability::PhysicalStorageBufferAddresses);
  Extension("SPV_KHR_physical_storage_buffer");
  mem_model_.clear();
  Instr(mem_model_, spv::Op::OpMemoryModel,
        {static_cast<u32>(spv::AddressingModel::PhysicalStorageBuffer64),
         static_cast<u32>(spv::MemoryModel::GLSL450)});
}

Id Module::BeginFunction(Id ret_type,
                         Id fn_type,
                         const base::Vector<Id>& parameter_types,
                         base::Vector<Id>* parameter_ids,
                         spv::FunctionControlMask control) {
  const Id fn = Alloc();
  Instr(fn_body_, spv::Op::OpFunction,
        {ret_type, fn, static_cast<u32>(control), fn_type});
  for (Id type : parameter_types) {
    const Id id = Alloc();
    Instr(fn_body_, spv::Op::OpFunctionParameter, {type, id});
    if (parameter_ids)
      parameter_ids->push_back(id);
  }
  const Id entry = Alloc();
  Instr(fn_body_, spv::Op::OpLabel, {entry});
  cur_block_ = entry;
  return fn;
}
Id Module::NewBlock() {
  return Alloc();
}
void Module::OpenBlock(Id label) {
  Instr(fn_body_, spv::Op::OpLabel, {label});
  cur_block_ = label;
}
void Module::EndFunction() {
  Instr(fn_body_, spv::Op::OpFunctionEnd, {});
  cur_block_ = 0;
}

namespace {
bool IsPure(spv::Op op) {
  switch (op) {
    case spv::Op::OpIAdd:
    case spv::Op::OpISub:
    case spv::Op::OpIMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpUMod:
    case spv::Op::OpSRem:
    case spv::Op::OpSMod:
    case spv::Op::OpFAdd:
    case spv::Op::OpFSub:
    case spv::Op::OpFMul:
    case spv::Op::OpFDiv:
    case spv::Op::OpFRem:
    case spv::Op::OpFMod:
    case spv::Op::OpFNegate:
    case spv::Op::OpSNegate:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpNot:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpBitFieldInsert:
    case spv::Op::OpBitFieldSExtract:
    case spv::Op::OpBitFieldUExtract:
    case spv::Op::OpBitReverse:
    case spv::Op::OpBitCount:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpSLessThan:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpFOrdEqual:
    case spv::Op::OpFOrdNotEqual:
    case spv::Op::OpFOrdLessThan:
    case spv::Op::OpFOrdGreaterThan:
    case spv::Op::OpFOrdLessThanEqual:
    case spv::Op::OpFOrdGreaterThanEqual:
    case spv::Op::OpFUnordEqual:
    case spv::Op::OpFUnordNotEqual:
    case spv::Op::OpFUnordLessThan:
    case spv::Op::OpFUnordGreaterThan:
    case spv::Op::OpFUnordLessThanEqual:
    case spv::Op::OpFUnordGreaterThanEqual:
    case spv::Op::OpIsNan:
    case spv::Op::OpIsInf:
    case spv::Op::OpLogicalEqual:
    case spv::Op::OpLogicalNotEqual:
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpLogicalNot:
    case spv::Op::OpSelect:
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpQuantizeToF16:
    case spv::Op::OpBitcast:
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpVectorExtractDynamic:
    case spv::Op::OpVectorInsertDynamic:
    case spv::Op::OpVectorTimesScalar:
    case spv::Op::OpDot:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
      return true;
    default:
      return false;
  }
}

}  // namespace

Id Module::Reuse(const u32 (&head)[2],
                 const u32* rest,
                 size_t count,
                 u64* key) const {
  u64 h = 1469598103934665603ull;
  h = (h ^ head[0]) * 1099511628211ull;
  h = (h ^ head[1]) * 1099511628211ull;
  for (size_t i = 0; i < count; i++)
    h = (h ^ rest[i]) * 1099511628211ull;
  *key = h;
  const auto it = computed_.find(h);
  if (it == computed_.end() || it->second.block != cur_block_)
    return 0;
  const u32* inst = fn_body_.data() + it->second.offset;
  const u32 wc = static_cast<u32>(count) + 3;
  if (inst[0] != ((wc << 16) | head[0]) || inst[1] != head[1])
    return 0;
  for (size_t i = 0; i < count; i++)
    if (inst[3 + i] != rest[i])
      return 0;
  return it->second.id;
}

void Module::Remember(u64 key, Id id, u32 offset) {
  if (computed_block_ != cur_block_) {
    computed_.clear();
    computed_block_ = cur_block_;
  }
  computed_[key] = {cur_block_, id, offset};
}

Id Module::Fold(spv::Op op, Id result_type, const base::Vector<Id>& operands) {
  switch (op) {
    case spv::Op::OpSelect:
      if (operands.size() != 3)
        return 0;
      if (operands[1] == operands[2])
        return operands[1];
      if (const auto c = const_bools_.find(operands[0]);
          c != const_bools_.end())
        return c->second ? operands[1] : operands[2];
      return 0;
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpLogicalNot: {
      const auto a = const_bools_.find(operands[0]);
      const bool ka = a != const_bools_.end();
      if (op == spv::Op::OpLogicalNot)
        return ka ? ConstBool(!a->second) : 0;
      const auto b = const_bools_.find(operands[1]);
      const bool kb = b != const_bools_.end();
      const bool absorbing = op == spv::Op::OpLogicalOr;
      if ((ka && a->second == absorbing) || (kb && b->second == absorbing))
        return ConstBool(absorbing);
      if (ka)
        return operands[1];
      if (kb)
        return operands[0];
      return 0;
    }
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSLessThan:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpIAdd:
    case spv::Op::OpISub:
    case spv::Op::OpIMul:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpNot:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
      break;
    default:
      return 0;
  }
  if (!t_u32_) {
    t_u32_ = TypeInt(32, false);
    t_i32_ = TypeInt(32, true);
    t_bool_ = TypeBool();
  }
  const bool u = result_type == t_u32_;
  const bool i = result_type == t_i32_;
  const bool b = result_type == t_bool_;
  if ((!u && !i && !b) || operands.empty() || operands.size() > 2)
    return 0;
  u32 x = 0, y = 0;
  const auto cx = const_bits_.find(operands[0]);
  if (cx == const_bits_.end())
    return 0;
  x = cx->second;
  if (operands.size() == 2) {
    const auto cy = const_bits_.find(operands[1]);
    if (cy == const_bits_.end())
      return 0;
    y = cy->second;
  }
  const auto make = [&](u32 v) {
    return u ? ConstU32(v) : ConstI32(static_cast<i32>(v));
  };
  const i32 sx = static_cast<i32>(x), sy = static_cast<i32>(y);
  if (b) {
    switch (op) {
      case spv::Op::OpIEqual:
        return ConstBool(x == y);
      case spv::Op::OpINotEqual:
        return ConstBool(x != y);
      case spv::Op::OpULessThan:
        return ConstBool(x < y);
      case spv::Op::OpULessThanEqual:
        return ConstBool(x <= y);
      case spv::Op::OpUGreaterThan:
        return ConstBool(x > y);
      case spv::Op::OpUGreaterThanEqual:
        return ConstBool(x >= y);
      case spv::Op::OpSLessThan:
        return ConstBool(sx < sy);
      case spv::Op::OpSLessThanEqual:
        return ConstBool(sx <= sy);
      case spv::Op::OpSGreaterThan:
        return ConstBool(sx > sy);
      case spv::Op::OpSGreaterThanEqual:
        return ConstBool(sx >= sy);
      default:
        return 0;
    }
  }
  switch (op) {
    case spv::Op::OpIAdd:
      return make(x + y);
    case spv::Op::OpISub:
      return make(x - y);
    case spv::Op::OpIMul:
      return make(x * y);
    case spv::Op::OpBitwiseAnd:
      return make(x & y);
    case spv::Op::OpBitwiseOr:
      return make(x | y);
    case spv::Op::OpBitwiseXor:
      return make(x ^ y);
    case spv::Op::OpNot:
      return make(~x);
    case spv::Op::OpShiftLeftLogical:
      return y < 32 ? make(x << y) : 0;
    case spv::Op::OpShiftRightLogical:
      return y < 32 ? make(x >> y) : 0;
    case spv::Op::OpShiftRightArithmetic:
      return y < 32 ? make(static_cast<u32>(sx >> y)) : 0;
    default:
      return 0;
  }
}

Id Module::Emit(spv::Op op, Id result_type, const base::Vector<Id>& operands) {
  if (const Id folded = Fold(op, result_type, operands))
    return folded;
  const bool pure = cur_block_ && IsPure(op);
  u64 key = 0;
  if (pure)
    if (const Id id = Reuse({static_cast<u32>(op), result_type},
                            operands.data(), operands.size(), &key))
      return id;
  const Id id = Alloc();
  const u32 offset = static_cast<u32>(fn_body_.size());
  fn_body_.push_back(((static_cast<u32>(operands.size()) + 3) << 16) |
                     static_cast<u32>(op));
  fn_body_.push_back(result_type);
  fn_body_.push_back(id);
  fn_body_.insert(fn_body_.end(), operands.begin(), operands.end());
  NoteType(id, result_type);
  if (pure)
    Remember(key, id, offset);
  return id;
}
void Module::EmitVoid(spv::Op op, const base::Vector<Id>& operands) {
  Instr(fn_body_, op, operands);
}
Id Module::ExtInst(Id result_type,
                   u32 glsl_op,
                   const base::Vector<Id>& operands) {
  const bool pure =
      cur_block_ && glsl_op != 35 && glsl_op != 51 && glsl_op < 76;
  base::Vector<u32> ops{result_type, 0, glsl_ext_, glsl_op};
  ops.insert(ops.end(), operands.begin(), operands.end());
  u64 key = 0;
  if (pure)
    if (const Id id = Reuse({static_cast<u32>(spv::Op::OpExtInst), result_type},
                            ops.data() + 2, ops.size() - 2, &key))
      return id;
  const Id id = Alloc();
  ops[1] = id;
  const u32 offset = static_cast<u32>(fn_body_.size());
  Instr(fn_body_, spv::Op::OpExtInst, ops);
  NoteType(id, result_type);
  if (pure)
    Remember(key, id, offset);
  return id;
}

Id Module::Load(Id type, Id ptr) {
  return Emit(spv::Op::OpLoad, type, {ptr});
}
void Module::Store(Id ptr, Id value) {
  EmitVoid(spv::Op::OpStore, {ptr, value});
}
Id Module::AccessChain(Id ptr_type, Id base, const base::Vector<Id>& indices) {
  base::Vector<Id> ops{base};
  ops.insert(ops.end(), indices.begin(), indices.end());
  return Emit(spv::Op::OpAccessChain, ptr_type, ops);
}
Id Module::Bitcast(Id type, Id value) {
  if (TypeOf(value) == type)
    return value;
  if (const auto src = bitcast_source_.find(value);
      src != bitcast_source_.end()) {
    value = src->second;
    if (TypeOf(value) == type)
      return value;
  }
  if (const auto bits = const_bits_.find(value); bits != const_bits_.end()) {
    if (type == TypeInt(32, false))
      return ConstU32(bits->second);
    if (type == TypeInt(32, true))
      return ConstI32(static_cast<i32>(bits->second));
    if (type == TypeFloat(32)) {
      float f;
      std::memcpy(&f, &bits->second, 4);
      return ConstF32(f);
    }
  }
  const Id id = Emit(spv::Op::OpBitcast, type, {value});
  bitcast_source_[id] = value;
  return id;
}
Id Module::CompositeExtract(Id type, Id composite, u32 index) {
  const u32 rest[2] = {composite, index};
  u64 key = 0;
  if (cur_block_)
    if (const Id id =
            Reuse({static_cast<u32>(spv::Op::OpCompositeExtract), type}, rest,
                  2, &key))
      return id;
  const Id id = Alloc();
  const u32 offset = static_cast<u32>(fn_body_.size());
  Instr(fn_body_, spv::Op::OpCompositeExtract, {type, id, composite, index});
  NoteType(id, type);
  if (cur_block_)
    Remember(key, id, offset);
  return id;
}
Id Module::CompositeConstruct(Id type, const base::Vector<Id>& parts) {
  return Emit(spv::Op::OpCompositeConstruct, type, parts);
}

void Module::SelectionMerge(Id merge_block) {
  EmitVoid(
      spv::Op::OpSelectionMerge,
      {merge_block, static_cast<u32>(spv::SelectionControlMask::MaskNone)});
}
void Module::LoopMerge(Id merge_block, Id continue_block) {
  EmitVoid(spv::Op::OpLoopMerge,
           {merge_block, continue_block,
            static_cast<u32>(spv::LoopControlMask::MaskNone)});
}
void Module::Branch(Id target) {
  EmitVoid(spv::Op::OpBranch, {target});
}
void Module::BranchConditional(Id cond, Id t, Id f) {
  EmitVoid(spv::Op::OpBranchConditional, {cond, t, f});
}
void Module::Switch(Id selector,
                    Id default_label,
                    const base::Vector<base::Pair<u32, Id>>& cases) {
  base::Vector<u32> ops{selector, default_label};
  for (const auto& c : cases) {
    ops.push_back(c.first);
    ops.push_back(c.second);
  }
  EmitVoid(spv::Op::OpSwitch, ops);
}
void Module::ReturnVoid() {
  EmitVoid(spv::Op::OpReturn, {});
}
void Module::Unreachable() {
  EmitVoid(spv::Op::OpUnreachable, {});
}
void Module::Kill() {
  EmitVoid(spv::Op::OpKill, {});
}

base::Vector<u32> Module::Assemble() const {
  base::Vector<u32> out;
  out.push_back(spv::MagicNumber);
  out.push_back(version_);
  out.push_back(0);
  out.push_back(bound_);
  out.push_back(0);
  const auto append = [&](const base::Vector<u32>& s) {
    out.insert(out.end(), s.begin(), s.end());
  };
  append(caps_);
  append(exts_);
  append(ext_imports_);
  append(mem_model_);
  append(entries_);
  append(exec_modes_);
  append(strings_);
  append(debug_);
  append(decos_);
  append(types_consts_);
  append(fn_body_);
  return out;
}

}  // namespace gpu::gcn::spirv

#endif
