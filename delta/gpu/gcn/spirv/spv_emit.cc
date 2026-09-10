/*
 * PS4Delta : PS4 emulation and research project
 *
 * Minimal SPIR-V module builder. See spv_emit.h.
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "gpu/gcn/spirv/spv_emit.h"
#include "base/arch.h"

#include <cstring>

namespace gpu::gcn::spirv {

// Pack a literal string into SPIR-V words (LSB-first, null-terminated, zero
// padded to a word boundary).
void Module::PutString(std::vector<u32>& sec, const std::string& s) {
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
  push(0);  // null terminator
  if (b != 0)
    sec.push_back(w);  // flush trailing partial word (incl. padding)
}

void Module::Instr(std::vector<u32>& sec,
                   spv::Op op,
                   const std::vector<u32>& ops) {
  const u32 wc = 1 + static_cast<u32>(ops.size());
  sec.push_back((wc << 16) | static_cast<u32>(op));
  sec.insert(sec.end(), ops.begin(), ops.end());
}

Module::Module() {
  Instr(caps_, spv::Op::OpCapability,
        {static_cast<u32>(spv::Capability::Shader)});
  glsl_ext_ = Alloc();
  std::vector<u32> ops{glsl_ext_};
  PutString(ops, "GLSL.std.450");
  Instr(ext_imports_, spv::Op::OpExtInstImport, ops);
  Instr(mem_model_, spv::Op::OpMemoryModel,
        {static_cast<u32>(spv::AddressingModel::Logical),
         static_cast<u32>(spv::MemoryModel::GLSL450)});
}

// ---- types -----------------------------------------------------------------
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
Id Module::TypeStruct(const std::vector<Id>& members) {
  auto it = struct_cache_.find(members);
  if (it != struct_cache_.end())
    return it->second;
  const Id id = Alloc();
  std::vector<u32> ops{id};
  ops.insert(ops.end(), members.begin(), members.end());
  Instr(types_consts_, spv::Op::OpTypeStruct, ops);
  struct_cache_[members] = id;
  return id;
}
Id Module::TypePointer(spv::StorageClass sc, Id pointee) {
  const u64 k =
      Key(CacheKind::kPointer, pointee, static_cast<u64>(sc));
  Id id;
  if (Lookup(k, id))
    return id;
  id = Alloc();
  Instr(types_consts_, spv::Op::OpTypePointer,
        {id, static_cast<u32>(sc), pointee});
  return Cached(k, id);
}
Id Module::TypeFunction(Id ret, const std::vector<Id>& params) {
  std::vector<Id> key{ret};
  key.insert(key.end(), params.begin(), params.end());
  auto it = fn_type_cache_.find(key);
  if (it != fn_type_cache_.end())
    return it->second;
  const Id id = Alloc();
  std::vector<u32> ops{id, ret};
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
        {id, sampled_type, static_cast<u32>(dim), depth, arrayed, ms,
         sampled, static_cast<u32>(fmt)});
  return id;
}
Id Module::TypeSampledImage(Id image_type) {
  const Id id = Alloc();
  Instr(types_consts_, spv::Op::OpTypeSampledImage, {id, image_type});
  return id;
}

// ---- constants -------------------------------------------------------------
Id Module::ConstU32(u32 v) {
  const u64 k = Key(CacheKind::kConstU32, v);
  Id id;
  if (Lookup(k, id))
    return id;
  const Id t = TypeInt(32, false);
  id = Alloc();
  Instr(types_consts_, spv::Op::OpConstant, {t, id, v});
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
  return Cached(k, id);
}
Id Module::ConstComposite(Id type, const std::vector<Id>& parts) {
  const Id id = Alloc();
  std::vector<u32> ops{type, id};
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

// ---- globals / decorations -------------------------------------------------
Id Module::Variable(Id ptr_type, spv::StorageClass sc, Id init) {
  const Id id = Alloc();
  std::vector<u32> ops{ptr_type, id, static_cast<u32>(sc)};
  if (init)
    ops.push_back(init);
  Instr(types_consts_, spv::Op::OpVariable, ops);
  return id;
}
void Module::Decorate(Id target,
                      spv::Decoration dec,
                      const std::vector<u32>& operands) {
  std::vector<u32> ops{target, static_cast<u32>(dec)};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(decos_, spv::Op::OpDecorate, ops);
}
void Module::MemberDecorate(Id struct_type,
                            u32 member,
                            spv::Decoration dec,
                            const std::vector<u32>& operands) {
  std::vector<u32> ops{struct_type, member, static_cast<u32>(dec)};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(decos_, spv::Op::OpMemberDecorate, ops);
}
void Module::Name(Id target, const std::string& n) {
  std::vector<u32> ops{target};
  PutString(ops, n);
  Instr(debug_, spv::Op::OpName, ops);
}
void Module::MemberName(Id struct_type, u32 member, const std::string& n) {
  std::vector<u32> ops{struct_type, member};
  PutString(ops, n);
  Instr(debug_, spv::Op::OpMemberName, ops);
}

Id Module::String(const std::string& s) {
  const Id id = Alloc();
  std::vector<u32> ops{id};
  PutString(ops, s);
  Instr(strings_, spv::Op::OpString, ops);
  return id;
}

void Module::Line(Id file, u32 line) {
  Instr(fn_body_, spv::Op::OpLine, {file, line, 0});
}

void Module::EntryPoint(spv::ExecutionModel model,
                        Id fn,
                        const std::string& n,
                        const std::vector<Id>& interface) {
  std::vector<u32> ops{static_cast<u32>(model), fn};
  PutString(ops, n);
  ops.insert(ops.end(), interface.begin(), interface.end());
  Instr(entries_, spv::Op::OpEntryPoint, ops);
}
void Module::ExecMode(Id fn,
                      spv::ExecutionMode mode,
                      const std::vector<u32>& operands) {
  std::vector<u32> ops{fn, static_cast<u32>(mode)};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(exec_modes_, spv::Op::OpExecutionMode, ops);
}
void Module::Capability(spv::Capability cap) {
  Instr(caps_, spv::Op::OpCapability, {static_cast<u32>(cap)});
}

void Module::Extension(const std::string& name) {
  std::vector<u32> ops;
  PutString(ops, name);
  Instr(exts_, spv::Op::OpExtension, ops);
}

// ---- function / block construction -----------------------------------------
void Module::PhysicalStorageBuffers() {
  Capability(spv::Capability::Int64);
  Capability(spv::Capability::PhysicalStorageBufferAddresses);
  Extension("SPV_KHR_physical_storage_buffer");
  mem_model_.clear();
  Instr(mem_model_, spv::Op::OpMemoryModel,
        {static_cast<u32>(spv::AddressingModel::PhysicalStorageBuffer64),
         static_cast<u32>(spv::MemoryModel::GLSL450)});
}

Id Module::BeginFunction(Id ret_type, Id fn_type,
                         const std::vector<Id>& parameter_types,
                         std::vector<Id>* parameter_ids) {
  const Id fn = Alloc();
  Instr(fn_body_, spv::Op::OpFunction,
        {ret_type, fn,
         static_cast<u32>(spv::FunctionControlMask::MaskNone), fn_type});
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

Id Module::Emit(spv::Op op, Id result_type, const std::vector<Id>& operands) {
  const Id id = Alloc();
  std::vector<u32> ops{result_type, id};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(fn_body_, op, ops);
  return id;
}
void Module::EmitVoid(spv::Op op, const std::vector<Id>& operands) {
  Instr(fn_body_, op, operands);
}
Id Module::ExtInst(Id result_type,
                   u32 glsl_op,
                   const std::vector<Id>& operands) {
  const Id id = Alloc();
  std::vector<u32> ops{result_type, id, glsl_ext_, glsl_op};
  ops.insert(ops.end(), operands.begin(), operands.end());
  Instr(fn_body_, spv::Op::OpExtInst, ops);
  return id;
}

Id Module::Load(Id type, Id ptr) {
  return Emit(spv::Op::OpLoad, type, {ptr});
}
void Module::Store(Id ptr, Id value) {
  EmitVoid(spv::Op::OpStore, {ptr, value});
}
Id Module::AccessChain(Id ptr_type, Id base, const std::vector<Id>& indices) {
  std::vector<Id> ops{base};
  ops.insert(ops.end(), indices.begin(), indices.end());
  return Emit(spv::Op::OpAccessChain, ptr_type, ops);
}
Id Module::Bitcast(Id type, Id value) {
  return Emit(spv::Op::OpBitcast, type, {value});
}
Id Module::CompositeExtract(Id type, Id composite, u32 index) {
  const Id id = Alloc();
  Instr(fn_body_, spv::Op::OpCompositeExtract, {type, id, composite, index});
  return id;
}
Id Module::CompositeConstruct(Id type, const std::vector<Id>& parts) {
  return Emit(spv::Op::OpCompositeConstruct, type, parts);
}
Id Module::VectorShuffle(Id type,
                         Id a,
                         Id b,
                         const std::vector<u32>& comps) {
  const Id id = Alloc();
  std::vector<u32> ops{type, id, a, b};
  ops.insert(ops.end(), comps.begin(), comps.end());
  Instr(fn_body_, spv::Op::OpVectorShuffle, ops);
  return id;
}

void Module::SelectionMerge(Id merge_block) {
  EmitVoid(spv::Op::OpSelectionMerge,
           {merge_block,
            static_cast<u32>(spv::SelectionControlMask::MaskNone)});
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
                    const std::vector<std::pair<u32, Id>>& cases) {
  std::vector<u32> ops{selector, default_label};
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

// ---- assembly --------------------------------------------------------------
std::vector<u32> Module::Assemble() const {
  std::vector<u32> out;
  out.push_back(spv::MagicNumber);  // 0x07230203
  out.push_back(0x00010300u);       // SPIR-V 1.3 (Vulkan 1.1)
  out.push_back(0);                 // generator (0 = unknown)
  out.push_back(bound_);            // id bound
  out.push_back(0);                 // schema
  const auto append = [&](const std::vector<u32>& s) {
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

#endif  // DELTA_HAVE_SPIRV_BACKEND
