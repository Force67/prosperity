#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

#include <spirv/unified1/spirv.hpp11>
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/strings/xstring.h"

namespace gpu::gcn::spirv {

using Id = u32;

class Module {
 public:
  Module();

  Id Alloc() { return bound_++; }
  base::Vector<u32> Assemble() const;

  Id TypeVoid();
  Id TypeBool();
  Id TypeInt(u32 width = 32, bool sign = false);
  Id TypeFloat(u32 width = 32);
  Id TypeVec(Id comp, u32 count);
  Id TypeArray(Id elem, u32 len);
  Id TypeRuntimeArray(Id elem);
  Id TypeStruct(const base::Vector<Id>& members);
  Id TypePointer(spv::StorageClass sc, Id pointee);
  Id TypeFunction(Id ret, const base::Vector<Id>& params = {});
  Id TypeImage(Id sampled_type,
               spv::Dim dim,
               u32 depth,
               u32 arrayed,
               u32 ms,
               u32 sampled,
               spv::ImageFormat fmt);
  Id TypeSampledImage(Id image_type);
  Id TypeSampler();

  Id ConstU32(u32 v);
  Id ConstI32(i32 v);
  Id ConstF32(float v);
  Id ConstBool(bool v);
  Id ConstComposite(Id type, const base::Vector<Id>& parts);
  Id ConstNull(Id type);

  Id Variable(Id ptr_type, spv::StorageClass sc, Id init = 0);
  void Decorate(Id target,
                spv::Decoration dec,
                const base::Vector<u32>& operands = {});
  void MemberDecorate(Id struct_type,
                      u32 member,
                      spv::Decoration dec,
                      const base::Vector<u32>& operands = {});
  void Name(Id target, const base::String& n);

  Id String(const base::String& s);
  void Line(Id file, u32 line);
  size_t BodyWords() const { return fn_body_.size(); }

  void EntryPoint(spv::ExecutionModel model,
                  Id fn,
                  const base::String& name,
                  const base::Vector<Id>& interface);
  void ExecMode(Id fn,
                spv::ExecutionMode mode,
                const base::Vector<u32>& operands = {});
  void Capability(spv::Capability cap);
  void Extension(const base::String& name);
  void PhysicalStorageBuffers();

  Id BeginFunction(
      Id ret_type,
      Id fn_type,
      const base::Vector<Id>& parameter_types = {},
      base::Vector<Id>* parameter_ids = nullptr,
      spv::FunctionControlMask control = spv::FunctionControlMask::MaskNone);
  Id NewBlock();
  void OpenBlock(Id label);
  Id CurrentBlock() const { return cur_block_; }
  void EndFunction();

  Id Emit(spv::Op op, Id result_type, const base::Vector<Id>& operands);
  void EmitVoid(spv::Op op, const base::Vector<Id>& operands);
  Id ExtInst(Id result_type, u32 glsl_op, const base::Vector<Id>& operands);

  Id Load(Id type, Id ptr);
  void Store(Id ptr, Id value);
  Id AccessChain(Id ptr_type, Id base, const base::Vector<Id>& indices);
  Id Bitcast(Id type, Id value);
  Id CompositeExtract(Id type, Id composite, u32 index);
  Id CompositeConstruct(Id type, const base::Vector<Id>& parts);

  void SelectionMerge(Id merge_block);
  void LoopMerge(Id merge_block, Id continue_block);
  void Branch(Id target);
  void BranchConditional(Id cond, Id t, Id f);
  void Switch(Id selector,
              Id default_label,
              const base::Vector<base::Pair<u32, Id>>& cases);
  void ReturnVoid();
  void Unreachable();
  void Kill();

 private:
  enum class CacheKind : u8 {
    kVoid,
    kBool,
    kInt,
    kFloat,
    kVec,
    kArray,
    kRuntimeArray,
    kPointer,
    kConstU32,
    kConstI32,
    kConstF32,
    kConstBool,
    kConstNull,
  };
  static u64 Key(CacheKind kind, u64 a = 0, u64 b = 0) {
    return (static_cast<u64>(kind) << 56) | (a << 24) | (b & 0xFFFFFF);
  }
  Id Cached(u64 key, Id id) {
    cache_[key] = id;
    return id;
  }
  bool Lookup(u64 key, Id& id) const {
    auto it = cache_.find(key);
    if (it == cache_.end())
      return false;
    id = it->second;
    return true;
  }

  void Instr(base::Vector<u32>& sec, spv::Op op, const base::Vector<u32>& ops);
  void PutString(base::Vector<u32>& sec, const base::String& s);

  void NoteType(Id id, Id type) {
    if (id >= type_of_.size())
      type_of_.resize(base::Max<size_t>(id + 1, type_of_.size() * 2), 0);
    type_of_[id] = type;
  }
  Id TypeOf(Id id) const { return id < type_of_.size() ? type_of_[id] : 0; }
  base::Vector<Id> type_of_;
  base::HashMap<Id, Id> bitcast_source_;
  base::HashMap<Id, u32> const_bits_;
  base::HashMap<Id, bool> const_bools_;

  struct Computed {
    Id block = 0, id = 0;
    u32 offset = 0;
  };
  base::HashMap<u64, Computed> computed_;
  Id computed_block_ = 0;
  Id Reuse(const u32 (&head)[2], const u32* rest, size_t count, u64* key) const;
  void Remember(u64 key, Id id, u32 offset);
  Id t_u32_ = 0, t_i32_ = 0, t_bool_ = 0;
  Id Fold(spv::Op op, Id result_type, const base::Vector<Id>& operands);
  base::HashSet<u64> decorated_;

  u32 bound_ = 1;
  u32 version_ = 0x00010600u;
  Id sampler_type_ = 0;
  base::Vector<Id> globals_;
  Id glsl_ext_ = 0;

  base::Vector<u32> caps_, exts_, ext_imports_, mem_model_, entries_,
      exec_modes_;
  base::Vector<u32> strings_, debug_, decos_, types_consts_, fn_body_;
  base::HashMap<u64, Id> cache_;
  base::Map<base::Vector<Id>, Id> struct_cache_;
  base::Map<base::Vector<Id>, Id> fn_type_cache_;
  Id cur_block_ = 0;
};

}  // namespace gpu::gcn::spirv
