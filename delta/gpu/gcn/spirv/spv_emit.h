#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Minimal SPIR-V module builder. Emits a valid SPIR-V binary word stream
 * directly, so the GCN recompiler can translate straight to SPIR-V and hand
 * the result to a SPIRV-Tools optimize pass. The translator models the GCN
 * register file as Private-storage variables and relies on spirv-opt's SSA
 * rewrite + performance passes to legalise and optimise the naive output.
 *
 * Scope: enough of SPIR-V 1.3 (Vulkan 1.1) to express the shaders the
 * recompiler needs -- scalar/vector int+float arithmetic, GLSL.std.450
 * ext-inst, sampled images, input/output/private/pushconstant/uniform/
 * storage-buffer/workgroup variables, structured control flow. Not a general
 * assembler.
 */

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <spirv/unified1/spirv.hpp11>

namespace gpu::gcn::spirv {

using Id = uint32_t;

// SSA-ish module builder. Instructions are accumulated into the SPIR-V logical
// sections (the layout rules require a fixed section order); Assemble()
// concatenates them behind the 5-word header. Types and constants are de-duped
// through a packed-integer cache (struct/function types through an exact
// member-list map).
class Module {
 public:
  Module();

  // ---- id + assembly ----
  Id Alloc() { return bound_++; }
  std::vector<uint32_t> Assemble() const;

  // ---- types (cached) ----
  Id TypeVoid();
  Id TypeBool();
  Id TypeInt(uint32_t width = 32, bool sign = false);
  Id TypeFloat(uint32_t width = 32);
  Id TypeVec(Id comp, uint32_t count);
  Id TypeArray(Id elem, uint32_t len);  // length via an emitted u32 const
  Id TypeRuntimeArray(Id elem);
  Id TypeStruct(const std::vector<Id>& members);
  Id TypePointer(spv::StorageClass sc, Id pointee);
  Id TypeFunction(Id ret, const std::vector<Id>& params = {});
  Id TypeImage(Id sampled_type,
               spv::Dim dim,
               uint32_t depth,
               uint32_t arrayed,
               uint32_t ms,
               uint32_t sampled,
               spv::ImageFormat fmt);
  Id TypeSampledImage(Id image_type);

  // ---- constants (cached) ----
  Id ConstU32(uint32_t v);
  Id ConstI32(int32_t v);
  Id ConstF32(float v);
  Id ConstBool(bool v);
  Id ConstComposite(Id type, const std::vector<Id>& parts);  // not cached
  Id ConstNull(Id type);

  // ---- global variables / decorations ----
  Id Variable(Id ptr_type, spv::StorageClass sc, Id init = 0);
  void Decorate(Id target,
                spv::Decoration dec,
                const std::vector<uint32_t>& operands = {});
  void MemberDecorate(Id struct_type,
                      uint32_t member,
                      spv::Decoration dec,
                      const std::vector<uint32_t>& operands = {});
  void Name(Id target, const std::string& n);
  void MemberName(Id struct_type, uint32_t member, const std::string& n);

  // ---- debug info ----
  // OpString (debug section); the id is what OpLine references as a file.
  Id String(const std::string& s);
  // OpLine marker in the function body: subsequent instructions carry
  // (file, line) until the next marker. The GCN recompiler uses line == the
  // instruction's dword pc, so spirv-dis output maps back to the guest code.
  void Line(Id file, uint32_t line);
  // Function-body words emitted so far; the delta across an emission tells a
  // caller exactly how much SPIR-V one guest instruction produced.
  size_t BodyWords() const { return fn_body_.size(); }

  // entry point + exec modes
  void EntryPoint(spv::ExecutionModel model,
                  Id fn,
                  const std::string& name,
                  const std::vector<Id>& interface);
  void ExecMode(Id fn,
                spv::ExecutionMode mode,
                const std::vector<uint32_t>& operands = {});
  void Capability(spv::Capability cap);
  // OpExtension: the SPIR-V extension string a capability belongs to. Emitted
  // after the capabilities and before the ext-inst imports, per the module's
  // required section order.
  void Extension(const std::string& name);

  Id GlslExt() const { return glsl_ext_; }

  // ---- function + block construction ----
  // Begin a function (emits OpFunction + the entry OpLabel); returns the fn id.
  Id BeginFunction(Id ret_type, Id fn_type);
  Id NewBlock();             // allocate a label id (not yet opened)
  void OpenBlock(Id label);  // emit OpLabel for a previously allocated id
  Id CurrentBlock() const { return cur_block_; }
  void EndFunction();

  // generic instruction emitters into the current function body
  Id Emit(spv::Op op, Id result_type, const std::vector<Id>& operands);
  void EmitVoid(spv::Op op, const std::vector<Id>& operands);
  Id ExtInst(Id result_type, uint32_t glsl_op, const std::vector<Id>& operands);

  // common ops
  Id Load(Id type, Id ptr);
  void Store(Id ptr, Id value);
  Id AccessChain(Id ptr_type, Id base, const std::vector<Id>& indices);
  Id Bitcast(Id type, Id value);
  Id CompositeExtract(Id type, Id composite, uint32_t index);
  Id CompositeConstruct(Id type, const std::vector<Id>& parts);
  Id VectorShuffle(Id type, Id a, Id b, const std::vector<uint32_t>& comps);

  // structured control flow helpers
  void SelectionMerge(Id merge_block);
  void LoopMerge(Id merge_block, Id continue_block);
  void Branch(Id target);
  void BranchConditional(Id cond, Id t, Id f);
  // OpSwitch: selector + default label + (literal, label) cases.
  void Switch(Id selector,
              Id default_label,
              const std::vector<std::pair<uint32_t, Id>>& cases);
  void ReturnVoid();
  void Unreachable();
  void Kill();  // OpKill (PS discard)

 private:
  // Packed cache key: kind (8 bits) | a (32 bits) | b (24 bits). Exact -- every
  // cached entity maps to a unique key, no hashing of the payload.
  enum class CacheKind : uint8_t {
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
  static uint64_t Key(CacheKind kind, uint64_t a = 0, uint64_t b = 0) {
    return (static_cast<uint64_t>(kind) << 56) | (a << 24) | (b & 0xFFFFFF);
  }
  Id Cached(uint64_t key, Id id) {
    cache_[key] = id;
    return id;
  }
  bool Lookup(uint64_t key, Id& id) const {
    auto it = cache_.find(key);
    if (it == cache_.end())
      return false;
    id = it->second;
    return true;
  }

  void PutWord(std::vector<uint32_t>& sec, uint32_t w) { sec.push_back(w); }
  void Instr(std::vector<uint32_t>& sec,
             spv::Op op,
             const std::vector<uint32_t>& ops);
  void PutString(std::vector<uint32_t>& sec, const std::string& s);

  uint32_t bound_ = 1;
  Id glsl_ext_ = 0;

  std::vector<uint32_t> caps_, exts_, ext_imports_, mem_model_, entries_,
      exec_modes_;
  // strings_ holds OpStrings, which the debug-section layout places before
  // the OpNames in debug_.
  std::vector<uint32_t> strings_, debug_, decos_, types_consts_, fn_body_;
  std::unordered_map<uint64_t, Id> cache_;
  std::map<std::vector<Id>, Id> struct_cache_;  // member-list keyed (exact)
  std::map<std::vector<Id>, Id> fn_type_cache_;
  Id cur_block_ = 0;
};

}  // namespace gpu::gcn::spirv
