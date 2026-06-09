#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Minimal SPIR-V module builder. Emits a valid SPIR-V binary word stream
 * directly (no GLSL, no shaderc front end), so the GCN recompiler can translate
 * straight to SPIR-V and hand the result to a SPIRV-Tools optimize pass. This is
 * the "emit SPIR-V then optimize" path; the translator (gcn_spirv) models the GCN
 * register file as Private-storage variables and relies on spirv-opt's SSA
 * rewrite + performance passes to legalise and optimise the naive output.
 *
 * Scope: enough of SPIR-V 1.3 (Vulkan 1.1) to express the VS/PS the recompiler
 * needs -- scalar/vector int+float arithmetic, GLSL.std.450 ext-inst, sampled
 * images, input/output/private/pushconstant/uniform variables, structured
 * control flow. Not a general assembler.
 */

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <spirv/unified1/spirv.hpp11>

namespace gpu::gcn::spirv {

using Id = uint32_t;

// SSA-ish module builder. Instructions are accumulated into the SPIR-V logical
// sections (the layout rules require a fixed section order); assemble()
// concatenates them behind the 5-word header. Types and constants are de-duped.
class Module {
 public:
  Module();

  // ---- id + assembly ----
  Id alloc() { return bound_++; }
  std::vector<uint32_t> assemble() const;

  // ---- types (cached) ----
  Id typeVoid();
  Id typeBool();
  Id typeInt(uint32_t width = 32, bool sign = false);
  Id typeFloat(uint32_t width = 32);
  Id typeVec(Id comp, uint32_t count);
  Id typeArray(Id elem, uint32_t len);            // length via an emitted u32 const
  Id typeRuntimeArray(Id elem);
  Id typeStruct(const std::vector<Id> &members);
  Id typePointer(spv::StorageClass sc, Id pointee);
  Id typeFunction(Id ret, const std::vector<Id> &params = {});
  Id typeImage(Id sampledType, spv::Dim dim, uint32_t depth, uint32_t arrayed,
               uint32_t ms, uint32_t sampled, spv::ImageFormat fmt);
  Id typeSampledImage(Id imageType);

  // ---- constants (cached) ----
  Id constU32(uint32_t v);
  Id constI32(int32_t v);
  Id constF32(float v);
  Id constBool(bool v);
  Id constComposite(Id type, const std::vector<Id> &parts);  // not cached
  Id constNull(Id type);

  // ---- global variables / decorations ----
  Id variable(Id ptrType, spv::StorageClass sc, Id init = 0);
  void decorate(Id target, spv::Decoration dec, const std::vector<uint32_t> &operands = {});
  void memberDecorate(Id structType, uint32_t member, spv::Decoration dec,
                      const std::vector<uint32_t> &operands = {});
  void name(Id target, const std::string &n);
  void memberName(Id structType, uint32_t member, const std::string &n);

  // entry point + exec modes
  void entryPoint(spv::ExecutionModel model, Id fn, const std::string &name,
                  const std::vector<Id> &interface);
  void execMode(Id fn, spv::ExecutionMode mode, const std::vector<uint32_t> &operands = {});
  void capability(spv::Capability cap);

  Id glslExt() const { return glslExt_; }

  // ---- function + block construction ----
  // Begin a function (emits OpFunction + the entry OpLabel) and return the fn id.
  Id beginFunction(Id retType, Id fnType);
  Id newBlock();                  // allocate a label id (not yet opened)
  void openBlock(Id label);       // emit OpLabel for a previously allocated id
  Id currentBlock() const { return curBlock_; }
  void endFunction();

  // generic instruction emitters into the current function body
  Id emit(spv::Op op, Id resultType, const std::vector<Id> &operands);
  void emitVoid(spv::Op op, const std::vector<Id> &operands);
  Id extInst(Id resultType, uint32_t glslOp, const std::vector<Id> &operands);

  // common ops
  Id load(Id type, Id ptr);
  void store(Id ptr, Id value);
  Id accessChain(Id ptrType, Id base, const std::vector<Id> &indices);
  Id bitcast(Id type, Id value);
  Id compositeExtract(Id type, Id composite, uint32_t index);
  Id compositeConstruct(Id type, const std::vector<Id> &parts);
  Id vectorShuffle(Id type, Id a, Id b, const std::vector<uint32_t> &comps);

  // structured control flow helpers
  void selectionMerge(Id mergeBlock);
  void loopMerge(Id mergeBlock, Id continueBlock);
  void branch(Id target);
  void branchConditional(Id cond, Id t, Id f);
  // OpSwitch: selector + default label + (literal, label) cases.
  void switchInst(Id selector, Id defaultLabel,
                  const std::vector<std::pair<uint32_t, Id>> &cases);
  void returnVoid();
  void unreachable();
  void kill();  // OpKill (PS discard)

 private:
  Id key(const std::string &k, Id id);  // cache helper
  void word(std::vector<uint32_t> &sec, uint32_t w) { sec.push_back(w); }
  void inst(std::vector<uint32_t> &sec, spv::Op op, const std::vector<uint32_t> &ops);
  void str(std::vector<uint32_t> &sec, const std::string &s);

  uint32_t bound_ = 1;
  Id glslExt_ = 0;

  std::vector<uint32_t> caps_, exts_, extImports_, memModel_, entries_, execModes_;
  std::vector<uint32_t> debug_, decos_, typesConsts_, fnBody_;
  std::map<std::string, Id> cache_;
  Id curBlock_ = 0;
};

}  // namespace gpu::gcn::spirv
