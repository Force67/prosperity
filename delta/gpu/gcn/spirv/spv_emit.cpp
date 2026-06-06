/*
 * PS4Delta : PS4 emulation and research project
 *
 * Minimal SPIR-V module builder. See spv_emit.h.
 */

#include "spv_emit.h"

#include <cstring>

#include <spirv/unified1/GLSL.std.450.h>

namespace gpu::gcn::spirv {

// Pack a literal string into SPIR-V words (LSB-first, null-terminated, zero
// padded to a word boundary).
void Module::str(std::vector<uint32_t> &sec, const std::string &s) {
  uint32_t w = 0;
  int b = 0;
  auto push = [&](uint8_t c) {
    w |= static_cast<uint32_t>(c) << (8 * b);
    if (++b == 4) { sec.push_back(w); w = 0; b = 0; }
  };
  for (char c : s) push(static_cast<uint8_t>(c));
  push(0);                       // null terminator
  if (b != 0) sec.push_back(w);  // flush trailing partial word (incl. padding)
}

void Module::inst(std::vector<uint32_t> &sec, spv::Op op,
                  const std::vector<uint32_t> &ops) {
  uint32_t wc = 1 + static_cast<uint32_t>(ops.size());
  sec.push_back((wc << 16) | static_cast<uint32_t>(op));
  for (uint32_t o : ops) sec.push_back(o);
}

Module::Module() {
  inst(caps_, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Shader)});
  glslExt_ = alloc();
  // OpExtInstImport <id> "GLSL.std.450"
  std::vector<uint32_t> ops{glslExt_};
  str(ops, "GLSL.std.450");
  inst(extImports_, spv::Op::OpExtInstImport, ops);
  // OpMemoryModel Logical GLSL450
  inst(memModel_, spv::Op::OpMemoryModel,
       {static_cast<uint32_t>(spv::AddressingModel::Logical),
        static_cast<uint32_t>(spv::MemoryModel::GLSL450)});
}

Id Module::key(const std::string &k, Id id) {
  cache_[k] = id;
  return id;
}

// ---- types -----------------------------------------------------------------
Id Module::typeVoid() {
  auto it = cache_.find("void"); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpTypeVoid, {id});
  return key("void", id);
}
Id Module::typeBool() {
  auto it = cache_.find("bool"); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpTypeBool, {id});
  return key("bool", id);
}
Id Module::typeInt(uint32_t width, bool sign) {
  std::string k = "int" + std::to_string(width) + (sign ? "s" : "u");
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc();
  inst(typesConsts_, spv::Op::OpTypeInt, {id, width, sign ? 1u : 0u});
  return key(k, id);
}
Id Module::typeFloat(uint32_t width) {
  std::string k = "flt" + std::to_string(width);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpTypeFloat, {id, width});
  return key(k, id);
}
Id Module::typeVec(Id comp, uint32_t count) {
  std::string k = "vec" + std::to_string(comp) + "x" + std::to_string(count);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpTypeVector, {id, comp, count});
  return key(k, id);
}
Id Module::typeArray(Id elem, uint32_t len) {
  Id lenId = constU32(len);
  std::string k = "arr" + std::to_string(elem) + "x" + std::to_string(len);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpTypeArray, {id, elem, lenId});
  return key(k, id);
}
Id Module::typeRuntimeArray(Id elem) {
  std::string k = "rarr" + std::to_string(elem);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpTypeRuntimeArray, {id, elem});
  return key(k, id);
}
Id Module::typeStruct(const std::vector<Id> &members) {
  std::string k = "struct";
  for (Id m : members) k += "_" + std::to_string(m);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc();
  std::vector<uint32_t> ops{id};
  for (Id m : members) ops.push_back(m);
  inst(typesConsts_, spv::Op::OpTypeStruct, ops);
  return key(k, id);
}
Id Module::typePointer(spv::StorageClass sc, Id pointee) {
  std::string k = "ptr" + std::to_string((int)sc) + "_" + std::to_string(pointee);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc();
  inst(typesConsts_, spv::Op::OpTypePointer, {id, static_cast<uint32_t>(sc), pointee});
  return key(k, id);
}
Id Module::typeFunction(Id ret, const std::vector<Id> &params) {
  std::string k = "fn" + std::to_string(ret);
  for (Id p : params) k += "_" + std::to_string(p);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc();
  std::vector<uint32_t> ops{id, ret};
  for (Id p : params) ops.push_back(p);
  inst(typesConsts_, spv::Op::OpTypeFunction, ops);
  return key(k, id);
}
Id Module::typeImage(Id sampledType, spv::Dim dim, uint32_t depth, uint32_t arrayed,
                     uint32_t ms, uint32_t sampled, spv::ImageFormat fmt) {
  Id id = alloc();
  inst(typesConsts_, spv::Op::OpTypeImage,
       {id, sampledType, static_cast<uint32_t>(dim), depth, arrayed, ms, sampled,
        static_cast<uint32_t>(fmt)});
  return id;
}
Id Module::typeSampledImage(Id imageType) {
  Id id = alloc();
  inst(typesConsts_, spv::Op::OpTypeSampledImage, {id, imageType});
  return id;
}

// ---- constants -------------------------------------------------------------
Id Module::constU32(uint32_t v) {
  std::string k = "cu" + std::to_string(v);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id t = typeInt(32, false), id = alloc();
  inst(typesConsts_, spv::Op::OpConstant, {t, id, v});
  return key(k, id);
}
Id Module::constI32(int32_t v) {
  std::string k = "ci" + std::to_string(v);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id t = typeInt(32, true), id = alloc();
  inst(typesConsts_, spv::Op::OpConstant, {t, id, static_cast<uint32_t>(v)});
  return key(k, id);
}
Id Module::constF32(float v) {
  uint32_t bits; std::memcpy(&bits, &v, 4);
  std::string k = "cf" + std::to_string(bits);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id t = typeFloat(32), id = alloc();
  inst(typesConsts_, spv::Op::OpConstant, {t, id, bits});
  return key(k, id);
}
Id Module::constBool(bool v) {
  std::string k = v ? "ctrue" : "cfalse";
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id t = typeBool(), id = alloc();
  inst(typesConsts_, v ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse, {t, id});
  return key(k, id);
}
Id Module::constComposite(Id type, const std::vector<Id> &parts) {
  Id id = alloc();
  std::vector<uint32_t> ops{type, id};
  for (Id p : parts) ops.push_back(p);
  inst(typesConsts_, spv::Op::OpConstantComposite, ops);
  return id;
}
Id Module::constNull(Id type) {
  std::string k = "cnull" + std::to_string(type);
  auto it = cache_.find(k); if (it != cache_.end()) return it->second;
  Id id = alloc(); inst(typesConsts_, spv::Op::OpConstantNull, {type, id});
  return key(k, id);
}

// ---- globals / decorations -------------------------------------------------
Id Module::variable(Id ptrType, spv::StorageClass sc, Id init) {
  Id id = alloc();
  std::vector<uint32_t> ops{ptrType, id, static_cast<uint32_t>(sc)};
  if (init) ops.push_back(init);
  inst(typesConsts_, spv::Op::OpVariable, ops);
  return id;
}
void Module::decorate(Id target, spv::Decoration dec, const std::vector<uint32_t> &operands) {
  std::vector<uint32_t> ops{target, static_cast<uint32_t>(dec)};
  for (uint32_t o : operands) ops.push_back(o);
  inst(decos_, spv::Op::OpDecorate, ops);
}
void Module::memberDecorate(Id structType, uint32_t member, spv::Decoration dec,
                            const std::vector<uint32_t> &operands) {
  std::vector<uint32_t> ops{structType, member, static_cast<uint32_t>(dec)};
  for (uint32_t o : operands) ops.push_back(o);
  inst(decos_, spv::Op::OpMemberDecorate, ops);
}
void Module::name(Id target, const std::string &n) {
  std::vector<uint32_t> ops{target};
  str(ops, n);
  inst(debug_, spv::Op::OpName, ops);
}
void Module::memberName(Id structType, uint32_t member, const std::string &n) {
  std::vector<uint32_t> ops{structType, member};
  str(ops, n);
  inst(debug_, spv::Op::OpMemberName, ops);
}

void Module::entryPoint(spv::ExecutionModel model, Id fn, const std::string &n,
                        const std::vector<Id> &interface) {
  std::vector<uint32_t> ops{static_cast<uint32_t>(model), fn};
  str(ops, n);
  for (Id i : interface) ops.push_back(i);
  inst(entries_, spv::Op::OpEntryPoint, ops);
}
void Module::execMode(Id fn, spv::ExecutionMode mode, const std::vector<uint32_t> &operands) {
  std::vector<uint32_t> ops{fn, static_cast<uint32_t>(mode)};
  for (uint32_t o : operands) ops.push_back(o);
  inst(execModes_, spv::Op::OpExecutionMode, ops);
}
void Module::capability(spv::Capability cap) {
  inst(caps_, spv::Op::OpCapability, {static_cast<uint32_t>(cap)});
}

// ---- function / block construction -----------------------------------------
Id Module::beginFunction(Id retType, Id fnType) {
  Id fn = alloc();
  inst(fnBody_, spv::Op::OpFunction,
       {retType, fn, static_cast<uint32_t>(spv::FunctionControlMask::MaskNone), fnType});
  Id entry = alloc();
  inst(fnBody_, spv::Op::OpLabel, {entry});
  curBlock_ = entry;
  return fn;
}
Id Module::newBlock() { return alloc(); }
void Module::openBlock(Id label) {
  inst(fnBody_, spv::Op::OpLabel, {label});
  curBlock_ = label;
}
void Module::endFunction() {
  inst(fnBody_, spv::Op::OpFunctionEnd, {});
  curBlock_ = 0;
}

Id Module::emit(spv::Op op, Id resultType, const std::vector<Id> &operands) {
  Id id = alloc();
  std::vector<uint32_t> ops{resultType, id};
  for (Id o : operands) ops.push_back(o);
  inst(fnBody_, op, ops);
  return id;
}
void Module::emitVoid(spv::Op op, const std::vector<Id> &operands) {
  inst(fnBody_, op, operands);
}
Id Module::extInst(Id resultType, uint32_t glslOp, const std::vector<Id> &operands) {
  Id id = alloc();
  std::vector<uint32_t> ops{resultType, id, glslExt_, glslOp};
  for (Id o : operands) ops.push_back(o);
  inst(fnBody_, spv::Op::OpExtInst, ops);
  return id;
}

Id Module::load(Id type, Id ptr) { return emit(spv::Op::OpLoad, type, {ptr}); }
void Module::store(Id ptr, Id value) { emitVoid(spv::Op::OpStore, {ptr, value}); }
Id Module::accessChain(Id ptrType, Id base, const std::vector<Id> &indices) {
  std::vector<Id> ops{base};
  for (Id i : indices) ops.push_back(i);
  return emit(spv::Op::OpAccessChain, ptrType, ops);
}
Id Module::bitcast(Id type, Id value) { return emit(spv::Op::OpBitcast, type, {value}); }
Id Module::compositeExtract(Id type, Id composite, uint32_t index) {
  Id id = alloc();
  inst(fnBody_, spv::Op::OpCompositeExtract, {type, id, composite, index});
  return id;
}
Id Module::compositeConstruct(Id type, const std::vector<Id> &parts) {
  return emit(spv::Op::OpCompositeConstruct, type, parts);
}
Id Module::vectorShuffle(Id type, Id a, Id b, const std::vector<uint32_t> &comps) {
  Id id = alloc();
  std::vector<uint32_t> ops{type, id, a, b};
  for (uint32_t c : comps) ops.push_back(c);
  inst(fnBody_, spv::Op::OpVectorShuffle, ops);
  return id;
}

void Module::selectionMerge(Id mergeBlock) {
  emitVoid(spv::Op::OpSelectionMerge,
           {mergeBlock, static_cast<uint32_t>(spv::SelectionControlMask::MaskNone)});
}
void Module::loopMerge(Id mergeBlock, Id continueBlock) {
  emitVoid(spv::Op::OpLoopMerge,
           {mergeBlock, continueBlock, static_cast<uint32_t>(spv::LoopControlMask::MaskNone)});
}
void Module::branch(Id target) { emitVoid(spv::Op::OpBranch, {target}); }
void Module::branchConditional(Id cond, Id t, Id f) {
  emitVoid(spv::Op::OpBranchConditional, {cond, t, f});
}
void Module::returnVoid() { emitVoid(spv::Op::OpReturn, {}); }
void Module::unreachable() { emitVoid(spv::Op::OpUnreachable, {}); }
void Module::kill() { emitVoid(spv::Op::OpKill, {}); }

// ---- assembly --------------------------------------------------------------
std::vector<uint32_t> Module::assemble() const {
  std::vector<uint32_t> out;
  out.push_back(spv::MagicNumber);          // 0x07230203
  out.push_back(0x00010300u);               // SPIR-V 1.3 (Vulkan 1.1)
  out.push_back(0);                         // generator (0 = unknown)
  out.push_back(bound_);                    // id bound
  out.push_back(0);                         // schema
  auto app = [&](const std::vector<uint32_t> &s) { out.insert(out.end(), s.begin(), s.end()); };
  app(caps_);
  app(exts_);
  app(extImports_);
  app(memModel_);
  app(entries_);
  app(execModes_);
  app(debug_);
  app(decos_);
  app(typesConsts_);
  app(fnBody_);
  return out;
}

}  // namespace gpu::gcn::spirv
