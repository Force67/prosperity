/*
 * Standalone validation harness for the SPIR-V builder (spv_emit). Builds a
 * passthrough VS and FS and validates the emitted binary with SPIRV-Tools.
 * Not part of the emulator build; compiled directly:
 *   c++ -std=c++20 $(pkg-config --cflags SPIRV-Headers SPIRV-Tools-shared) \
 *       test_spv.cpp spv_emit.cpp $(pkg-config --libs SPIRV-Tools-shared) -o /tmp/test_spv
 */

#include "spv_emit.h"
#include "spv_post.h"

#include <cstdio>
#include <spirv-tools/libspirv.h>
#include <spirv/unified1/GLSL.std.450.h>

using namespace gpu::gcn::spirv;

static bool validate(const std::vector<uint32_t> &spv, const char *tag) {
  spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_1);
  spv_diagnostic diag = nullptr;
  spv_const_binary_t bin{spv.data(), spv.size()};
  spv_result_t r = spvValidate(ctx, &bin, &diag);
  bool ok = r == SPV_SUCCESS;
  std::printf("[%s] %u words -> %s\n", tag, (unsigned)spv.size(), ok ? "VALID" : "INVALID");
  if (!ok && diag) std::printf("    %s\n", diag->error);
  spvDiagnosticDestroy(diag);
  spvContextDestroy(ctx);
  return ok;
}

static std::vector<uint32_t> buildVs() {
  Module m;
  Id tVoid = m.typeVoid();
  Id tFn = m.typeFunction(tVoid);
  Id tF = m.typeFloat(32);
  Id tV4 = m.typeVec(tF, 4);
  Id pOutV4 = m.typePointer(spv::StorageClass::Output, tV4);
  Id posOut = m.variable(pOutV4, spv::StorageClass::Output);
  m.decorate(posOut, spv::Decoration::BuiltIn, {(uint32_t)spv::BuiltIn::Position});
  Id main = m.beginFunction(tVoid, tFn);
  Id c0 = m.constF32(0.f), c1 = m.constF32(1.f);
  Id pos = m.constComposite(tV4, {c0, c0, c0, c1});
  m.store(posOut, pos);
  m.returnVoid();
  m.endFunction();
  m.entryPoint(spv::ExecutionModel::Vertex, main, "main", {posOut});
  return m.assemble();
}

static std::vector<uint32_t> buildFs() {
  Module m;
  Id tVoid = m.typeVoid();
  Id tFn = m.typeFunction(tVoid);
  Id tF = m.typeFloat(32);
  Id tV4 = m.typeVec(tF, 4);
  Id pOutV4 = m.typePointer(spv::StorageClass::Output, tV4);
  Id colorOut = m.variable(pOutV4, spv::StorageClass::Output);
  m.decorate(colorOut, spv::Decoration::Location, {0});
  Id main = m.beginFunction(tVoid, tFn);
  Id c1 = m.constF32(1.f);
  Id col = m.constComposite(tV4, {c1, c1, c1, c1});
  m.store(colorOut, col);
  m.returnVoid();
  m.endFunction();
  m.entryPoint(spv::ExecutionModel::Fragment, main, "main", {colorOut});
  m.execMode(main, spv::ExecutionMode::OriginUpperLeft);
  return m.assemble();
}

// A register-VM-style VS: a Private uint[8] "vgpr" file, written then read back,
// position computed via float<->uint bitcasts. Exercises exactly what the GCN
// translator emits; legalization must promote vgpr[] to SSA.
static std::vector<uint32_t> buildRegVmVs() {
  Module m;
  Id tVoid = m.typeVoid(), tFn = m.typeFunction(tVoid);
  Id tU = m.typeInt(32, false), tF = m.typeFloat(32), tV4 = m.typeVec(tF, 4);
  Id arr = m.typeArray(tU, 8);
  Id pPrivArr = m.typePointer(spv::StorageClass::Private, arr);
  Id pPrivU = m.typePointer(spv::StorageClass::Private, tU);
  Id vgpr = m.variable(pPrivArr, spv::StorageClass::Private);
  m.name(vgpr, "vgpr");
  Id pOutV4 = m.typePointer(spv::StorageClass::Output, tV4);
  Id posOut = m.variable(pOutV4, spv::StorageClass::Output);
  m.decorate(posOut, spv::Decoration::BuiltIn, {(uint32_t)spv::BuiltIn::Position});
  Id main = m.beginFunction(tVoid, tFn);
  // vgpr[0] = bitcast(1.0); vgpr[1] = bitcast(0.5)
  m.store(m.accessChain(pPrivU, vgpr, {m.constU32(0)}), m.bitcast(tU, m.constF32(1.f)));
  m.store(m.accessChain(pPrivU, vgpr, {m.constU32(1)}), m.bitcast(tU, m.constF32(0.5f)));
  Id x = m.bitcast(tF, m.load(tU, m.accessChain(pPrivU, vgpr, {m.constU32(0)})));
  Id y = m.bitcast(tF, m.load(tU, m.accessChain(pPrivU, vgpr, {m.constU32(1)})));
  Id pos = m.compositeConstruct(tV4, {x, y, m.constF32(0.f), m.constF32(1.f)});
  m.store(posOut, pos);
  m.returnVoid();
  m.endFunction();
  m.entryPoint(spv::ExecutionModel::Vertex, main, "main", {posOut});
  return m.assemble();
}

int main() {
  bool ok = true;
  ok &= validate(buildVs(), "vs");
  ok &= validate(buildFs(), "fs");
  // register-VM shader: validate naive, optimize, re-validate, check it shrank.
  auto rv = buildRegVmVs();
  ok &= validate(rv, "regvm-naive");
  auto opt = optimize(rv);
  bool optOk = validate(opt, "regvm-opt");
  ok &= optOk;
  std::printf("[regvm] %u words -> %u words after opt\n", (unsigned)rv.size(), (unsigned)opt.size());
  std::printf("%s\n", ok ? "ALL VALID" : "FAILED");
  return ok ? 0 : 1;
}
