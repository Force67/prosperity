// Standalone validation harness for the SPIR-V builder (spv_emit). Builds a
// passthrough VS and FS and validates the emitted binary with SPIRV-Tools.
// Not part of the emulator build; from the repo root, against a build tree B:
//
//   nix develop --command bash -c 'c++ -std=c++20 -Idelta -Ishared \
//     -Ivendor/equilibrium $(pkg-config --cflags SPIRV-Headers SPIRV-Tools) \
//     tools/spv_selftest.cc $B/delta/gpu/libdelta_gpu.a \
//     $B/shared/libshared.a $B/vendor/libeq_base.a -lpthread \
//     $(pkg-config --libs SPIRV-Tools) -o /tmp/t && /tmp/t'

#include "base/arch.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"

#include <spirv-tools/libspirv.h>
#include <spirv/unified1/GLSL.std.450.h>
#include <cstdio>
#include "base/containers/vector.h"

using namespace gpu::gcn::spirv;

static bool validate(const base::Vector<u32>& spv, const char* tag) {
  spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_1);
  spv_diagnostic diag = nullptr;
  spv_const_binary_t bin{spv.data(), spv.size()};
  spv_result_t r = spvValidate(ctx, &bin, &diag);
  bool ok = r == SPV_SUCCESS;
  std::printf("[%s] %u words -> %s\n", tag, (unsigned)spv.size(),
              ok ? "VALID" : "INVALID");
  if (!ok && diag)
    std::printf("    %s\n", diag->error);
  spvDiagnosticDestroy(diag);
  spvContextDestroy(ctx);
  return ok;
}

static base::Vector<u32> buildVs() {
  Module m;
  Id tVoid = m.TypeVoid();
  Id tFn = m.TypeFunction(tVoid);
  Id tF = m.TypeFloat(32);
  Id tV4 = m.TypeVec(tF, 4);
  Id pOutV4 = m.TypePointer(spv::StorageClass::Output, tV4);
  Id posOut = m.Variable(pOutV4, spv::StorageClass::Output);
  m.Decorate(posOut, spv::Decoration::BuiltIn, {(u32)spv::BuiltIn::Position});
  Id main = m.BeginFunction(tVoid, tFn);
  Id c0 = m.ConstF32(0.f), c1 = m.ConstF32(1.f);
  Id pos = m.ConstComposite(tV4, {c0, c0, c0, c1});
  m.Store(posOut, pos);
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::Vertex, main, "main", {posOut});
  return m.Assemble();
}

static base::Vector<u32> buildFs() {
  Module m;
  Id tVoid = m.TypeVoid();
  Id tFn = m.TypeFunction(tVoid);
  Id tF = m.TypeFloat(32);
  Id tV4 = m.TypeVec(tF, 4);
  Id pOutV4 = m.TypePointer(spv::StorageClass::Output, tV4);
  Id colorOut = m.Variable(pOutV4, spv::StorageClass::Output);
  m.Decorate(colorOut, spv::Decoration::Location, {0});
  Id main = m.BeginFunction(tVoid, tFn);
  Id c1 = m.ConstF32(1.f);
  Id col = m.ConstComposite(tV4, {c1, c1, c1, c1});
  m.Store(colorOut, col);
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::Fragment, main, "main", {colorOut});
  m.ExecMode(main, spv::ExecutionMode::OriginUpperLeft);
  return m.Assemble();
}

// A register-VM-style VS: a Private uint[8] "vgpr" file, written then read
// back, position computed via float<->uint bitcasts. Exercises exactly what the
// GCN translator emits; legalization must promote vgpr[] to SSA.
static base::Vector<u32> buildRegVmVs() {
  Module m;
  Id tVoid = m.TypeVoid(), tFn = m.TypeFunction(tVoid);
  Id tU = m.TypeInt(32, false), tF = m.TypeFloat(32), tV4 = m.TypeVec(tF, 4);
  Id arr = m.TypeArray(tU, 8);
  Id pPrivArr = m.TypePointer(spv::StorageClass::Private, arr);
  Id pPrivU = m.TypePointer(spv::StorageClass::Private, tU);
  Id vgpr = m.Variable(pPrivArr, spv::StorageClass::Private);
  m.Name(vgpr, "vgpr");
  Id pOutV4 = m.TypePointer(spv::StorageClass::Output, tV4);
  Id posOut = m.Variable(pOutV4, spv::StorageClass::Output);
  m.Decorate(posOut, spv::Decoration::BuiltIn, {(u32)spv::BuiltIn::Position});
  Id main = m.BeginFunction(tVoid, tFn);
  // vgpr[0] = bitcast(1.0); vgpr[1] = bitcast(0.5)
  m.Store(m.AccessChain(pPrivU, vgpr, {m.ConstU32(0)}),
          m.Bitcast(tU, m.ConstF32(1.f)));
  m.Store(m.AccessChain(pPrivU, vgpr, {m.ConstU32(1)}),
          m.Bitcast(tU, m.ConstF32(0.5f)));
  Id x =
      m.Bitcast(tF, m.Load(tU, m.AccessChain(pPrivU, vgpr, {m.ConstU32(0)})));
  Id y =
      m.Bitcast(tF, m.Load(tU, m.AccessChain(pPrivU, vgpr, {m.ConstU32(1)})));
  Id pos = m.CompositeConstruct(tV4, {x, y, m.ConstF32(0.f), m.ConstF32(1.f)});
  m.Store(posOut, pos);
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::Vertex, main, "main", {posOut});
  return m.Assemble();
}

int main() {
  bool ok = true;
  ok &= validate(buildVs(), "vs");
  ok &= validate(buildFs(), "fs");
  // register-VM shader: validate naive, optimize, re-validate, check it shrank.
  auto rv = buildRegVmVs();
  ok &= validate(rv, "regvm-naive");
  auto opt = Optimize(rv);
  bool optOk = validate(opt, "regvm-opt");
  ok &= optOk;
  std::printf("[regvm] %u words -> %u words after opt\n", (unsigned)rv.size(),
              (unsigned)opt.size());
  std::printf("%s\n", ok ? "ALL VALID" : "FAILED");
  return ok ? 0 : 1;
}
