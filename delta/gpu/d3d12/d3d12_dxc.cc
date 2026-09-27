/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * DXC behind a plain interface. This unit sees only DXC's headers: on Linux
 * they bring their own Windows shims (WinAdapter.h), which conflict with
 * vkd3d's, so no D3D12 header may be included here.
 */

#include "gpu/d3d12/d3d12_shader.h"

#if defined(_WIN32)
#include <dxcapi.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <dxc/dxcapi.h>
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#endif

#ifndef DELTA_DXCOMPILER_PATH
#define DELTA_DXCOMPILER_PATH ""
#endif

namespace gpu::d3d12 {

namespace {

DxcCreateInstanceProc g_create = nullptr;

struct ThreadCompiler {
  IDxcCompiler3* compiler = nullptr;
};

ThreadCompiler& Local() {
  thread_local ThreadCompiler tc;
  if (!tc.compiler && g_create)
    g_create(CLSID_DxcCompiler, IID_PPV_ARGS(&tc.compiler));
  return tc;
}

DxcCreateInstanceProc LoadLibraryEntry() {
#if defined(_WIN32)
  HMODULE lib = LoadLibraryW(L"dxcompiler.dll");
  if (!lib)
    return nullptr;
  return reinterpret_cast<DxcCreateInstanceProc>(
      GetProcAddress(lib, "DxcCreateInstance"));
#else
  // The build's own copy first: on NixOS it is on no default search path.
  void* lib = nullptr;
  if (DELTA_DXCOMPILER_PATH[0])
    lib = dlopen(DELTA_DXCOMPILER_PATH, RTLD_NOW | RTLD_LOCAL);
  if (!lib)
    lib = dlopen("libdxcompiler.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib)
    return nullptr;
  return reinterpret_cast<DxcCreateInstanceProc>(
      dlsym(lib, "DxcCreateInstance"));
#endif
}

}  // namespace

bool Dxc::Load() {
  static const bool kOnce = ([] { g_create = LoadLibraryEntry(); }(), true);
  (void)kOnce;
  return g_create && Local().compiler;
}

bool Dxc::Compile(const base::String& hlsl,
                  const base::String& profile,
                  base::Vector<u8>* dxil,
                  base::String* error) {
  ThreadCompiler& tc = Local();
  if (!tc.compiler) {
    *error = "DXC is not loaded";
    return false;
  }
  base::StringW wprofile;
  for (char c : profile)
    wprofile += static_cast<wchar_t>(c);
  // HLSL 2018: SPIRV-Cross relies on its component-wise vector ternary.
  LPCWSTR args[] = {L"-E",   L"main",          L"-T", wprofile.c_str(), L"-HV",
                    L"2018", L"-Qstrip_debug", L"-O3"};
  DxcBuffer source{hlsl.data(), hlsl.size(), DXC_CP_UTF8};
  IDxcResult* result = nullptr;
  HRESULT hr =
      tc.compiler->Compile(&source, args, sizeof(args) / sizeof(args[0]),
                           nullptr, IID_PPV_ARGS(&result));
  if (FAILED(hr) || !result) {
    *error = "DXC Compile failed";
    return false;
  }
  HRESULT status = E_FAIL;
  result->GetStatus(&status);
  IDxcBlobUtf8* errors = nullptr;
  result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
  if (errors && errors->GetStringLength())
    *error =
        base::String(errors->GetStringPointer(), errors->GetStringLength());
  if (errors)
    errors->Release();
  bool ok = false;
  if (SUCCEEDED(status)) {
    IDxcBlob* object = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr);
    if (object && object->GetBufferSize()) {
      const u8* p = static_cast<const u8*>(object->GetBufferPointer());
      dxil->assign(p, p + object->GetBufferSize());
      ok = true;
    }
    if (object)
      object->Release();
  }
  result->Release();
  return ok;
}

}  // namespace gpu::d3d12
