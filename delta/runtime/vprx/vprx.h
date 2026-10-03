#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>

#include "base/arch.h"
#include "logger/logger.h"
#include "runtime/vprx/init_function.h"

namespace runtime::vprx {

// One HLE export: the host function a guest import of `nid` binds to.
struct ExportEntry {
  u64 nid;
  const void* address;
};

// The exports an HLE module provides for one library.
struct ExportTable {
  const ExportEntry* entries;
  size_t count;
  const char* library;
};

void Init();
void Register(const ExportTable* table);
uintptr_t Lookup(const char* lib, u64 nid);
// Table lookup that ignores the LLE-by-default policy gate (UseHleShim). The
// PS5 import resolver uses it to force libSceVideoOut onto the HLE shim (its
// LLE port backend never registers in our env, so sceVideoOutOpen fails).
uintptr_t LookupForced(const char* lib, u64 nid);

// PS5-only HLE alias tables. Prospero modules export some functions under NIDs
// that differ from the PS4 ABI (e.g. sceVideoOutRegisterBuffers). Rather than
// pollute the PS4 tables, PS5 aliases register here (runtime/vprx/ps5/*) and
// are consulted first by LookupForced, which is PS5-only. PS4 paths never
// touch it.
void RegisterPs5(const ExportTable* table);

}  // namespace runtime::vprx

#define MODULE_INIT(tname)                                       \
  static const runtime::vprx::ExportTable info_##tname{          \
      kExports, sizeof(kExports) / sizeof(kExports[0]), #tname}; \
  static runtime::InitFunction init_##tname(                     \
      []() { runtime::vprx::Register(&info_##tname); })

// Register a PS5-only NID alias table under the module name `tname`. Same shape
// as MODULE_INIT but lands in the separate PS5 registry (RegisterPs5).
#define MODULE_INIT_PS5(tname)                                   \
  static const runtime::vprx::ExportTable info_ps5_##tname{      \
      kExports, sizeof(kExports) / sizeof(kExports[0]), #tname}; \
  static runtime::InitFunction init_ps5_##tname(                 \
      []() { runtime::vprx::RegisterPs5(&info_ps5_##tname); })
