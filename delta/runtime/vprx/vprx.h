#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "logger/logger.h"
#include "runtime/vprx/init_function.h"

namespace runtime {
struct FuncInfo {
  u64 hash_id;
  const void* address;
};

struct ModInfo {
  FuncInfo* func_nodes;
  size_t func_count;
  const char* name_ptr;
};

void VprxInit();
void VprxReg(const ModInfo*);
uintptr_t VprxGet(const char* lib, u64 hid);
// Table lookup that ignores the LLE-by-default policy gate (useHleShim). The
// PS5 import resolver uses it to force libSceVideoOut onto the HLE shim (its
// LLE port backend never registers in our env, so sceVideoOutOpen fails).
uintptr_t VprxGetForced(const char* lib, u64 hid);

// PS5-only HLE alias tables. Prospero modules export some functions under NIDs
// that differ from the PS4 ABI (e.g. sceVideoOutRegisterBuffers). Rather than
// pollute the PS4 tables, PS5 aliases register here (runtime/vprx/ps5/*) and
// are consulted first by VprxGetForced, which is PS5-only. PS4 paths never
// touch it.
void VprxRegPs5(const ModInfo*);

bool DecodeNid(const char* subset, size_t len, u64&);
void EncodeNid(const char* sym_name, u8* out);
}  // namespace runtime

#define MODULE_INIT(tname)                                      \
                                                                \
  static const runtime::ModInfo info_##tname{                   \
      (runtime::FuncInfo*)&functions,                           \
      (sizeof(functions) / sizeof(runtime::FuncInfo)), #tname}; \
                                                                \
  static runtime::InitFunction init_##tname(                    \
      []() { runtime::VprxReg(&info_##tname); })

// Register a PS5-only NID alias table under the module name `tname`. Same shape
// as MODULE_INIT but lands in the separate PS5 registry (VprxRegPs5).
#define MODULE_INIT_PS5(tname)                                  \
                                                                \
  static const runtime::ModInfo info_ps5_##tname{               \
      (runtime::FuncInfo*)&functions,                           \
      (sizeof(functions) / sizeof(runtime::FuncInfo)), #tname}; \
                                                                \
  static runtime::InitFunction init_ps5_##tname(                \
      []() { runtime::VprxRegPs5(&info_ps5_##tname); })