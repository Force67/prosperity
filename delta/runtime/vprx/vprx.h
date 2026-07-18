#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <logger/logger.h>
#include <string>
#include <utl/init_func.h>

#include <base.h>

namespace runtime {
struct funcInfo {
  uint64_t hashId;
  const void *address;
};

struct modInfo {
  funcInfo *funcNodes;
  size_t funcCount;
  const char *namePtr;
};

void vprx_init();
void vprx_reg(const modInfo *);
uintptr_t vprx_get(const char *lib, uint64_t hid);
// Table lookup that ignores the LLE-by-default policy gate (useHleShim). The PS5
// import resolver uses it to force libSceVideoOut onto the HLE shim (its LLE port
// backend never registers in our env, so sceVideoOutOpen fails).
uintptr_t vprx_get_forced(const char *lib, uint64_t hid);

// PS5-only HLE alias tables. Prospero modules export some functions under NIDs
// that differ from the PS4 ABI (e.g. sceVideoOutRegisterBuffers). Rather than
// pollute the PS4 tables, PS5 aliases register here (runtime/vprx/ps5/*) and are
// consulted first by vprx_get_forced, which is PS5-only. PS4 paths never touch it.
void vprx_reg_ps5(const modInfo *);

bool decode_nid(const char *subset, size_t len, uint64_t &);
void encode_nid(const char *symName, uint8_t *out);
}

#define MODULE_INIT(tname)                                                     \
  \
static const runtime::modInfo info_##tname{                                    \
      (runtime::funcInfo *)&functions,                                         \
      (sizeof(functions) / sizeof(runtime::funcInfo)), #tname};                \
  \
static utl::init_function init_##tname(                                        \
      []() { runtime::vprx_reg(&info_##tname); })

// Register a PS5-only NID alias table under the module name `tname`. Same shape
// as MODULE_INIT but lands in the separate PS5 registry (vprx_reg_ps5).
#define MODULE_INIT_PS5(tname)                                                  \
  \
static const runtime::modInfo info_ps5_##tname{                                \
      (runtime::funcInfo *)&functions,                                         \
      (sizeof(functions) / sizeof(runtime::funcInfo)), #tname};                \
  \
static utl::init_function init_ps5_##tname(                                    \
      []() { runtime::vprx_reg_ps5(&info_ps5_##tname); })