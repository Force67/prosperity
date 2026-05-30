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