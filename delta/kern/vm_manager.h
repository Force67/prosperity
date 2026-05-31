#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>

#include <base/containers/vector.h>

#include <utl/mem.h>

namespace krnl {
struct procInfo;

using mprot = utl::pageProtection;
using alloct = utl::allocationType;

struct pageInfo {
  uint8_t *ptr;
  size_t size;
  mprot prot;
  const char *name = nullptr;

  pageInfo(uint8_t *p, size_t s, mprot mp) : ptr(p), size(s), prot(mp) {}
};

class vmManager {
public:
  vmManager(procInfo &);
  ~vmManager();

  bool init();
  void add(uint8_t *ptr, size_t size, mprot);
  pageInfo *get(uint8_t *ptr);

  // true if [ptr, ptr+size) hits a tracked mapping
  bool overlaps(uint8_t *ptr, size_t size) const;

  uint8_t *mapMemory(uint8_t *preference, size_t size, utl::pageProtection);
  void unmapRtMemory(uint8_t *);

private:
  procInfo &pinfo;

  size_t codeMemTotal{0};
  size_t rtMemTotal{0};

  base::Vector<pageInfo> codePages;
  base::Vector<pageInfo> rtPages;
};
}
