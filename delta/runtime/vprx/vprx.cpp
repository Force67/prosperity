
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "vprx.h"
#include <crypto/sha1.h>
#include <base/containers/vector.h>

namespace runtime {
static base::Vector<const modInfo *> vprxTable;

// HLE-module anchors. Each vprx HLE module's _api.cpp defines one of these; we
// reference them here so the linker keeps those archive members (otherwise the
// MODULE_INIT static initializers never run and the HLE tables stay empty).
extern "C" int vprx_anchor_libSceVideoOut;
extern "C" int vprx_anchor_libSceGnmDriver;
static volatile int *const vprx_anchors[] = {&vprx_anchor_libSceVideoOut,
                                             &vprx_anchor_libSceGnmDriver};

void vprx_init() {
  // Touch the anchors so the references aren't optimized away.
  int sum = 0;
  for (auto *a : vprx_anchors)
    sum += *a;
  (void)sum;
  utl::init_function::init();
}

void vprx_reg(const modInfo *info) { vprxTable.push_back(info); }

uintptr_t vprx_get(const char *lib, uint64_t hid) {
  const modInfo *table = nullptr;

  // find the right table
  for (const auto &t : vprxTable) {
    if (std::strcmp(lib, t->namePtr) == 0) {
      table = t;
      break;
    }
  }

  if (table) {
    // search the table
    for (int i = 0; i < table->funcCount; i++) {
      auto *f = &table->funcNodes[i];
      if (f->hashId == hid) {
        return reinterpret_cast<uintptr_t>(f->address);
      }
    }
  }

  return 0;
}

const char base64Lookup[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

// base64 fast lookup
bool decode_nid(const char *subset, size_t len, uint64_t &out) {
  for (size_t i = 0; i < len; i++) {
    auto pos = std::strchr(base64Lookup, subset[i]);

    // invalid NID?
    if (!pos) {
      return false;
    }

    auto offset = static_cast<uint32_t>(pos - base64Lookup);

    // max NID is 11
    if (i < 10) {
      out <<= 6;
      out |= offset;
    } else {
      out <<= 4;
      out |= (offset >> 2);
    }
  }

  return true;
}

static void obfuscate_sym(uint64_t in, uint8_t *out, size_t xlen) {
  out[xlen--] = 0;
  out[xlen--] = base64Lookup[(in & 0xF) * 4];
  uint64_t exp = in >> 4;
  while (exp != 0) {
    out[xlen--] = base64Lookup[exp & 0x3F];
    exp = exp >> 6;
  }
}

void encode_nid(const char *name, uint8_t *x) {
  static const char suffix[] =
      "\x51\x8D\x64\xA6\x35\xDE\xD8\xC1\xE6\xB0\x39\xB1\xC3\xE5\x52\x30";

  uint8_t sha[20]{};
  sha1_context ctx;

  sha1_starts(&ctx);
  sha1_update(&ctx, reinterpret_cast<const uint8_t *>(name), std::strlen(name));
  sha1_update(&ctx, reinterpret_cast<const uint8_t *>(suffix),
              std::strlen(suffix));
  sha1_finish(&ctx, sha);

  /*the rest is ignored*/
  uint64_t target = *(uint64_t *)(&sha);

  // uint8_t out[11]{};
  obfuscate_sym(target, x, 11);
}
} // namespace runtime