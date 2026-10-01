// NIDs: the 11-character names a module's imports and exports go by, the
// SHA-1 of the symbol name (salted) folded into a base64 alphabet.

#include "runtime/vprx/nid.h"

#include <cstring>

#include "base/containers/array.h"
#include "crypto/sha1.h"

namespace runtime::nid {
namespace {

constexpr char kBase64Lookup[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

void ObfuscateSym(u64 in, u8* out, size_t xlen) {
  out[xlen--] = 0;
  out[xlen--] = kBase64Lookup[(in & 0xF) * 4];
  u64 exp = in >> 4;
  while (exp != 0) {
    out[xlen--] = kBase64Lookup[exp & 0x3F];
    exp = exp >> 6;
  }
}

}  // namespace

bool Decode(const char* subset, size_t len, u64& out) {
  static const auto kIndex = [] {
    base::Array<u8, 256> index;
    index.fill(0xff);
    for (u32 i = 0; i < 64; i++)
      index[static_cast<u8>(kBase64Lookup[i])] = static_cast<u8>(i);
    return index;
  }();
  for (size_t i = 0; i < len; i++) {
    const u32 offset = kIndex[static_cast<u8>(subset[i])];
    if (offset == 0xff)
      return false;

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

void Encode(const char* name, u8* x) {
  static const char kSuffix[] =
      "\x51\x8D\x64\xA6\x35\xDE\xD8\xC1\xE6\xB0\x39\xB1\xC3\xE5\x52\x30";

  u8 sha[20]{};
  sha1_context ctx;

  Sha1Starts(&ctx);
  Sha1Update(&ctx, reinterpret_cast<const u8*>(name), std::strlen(name));
  Sha1Update(&ctx, reinterpret_cast<const u8*>(kSuffix), std::strlen(kSuffix));
  Sha1Finish(&ctx, sha);

  /*the rest is ignored*/
  u64 target = *(u64*)(&sha);

  // u8 out[11]{};
  ObfuscateSym(target, x, 11);
}

}  // namespace runtime::nid
