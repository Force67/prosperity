#pragma once

#include <cstddef>

#include "base/arch.h"

namespace runtime::nid {

// Parse the base64 NID text `text[0..len)` into its 64-bit value.
bool Decode(const char* text, size_t len, u64& out);
// Write the 11-character NID of `symbol` plus a terminator into `out[12]`.
void Encode(const char* symbol, u8* out);

}  // namespace runtime::nid
