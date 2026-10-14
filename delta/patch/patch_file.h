/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Game patch files: YAML, one block per executable, keyed by the hash of its
// loaded segments (see ImageKey in game_patches.h).
//
//   Version: 1.0
//
//   EBOOT-0123...cdef:
//     "60 FPS":
//       Games:
//         "P.T.":
//           CUSA01127: [ 01.00 ]
//       Author: someone
//       Notes: free text
//       Patch:
//         - [ byte, 0xd8a714, 0 ]
//
// Addresses are ELF vaddrs. Entry types: byte, le16, le32, le64, be16, be32,
// be64, lef32, lef64, utf8 (no terminator), c_utf8 (NUL-terminated) and bytes
// (hex digits in memory order). Every key but Patch is informational.

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"

namespace patch {

struct Write {
  u64 address;
  base::Vector<u8> bytes;
};

struct Patch {
  base::String image;
  base::String name;
  base::Vector<Write> writes;
};

// Appends every patch in `text` to `out`. Returns false with `error` naming
// the offending line when the text is not in the format above.
bool ParsePatchFile(base::StringRef text,
                    base::Vector<Patch>* out,
                    base::String* error);

// Writes `patch` into the image whose vaddr 0 lives at host address `base`.
// Writes nothing and returns false when any write falls outside [0, end).
bool ApplyPatch(const Patch& patch, uintptr_t base, u64 end);

}  // namespace patch
