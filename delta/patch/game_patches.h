/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/strings/xstring.h"

namespace patch {

struct Segment {
  const u8* data;
  u64 size;
};

// "EBOOT-<sha1>" over the file bytes of the loaded segments, in program
// header order: the key a patch file names an executable by.
base::String ImageKey(const Segment* segments, mem_size count);

// Applies the patches DELTA_PATCHES names from DELTA_PATCH_FILE to the main
// image (vaddr 0 at host address `base`, mapped up to vaddr `end`).
void PatchMainImage(uintptr_t base,
                    u64 end,
                    const Segment* segments,
                    mem_size count);

}  // namespace patch
