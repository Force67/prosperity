#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <utl/file.h>
#include "base/arch.h"
#include <base/containers/vector.h>
#include <base/strings/xstring.h>

namespace utl {
class File;
}

namespace vfs {
struct pup_header {
  u32 magic;
  u32 unk;
  u8 contentType;
  u8 productType;
  u16 pad;
  u16 headerSize;
  u16 sigSize;
  u32 sizeSELF;
  u32 pad2;
  u16 numSegments;
  u16 unk2;
  u32 pad3;
};

struct pup_entry {
  u32 flags;
  u32 unk;
  u64 offset;
  u64 sizeCompressed;
  u64 sizeUncompressed;
};

static_assert(sizeof(pup_header) == 32);
static_assert(sizeof(pup_entry) == 32);

// Outer container magics. Both use the same 32-byte header + 32-byte entry
// table; only the magic (and the PS5 block-compression layout) differ.
constexpr u32 kPupMagicPS4 = 0x1D3D154Fu;
constexpr u32 kPupMagicPS5 = 0xEEF51454u;

// PS4/PS5 firmware (.PUP) reader: outer container here, segments written by
// extractAll(). Retail PUPs are encrypted; decrypted images (*.PUP.dec) recover
// container segments in full (FS images, SLB2, nested PUPs), but the SELF modules
// inside still need the per-title crypto chain to become loadable .sprx.
class pupReader {
public:
  explicit pupReader(const base::String &);

  bool load();

  // Extract every non-special segment into outDir (must exist), named by firmware name
  // or segment_<id>.<type>; inflate zlib segments when possible. Returns a summary and
  // sets looksEncrypted when segments don't parse as plaintext.
  base::String extractAll(const base::String &outDir, bool &looksEncrypted);

  int segmentCount() const { return header.numSegments; }
  bool ps5() const { return isPS5; }

private:
  bool inflateEntry(const pup_entry &, base::Vector<u8> &in,
                    base::Vector<u8> &out);

  // PS5 firmware uses a block-compressed layout: large segments are split into
  // fixed-size blocks, each stored either raw or zlib-compressed, with a paired
  // block table giving the per-block (offset,size) extents. extractAllPS5()
  // streams those out without holding a whole (multi-GB) segment in memory.
  base::String extractAllPS5(const base::String &outDir);
  int tableForData(size_t dataIdx) const;
  bool extractPS5Segment(const pup_entry &, size_t idx,
                         const base::String &outDir, base::String &summary);

  utl::File file;
  pup_header header{};
  base::Vector<pup_entry> entries;
  bool isPS5 = false;
};
} // namespace vfs
