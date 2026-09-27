#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "io/file.h"

namespace io {
class File;
}

namespace formats {
struct PupHeader {
  u32 magic;
  u32 unk;
  u8 content_type;
  u8 product_type;
  u16 pad;
  u16 header_size;
  u16 sig_size;
  u32 size_self;
  u32 pad2;
  u16 num_segments;
  u16 unk2;
  u32 pad3;
};

struct PupEntry {
  u32 flags;
  u32 unk;
  u64 offset;
  u64 size_compressed;
  u64 size_uncompressed;
};

static_assert(sizeof(PupHeader) == 32);
static_assert(sizeof(PupEntry) == 32);

// Outer container magics. Both use the same 32-byte header + 32-byte entry
// table; only the magic (and the PS5 block-compression layout) differ.
constexpr u32 kPupMagicPS4 = 0x1D3D154Fu;
constexpr u32 kPupMagicPS5 = 0xEEF51454u;

// PS4/PS5 firmware (.PUP) reader: outer container here, segments written by
// ExtractAll(). Retail PUPs are encrypted; decrypted images (*.PUP.dec) recover
// container segments in full (FS images, SLB2, nested PUPs), but the SELF
// modules inside still need the per-title crypto chain to become loadable
// .sprx.
class PupReader {
 public:
  explicit PupReader(const base::String&);

  bool Load();

  // Extract every non-special segment into outDir (must exist), named by
  // firmware name or segment_<id>.<type>; inflate zlib segments when possible.
  // Returns a summary and sets looksEncrypted when segments don't parse as
  // plaintext.
  base::String ExtractAll(const base::String& out_dir, bool& looks_encrypted);

  int SegmentCount() const { return header_.num_segments; }
  bool ps5() const { return is_ps5_; }

 private:
  bool InflateEntry(const PupEntry&,
                    base::Vector<u8>& in,
                    base::Vector<u8>& out);

  // PS5 firmware uses a block-compressed layout: large segments are split into
  // fixed-size blocks, each stored either raw or zlib-compressed, with a paired
  // block table giving the per-block (offset,size) extents. ExtractAllPS5()
  // streams those out without holding a whole (multi-GB) segment in memory.
  base::String ExtractAllPS5(const base::String& out_dir);
  int TableForData(size_t data_idx) const;
  bool ExtractPS5Segment(const PupEntry&,
                         size_t idx,
                         const base::String& out_dir,
                         base::String& summary);

  io::File file_;
  PupHeader header_{};
  base::Vector<PupEntry> entries_;
  bool is_ps5_ = false;
};
}  // namespace formats
