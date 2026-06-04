#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <utl/file.h>
#include <base/containers/vector.h>
#include <base/strings/xstring.h>

namespace utl {
class File;
}

namespace vfs {
struct pup_header {
  uint32_t magic;
  uint32_t unk;
  uint8_t contentType;
  uint8_t productType;
  uint16_t pad;
  uint16_t headerSize;
  uint16_t sigSize;
  uint32_t sizeSELF;
  uint32_t pad2;
  uint16_t numSegments;
  uint16_t unk2;
  uint32_t pad3;
};

struct pup_entry {
  uint32_t flags;
  uint32_t unk;
  uint64_t offset;
  uint64_t sizeCompressed;
  uint64_t sizeUncompressed;
};

static_assert(sizeof(pup_header) == 32);
static_assert(sizeof(pup_entry) == 32);

// Reader for a PS4 firmware update (.PUP). The outer container (magic
// 0x1D3D154F) is parsed here; segments are written out by extractAll(). Note
// retail PUPs are encrypted (and their inner modules are encrypted SELFs in a
// filesystem image), so without the SAMU/SELF crypto chain extractAll() can
// recover the container segments but not produce loadable .sprx modules. See
// the summary string it returns.
class pupReader {
public:
  explicit pupReader(const base::String &);

  bool load();

  // Extract every non-special segment into outDir (which must already exist),
  // named by its known firmware name or segment_<id>.bin. zlib-compressed
  // segments are inflated when possible. Returns a human-readable multi-line
  // summary; sets `looksEncrypted` when the segments don't parse as plaintext
  // (i.e. the PUP needs decryption we can't do here).
  base::String extractAll(const base::String &outDir, bool &looksEncrypted);

  int segmentCount() const { return header.numSegments; }

private:
  bool inflateEntry(const pup_entry &, base::Vector<uint8_t> &in,
                    base::Vector<uint8_t> &out);

  utl::File file;
  pup_header header{};
  base::Vector<pup_entry> entries;
};
} // namespace vfs
