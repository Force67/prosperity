#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include <memory>
#include <string>
#include <vector>

#include <base/strings/xstring.h>

namespace vfs {

// One regular file inside a container archive, as reported by a backend's
// index. The offset/method fields are backend-private bookkeeping that let a
// later read resume decoding at this entry without re-walking the container,
// and they are what gets written to the on-disk index cache, so a backend must
// be able to decode an entry from these fields alone.
struct ArchiveEntry {
  std::string path;      // archive-relative, '/' separators, no leading slash
  u64 size = 0;          // uncompressed
  u64 packedSize = 0;
  u64 dataOffset = 0;    // backend-private: where the entry's stream begins
  u64 extra = 0;         // backend-private (rar: header offset; zip: local hdr)
  u32 method = 0;        // backend-private compression method id
  u32 crc = 0;
};

// A read-only container backend (rar, zip, ...). One instance is shared by
// every guest thread, so implementations must serialise access to their own
// file handle and decoder state internally.
struct ArchiveBackend {
  virtual ~ArchiveBackend() = default;

  // Walk the container once and append every regular file it holds. Returns
  // false if the container is malformed.
  virtual bool index(std::vector<ArchiveEntry> &out) = 0;

  // Decompress [off, off+len) of one entry into buf. Returns the number of
  // bytes produced (clamped to the entry size, 0 past the end) or -1 on a
  // corrupt stream. Entries are far too large to hold whole (this archive has
  // 3.6 GB ones), so this is the only decode entry point.
  //
  // Backends should keep a per-entry decoder cursor: a forward sequential read
  // must resume rather than restart the stream, or a guest streaming a large
  // file degrades to quadratic decompression.
  virtual i64 extractRange(const ArchiveEntry &entry, void *buf, i64 off,
                           i64 len) = 0;

  // Human-readable backend name for logging ("rar", "zip").
  virtual const char *name() const = 0;
};

// Backend constructors. Each returns nullptr when `path` is not a container of
// that kind; ArchiveFilesystem sniffs by trying them in turn.
std::unique_ptr<ArchiveBackend> openRarBackend(const base::String &path);
std::unique_ptr<ArchiveBackend> openZipBackend(const base::String &path);

} // namespace vfs
