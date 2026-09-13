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
struct PkgImpl;

// On-demand fake-signed PS4 .pkg reader: recover the EKPFS, decrypt the PFS, inflate
// the PFSC image lazily (only blocks actually read), so a multi-GB game never extracts
// to disk. Retail pkgs unsupported; same key-free path as pkg_extract.py. Thread-safe
// across guest threads (one mutex over the shared fd + stateless decrypt chain).
class PkgFilesystem {
public:
  // A regular file inside the image: its byte size and the inner-image block it
  // starts at (file data is stored contiguously from there).
  struct Node {
    u64 size = 0;
    u32 startBlock = 0;
  };

  explicit PkgFilesystem(const base::String &pkgPath);
  ~PkgFilesystem();

  PkgFilesystem(const PkgFilesystem &) = delete;
  PkgFilesystem &operator=(const PkgFilesystem &) = delete;

  bool valid() const;

  // Look up a file by its image-relative path with a leading '/', e.g.
  // "/eboot.bin" or "/sce_sys/param.sfo". Returns nullptr if absent.
  const Node *find(const char *relPath) const;

  // Read up to len bytes of a file starting at byte offset off. Returns the
  // number of bytes read (clamped to the file size, 0 past the end), or -1 on
  // error.
  i64 read(const Node &node, void *buf, i64 off, i64 len);

  // Collect every file path in the image (tooling / debugging).
  void paths(std::vector<std::string> &out) const;

  // Read a well-known outer-PKG entry by id (param.sfo = 0x1000, icon0.png = 0x1200)
  // from the header's entry table, which sits outside the encrypted PFS and works even
  // when the inner image didn't decrypt. Bytes read, or -1 if absent.
  i64 readPkgEntry(u32 entryId, std::vector<u8> &out);

private:
  std::unique_ptr<PkgImpl> impl_;
};
} // namespace vfs
