#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <base/strings/xstring.h>

namespace vfs {
struct PkgImpl;

// On-demand reader for a fake-signed PS4 .pkg. Recovers the EKPFS, decrypts the
// PFS and inflates the inner PFSC image lazily: only the blocks actually read
// are decrypted/inflated, so a multi-GB game is never extracted to disk. Retail
// (Sony-signed) pkgs are not supported; same key-free path as pkg_extract.py.
//
// Thread-safe across guest threads: a single mutex guards the shared pkg fd and
// the (otherwise stateless) decrypt chain.
class PkgFilesystem {
public:
  // A regular file inside the image: its byte size and the inner-image block it
  // starts at (file data is stored contiguously from there).
  struct Node {
    uint64_t size = 0;
    uint32_t startBlock = 0;
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
  int64_t read(const Node &node, void *buf, int64_t off, int64_t len);

  // Collect every file path in the image (tooling / debugging).
  void paths(std::vector<std::string> &out) const;

private:
  std::unique_ptr<PkgImpl> impl_;
};
} // namespace vfs
