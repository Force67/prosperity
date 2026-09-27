#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"

#include <base/strings/xstring.h>
#include <base/containers/vector.h>
#include <base/memory/unique_pointer.h>

namespace vfs {
struct Ufs2Impl;

// Read-only reader for a UFS2 (FreeBSD FFSv2) image, as produced for PS5 game
// backups (*.ffpkg, made by newfs -O 2 -b 32768 -f 4096 over a decrypted dump).
// The file data inside is already plaintext, so unlike a retail .pkg no crypto
// is involved: this just walks the superblock/inodes/dirents and reads file
// bytes on demand (a multi-GB image is never extracted to disk).
//
// Same public shape as PkgFilesystem so it drops into the same VirtualProvider.
class Ufs2Filesystem {
public:
  // A regular file inside the image, identified by its UFS2 inode number.
  struct Node {
    u64 size = 0;
    u32 inode = 0;
  };

  explicit Ufs2Filesystem(const base::String &imagePath);
  ~Ufs2Filesystem();

  Ufs2Filesystem(const Ufs2Filesystem &) = delete;
  Ufs2Filesystem &operator=(const Ufs2Filesystem &) = delete;

  bool valid() const;

  // Look up a file by image-relative path with a leading '/', e.g. "/eboot.bin"
  // or "/decrypted/eboot.bin". Returns nullptr if absent or a directory.
  const Node *find(const char *relPath) const;

  // Read up to len bytes of a file starting at byte offset off. Returns the
  // number of bytes read (clamped to the file size, 0 past the end), or -1 on
  // error. Sparse holes read back as zeros.
  i64 read(const Node &node, void *buf, i64 off, i64 len);

  // Collect every regular-file path in the image (leading '/'), for tooling.
  void paths(base::Vector<base::String> &out) const;

private:
  base::UniquePointer<Ufs2Impl> impl_;
};
} // namespace vfs
