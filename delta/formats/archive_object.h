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
struct ArchiveImpl;

// On-demand reader for a game shipped as a plain compressed container (.rar,
// .zip). Entries are decompressed lazily, so a 54 GB archive is never extracted
// to disk, which matters because a host with room for the extracted game is not
// something we can assume.
//
// A dump archived this way almost always wraps the game in one top-level
// directory (PPSA01342-app/...). That wrapper is stripped at index time so the
// paths here start at the game root and /app0 lines up with what the guest
// expects.
//
// Same public shape as PkgFilesystem and Ufs2Filesystem so it drops into the
// same VirtualProvider. Thread-safe across guest threads.
class ArchiveFilesystem {
public:
  // A regular file inside the container, identified by its index in the entry
  // table (the backend keeps the offsets needed to decode it).
  struct Node {
    u64 size = 0;
    u32 index = 0;
  };

  explicit ArchiveFilesystem(const base::String &archivePath);
  ~ArchiveFilesystem();

  ArchiveFilesystem(const ArchiveFilesystem &) = delete;
  ArchiveFilesystem &operator=(const ArchiveFilesystem &) = delete;

  bool valid() const;

  // Look up a file by archive-relative path with a leading '/', e.g.
  // "/eboot.bin". Returns nullptr if absent or a directory.
  const Node *find(const char *relPath) const;

  // Read up to len bytes of a file starting at byte offset off. Returns the
  // number of bytes read (clamped to the file size, 0 past the end), or -1 on
  // error.
  i64 read(const Node &node, void *buf, i64 off, i64 len);

  // One immediate child of a directory.
  struct Child {
    std::string name;
    bool isDir;
  };

  // List a directory's immediate children. relPath is matched case-insensitively
  // like find(). Returns false if the directory holds nothing. The child index is
  // built on first use, since a title that never enumerates should not pay for
  // it (this container has 223k entries).
  bool list(const char *relPath, std::vector<Child> &out);

  // Collect every regular-file path in the container (leading '/').
  void paths(std::vector<std::string> &out) const;

  // Which backend claimed the file ("rar", "zip"), or "" if none did.
  const char *backendName() const;

private:
  std::unique_ptr<ArchiveImpl> impl_;
};

// True if the path's extension is a container ArchiveFilesystem can open. Only
// checks the name, not the contents.
bool isArchivePath(const char *path);

} // namespace vfs
