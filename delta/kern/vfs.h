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
#include <utl/file.h>

namespace krnl::vfs {
// One entry in a directory listing.
struct DirEntry {
  std::string name;
  bool isDir;
};
// Map a guest path prefix (e.g. "/app0") onto a host directory. Longest prefix
// wins at resolve time.
void mount(const char *guestPrefix, const char *hostDir);

// Resolve a guest path to a host path, or empty if no host mount matches.
// (Virtual mounts have no host path; use openRead/stat for those.)
base::String resolve(const char *guestPath);

// A lazily-read file backing a virtual mount, e.g. a file inside a .pkg PFS.
struct VirtualFile {
  virtual ~VirtualFile() = default;
  // Read up to len bytes at byte offset off. Returns bytes read (0 past EOF) or
  // -1 on error.
  virtual int64_t read(void *buf, int64_t off, int64_t len) = 0;
  virtual int64_t size() = 0;
};

// Serves files for a virtual mount prefix. relPath is the remainder after the
// prefix and keeps its leading '/', e.g. "/eboot.bin".
struct VirtualProvider {
  virtual ~VirtualProvider() = default;
  virtual std::unique_ptr<VirtualFile> open(const char *relPath) = 0;
  virtual bool stat(const char *relPath, int64_t &size) = 0;
  // List the immediate children of a directory. Returns false if relPath is not
  // a directory (or listing is unsupported). Default: not a directory.
  virtual bool list(const char * /*relPath*/, std::vector<DirEntry> & /*out*/) {
    return false;
  }
};

// Map a guest path prefix onto an on-demand provider (kept alive for the
// process lifetime).
void mountVirtual(const char *guestPrefix,
                  std::shared_ptr<VirtualProvider> provider);

// Open a guest path for reading, resolving both host and virtual mounts.
// Returns an empty File (Exists() == false) if nothing matches / the file is
// absent; otherwise a File ready to read.
utl::File openRead(const char *guestPath);

// Stat a guest path across host and virtual mounts. Returns false if absent.
bool stat(const char *guestPath, int64_t &size, bool &isDir);

// List a directory's immediate children. Returns false if not a directory.
bool listDir(const char *guestPath, std::vector<DirEntry> &out);
} // namespace krnl::vfs
