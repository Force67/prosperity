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

// Like mount(), but the mount is writable: guest opens with a write/create flag
// create/modify files on the host underneath it (used for savedata). The host
// directory is created if absent. Read-only titles never open host mounts for
// write, so this cannot affect them.
void mountWritable(const char *guestPrefix, const char *hostDir);

// Resolve a guest path to a host path, or empty if no host mount matches.
// (Virtual mounts have no host path; use openRead/stat for those.)
base::String resolve(const char *guestPath);

// Resolve a guest path to its host path IFF it lies under a WRITABLE host mount
// (else empty). sys_open/sys_mkdir use this to service create/write requests.
base::String resolveWritable(const char *guestPath);

// Create a directory (and parents) on the host for a guest path under a
// writable mount. Returns false if the path is not under a writable mount.
bool makeDir(const char *guestPath);

// Remove a file on the host for a guest path under a writable mount.
bool removeFile(const char *guestPath);

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

// The booted title's TITLE_ID (e.g. "CUSA00792"), or empty if unknown. Set once
// at boot by dcore from the pkg's param.sfo (the outer PKG metadata entry, which
// is the only copy for titles like Isaac). savedata reads it to give each title
// its own host save root.
void setTitleId(const std::string &id);
const std::string &titleId();

// Small content cache keyed by a short name (SOTTR workaround: the engine's
// async manifest reader races, so we cache the real manifest contents at mount
// and let the count-setter fill the header buffer with correct data).
void cacheFile(const std::string &key, std::vector<uint8_t> data);
const std::vector<uint8_t> *getCachedFile(const char *key);
} // namespace krnl::vfs
