
/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "file.h"
#include <algorithm>
#include <cstdio>

namespace utl {
namespace {
// file on disk impl
class PhysFile final : public fileBase {
  size_t sizeTracker;
  std::FILE *fptr;

public:
  PhysFile(const base::String &name, fileMode mode) : fptr(nullptr) {
    // convert access mode. The writable modes open "+" (read+write) so a
    // caller can read a file back after writing it (savedata round-trips a save
    // within one mount); `write` stays create-or-truncate but is now also
    // readable (a superset -- the write-once exporters that use it are
    // unaffected). `readWrite` opens an existing file without truncating.
    const char *modeStr = "a+";
    if (mode == fileMode::read)
      modeStr = "rb";
    else if (mode == fileMode::write || mode == fileMode::create ||
             mode == fileMode::trunc)
      modeStr = "wb+";
    else if (mode == fileMode::readWrite)
      modeStr = "rb+";
    else if (mode == fileMode::append)
      modeStr = "ab+";

    fopen_s(&fptr, name.c_str(), modeStr);

    // Cache the initial size for any mode that opens an existing file (read or
    // read-write); the write/create/trunc modes start empty.
    if (fptr && (mode == fileMode::read || mode == fileMode::readWrite ||
                 mode == fileMode::append)) {
      std::fseek(fptr, 0, SEEK_END);
      sizeTracker = static_cast<size_t>(std::ftell(fptr));
    } else {
      sizeTracker = 0;
    }

    if (fptr) {
      // ensure that we can always start from offset 0
      std::fseek(fptr, 0, SEEK_SET);
    }
  }

  ~PhysFile() { Close(); }

  void Close() override {
    if (fptr) {
      std::fclose(fptr);
      sizeTracker = 0;
    }
  }

  uint64_t Read(void *buf, size_t size) override {
    // Count bytes, not one object of `size` bytes: fread(buf, size, 1, f)
    // returns 0 for any partial object, so a guest read() that asks for more
    // than the file holds (Skyrim reads its 2 KiB INI in 16 KiB chunks) came
    // back as end-of-file and the title loaded nothing.
    return std::fread(buf, 1, size, fptr);
  }

  uint64_t Write(const void *buf, size_t size) override {
    sizeTracker += size;
    return std::fwrite(buf, size, 1, fptr);
  }

  uint64_t Seek(int64_t ofs, seekMode mode) override {
    // translate mode
    int32_t origin = 0;
    switch (mode) {
    case seekMode::seek_cur:
      origin = SEEK_CUR;
      break;
    case seekMode::seek_end:
      origin = SEEK_END;
      break;
    case seekMode::seek_set:
      origin = SEEK_SET;
      break;
    default:
      return 0;
    }

    // 64-bit seek: the cast-to-uint32 truncation here corrupted reads past 4 GiB
    // (e.g. files deep in a multi-GiB pkg's inner PFS image -> wrong file offset).
#ifdef _WIN32
    int64_t x = _fseeki64(fptr, ofs, origin);
    if (x == 0)
      return static_cast<uint64_t>(_ftelli64(fptr));
#else
    int64_t x = fseeko(fptr, static_cast<off_t>(ofs), origin);
    if (x == 0)
      return static_cast<uint64_t>(ftello(fptr));
#endif
    return static_cast<uint64_t>(x);
  }

  uint64_t Tell() override { return std::ftell(fptr); }

  uint64_t GetSize() override {
    /*if (sizeTracker == 0) {
            std::FILE* fptr = static_cast<std::FILE*>(handle);

            std::fseek(fptr, 0, SEEK_END);
            sizeTracker = static_cast<size_t>(std::ftell(fptr));
            std::fseek(fptr, 0, SEEK_SET);
    }*/

    return sizeTracker;
  }

  native_handle GetNativeHandle() override {
    return static_cast<native_handle>(fptr);
  }

  bool IsOpen() override { return fptr != nullptr; }
};

// from memory
class MemStream final : public fileBase {
  uint64_t pos;
  const char *const ptr;
  const uint64_t size;

public:
  MemStream(const void *ptr, uint64_t size)
      : ptr(static_cast<const char *>(ptr)), size(size) {}

  uint64_t Read(void *buf, size_t count) override {
    if (pos < size) {
      // get readable size
      if (const uint64_t result = std::min<uint64_t>(count, size - pos)) {
        std::memcpy(buf, ptr + pos, result);
        pos += result;
        return result;
      }
    }

    return 0;
  }

  // TODO
  uint64_t Write(const void *, size_t) override { return 0; }

  uint64_t Seek(int64_t ofs, seekMode mode) override {
    switch (mode) {
    case seekMode::seek_cur:
      pos += ofs;
      break;
    case seekMode::seek_end:
      pos = size;
      break;
    case seekMode::seek_set:
      pos = ofs;
      break;
    default:
      return 0;
    }

    return 0;
  }

  uint64_t Tell() override { return pos; }

  uint64_t GetSize() override { return size; }

  native_handle GetNativeHandle() override { return nullptr; }
};
}

File::File(const base::String &dir, fileMode mode /* = fileMode::read */)
    : file(base::MakeUnique<PhysFile>(dir, mode)) {}

File::File(const void *ptr, size_t size)
    : file(base::MakeUnique<MemStream>(ptr, size)) {}

File::File(base::UniquePointer<fileBase> &&base) : file(std::move(base)) {}

File::~File() { Close(); }
}
