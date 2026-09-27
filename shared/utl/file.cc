
/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "utl/file.h"
#include <cstdio>
#include "base.h"
#include "base/algorithm.h"
#include "base/arch.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"

namespace utl {
namespace {
// file on disk impl
class PhysFile final : public FileBase {
  size_t size_tracker_;
  std::FILE* fptr_;

 public:
  PhysFile(const base::String& name, FileMode mode) : fptr_(nullptr) {
    // convert access mode. The writable modes open "+" (read+write) so a
    // caller can read a file back after writing it (savedata round-trips a save
    // within one mount); `write` stays create-or-truncate but is now also
    // readable (a superset; the write-once exporters that use it are
    // unaffected). `readWrite` opens an existing file without truncating.
    const char* mode_str = "a+";
    if (mode == FileMode::kRead)
      mode_str = "rb";
    else if (mode == FileMode::kWrite || mode == FileMode::kCreate ||
             mode == FileMode::kTrunc)
      mode_str = "wb+";
    else if (mode == FileMode::kReadWrite)
      mode_str = "rb+";
    else if (mode == FileMode::kAppend)
      mode_str = "ab+";

    fopen_s(&fptr_, name.c_str(), mode_str);

    // Cache the initial size for any mode that opens an existing file (read or
    // read-write); the write/create/trunc modes start empty.
    if (fptr_ && (mode == FileMode::kRead || mode == FileMode::kReadWrite ||
                  mode == FileMode::kAppend)) {
      std::fseek(fptr_, 0, SEEK_END);
      size_tracker_ = static_cast<size_t>(std::ftell(fptr_));
    } else {
      size_tracker_ = 0;
    }

    if (fptr_) {
      // ensure that we can always start from offset 0
      std::fseek(fptr_, 0, SEEK_SET);
    }
  }

  ~PhysFile() { Close(); }

  void Close() override {
    if (fptr_) {
      std::fclose(fptr_);
      size_tracker_ = 0;
    }
  }

  u64 Read(void* buf, size_t size) override {
    // Count bytes, not one object of `size` bytes: fread(buf, size, 1, f)
    // returns 0 for any partial object, so a guest read() that asks for more
    // than the file holds (Skyrim reads its 2 KiB INI in 16 KiB chunks) came
    // back as end-of-file and the title loaded nothing.
    return std::fread(buf, 1, size, fptr_);
  }

  u64 Write(const void* buf, size_t size) override {
    size_tracker_ += size;
    return std::fwrite(buf, size, 1, fptr_);
  }

  void Flush() override {
    if (fptr_)
      std::fflush(fptr_);
  }

  u64 Seek(i64 ofs, SeekMode mode) override {
    // translate mode
    i32 origin = 0;
    switch (mode) {
      case SeekMode::kSeekCur:
        origin = SEEK_CUR;
        break;
      case SeekMode::kSeekEnd:
        origin = SEEK_END;
        break;
      case SeekMode::kSeekSet:
        origin = SEEK_SET;
        break;
      default:
        return 0;
    }

      // 64-bit seek: the cast-to-uint32 truncation here corrupted reads past 4
      // GiB (e.g. files deep in a multi-GiB pkg's inner PFS image -> wrong file
      // offset).
#ifdef _WIN32
    i64 x = _fseeki64(fptr, ofs, origin);
    if (x == 0)
      return static_cast<u64>(_ftelli64(fptr));
#else
    i64 x = fseeko(fptr_, static_cast<off_t>(ofs), origin);
    if (x == 0)
      return static_cast<u64>(ftello(fptr_));
#endif
    return static_cast<u64>(x);
  }

  u64 Tell() override { return std::ftell(fptr_); }

  u64 GetSize() override {
    /*if (sizeTracker == 0) {
            std::FILE* fptr = static_cast<std::FILE*>(handle);

            std::fseek(fptr, 0, SEEK_END);
            sizeTracker = static_cast<size_t>(std::ftell(fptr));
            std::fseek(fptr, 0, SEEK_SET);
    }*/

    return size_tracker_;
  }

  NativeHandle GetNativeHandle() override {
    return static_cast<NativeHandle>(fptr_);
  }

  bool IsOpen() override { return fptr_ != nullptr; }
};

// from memory
class MemStream final : public FileBase {
  u64 pos_;
  const char* const ptr_;
  const u64 size_;

 public:
  MemStream(const void* ptr, u64 size)
      : ptr_(static_cast<const char*>(ptr)), size_(size) {}

  u64 Read(void* buf, size_t count) override {
    if (pos_ < size_) {
      // get readable size
      if (const u64 result = base::Min<u64>(count, size_ - pos_)) {
        std::memcpy(buf, ptr_ + pos_, result);
        pos_ += result;
        return result;
      }
    }

    return 0;
  }

  u64 Write(const void*, size_t) override { return 0; }

  u64 Seek(i64 ofs, SeekMode mode) override {
    switch (mode) {
      case SeekMode::kSeekCur:
        pos_ += ofs;
        break;
      case SeekMode::kSeekEnd:
        pos_ = size_;
        break;
      case SeekMode::kSeekSet:
        pos_ = ofs;
        break;
      default:
        return 0;
    }

    return 0;
  }

  u64 Tell() override { return pos_; }

  u64 GetSize() override { return size_; }

  NativeHandle GetNativeHandle() override { return nullptr; }
};
}  // namespace

File::File(const base::String& dir, FileMode mode /* = FileMode::kRead */)
    : file_(base::MakeUnique<PhysFile>(dir, mode)) {}

File::File(const void* ptr, size_t size)
    : file_(base::MakeUnique<MemStream>(ptr, size)) {}

File::File(base::UniquePointer<FileBase>&& base) : file_(base::move(base)) {}

File::~File() {
  Close();
}
}  // namespace utl
