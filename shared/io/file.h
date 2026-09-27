#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"

#include "base/algorithm.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/memory/unique_pointer.h"
#include "base/meta/traits.h"
#include "base/strings/xstring.h"

namespace io {
// Use void* uniformly: on Windows we stash the HANDLE, on POSIX we stash
// either a FILE* or nullptr. Callers that actually need an int fd can
// reach down to the FILE* themselves.
using NativeHandle = void*;

enum class FileMode { kRead, kWrite, kAppend, kCreate, kTrunc, kReadWrite };

enum class SeekMode : u32 {
  kSeekSet,
  kSeekCur,
  kSeekEnd,
};

class FileBase {
 public:
  virtual ~FileBase() = default;

  virtual void Close() {};
  virtual bool IsOpen() { return true; }
  virtual u64 Read(void*, size_t) = 0;
  virtual u64 Write(const void*, size_t) = 0;
  virtual void Flush() {}
  virtual u64 Seek(i64, SeekMode) = 0;
  virtual u64 Tell() = 0;
  virtual u64 GetSize() = 0;
  virtual NativeHandle GetNativeHandle() = 0;
};

class File {
  base::UniquePointer<FileBase> file_{};

 public:
  File() = default;
  File(const base::String&, FileMode mode = FileMode::kRead);
  File(const void*, size_t);
  File(base::UniquePointer<FileBase>&&);
  ~File();

  // move
  File(File& rhs) : file_(rhs.GetBase()) {}

  void Close() {
    if (file_)
      file_ = {};
  }

  inline void Reset(base::UniquePointer<FileBase>&& ptr) {
    file_ = base::move(ptr);
  }

  inline base::UniquePointer<FileBase> GetBase() { return base::move(file_); }

  inline u64 Read(void* ptr, size_t size) { return file_->Read(ptr, size); }
  inline u64 Write(const void* ptr, size_t size) {
    return file_->Write(ptr, size);
  }
  inline void Flush() {
    if (file_)
      file_->Flush();
  }
  inline u64 Seek(u64 ofs, SeekMode mods) { return file_->Seek(ofs, mods); }
  inline u64 GetSize() { return file_->GetSize(); }
  inline u64 Tell() { return file_->Tell(); }
  inline NativeHandle GetNativeHandle() { return file_->GetNativeHandle(); }
  inline bool IsOpen() { return file_->IsOpen(); }
  inline bool Exists() { return static_cast<bool>(file_); }

  // POD to base::Vector
  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                    bool>
  Read(base::Vector<T>& vec, mem_size size) {
    vec.resize(size);
    return this->Read(vec.data(), sizeof(T) * size) == sizeof(T) * size;
  }

  // Read POD vector, size set via resize() externally.
  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                    bool>
  Read(base::Vector<T>& vec) {
    return this->Read(vec.data(), sizeof(T) * vec.size()) ==
           sizeof(T) * vec.size();
  }

  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                    bool>
  Read(T& data) {
    return Read(&data, sizeof(T)) == sizeof(T);
  }

  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                    const File&>
  Write(const T& data) {
    Write(base::AddressOf(data), sizeof(T));
    return *this;
  }

  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                    const File&>
  Write(const base::Vector<T>& vec) {
    Write(vec.data(), vec.size() * sizeof(T));
    return *this;
  }
};

using FileHandle = base::SharedPointer<File>;

template <typename T>
struct ContainerStream final : FileBase {
  using ValueType = typename base::remove_reference_t<T>::ValueType;

  T obj;
  u64 pos;

  ContainerStream(T&& obj) : obj(base::forward<T>(obj)), pos(0) {}

  ~ContainerStream() override {}

  u64 Read(void* buffer, size_t size) override {
    const u64 end = obj.size();

    if (pos < end) {
      if (const u64 max = base::Min<u64>(size, end - pos)) {
        base::Copy(obj.begin() + pos, obj.begin() + pos + max,
                   static_cast<ValueType*>(buffer));
        pos = pos + max;
        return max;
      }
    }

    return 0;
  }

  u64 Write(const void* buffer, size_t size) override {
    const u64 old_size = obj.size();
    (void)old_size;

    if (pos > obj.size()) {
      obj.resize(pos);
    }

    const auto src = static_cast<const ValueType*>(buffer);

    const u64 overlap = base::Min<u64>(obj.size() - pos, size);
    base::Copy(src, src + overlap, obj.begin() + pos);

    obj.insert(obj.end(), src + overlap, src + size);
    pos += size;

    return size;
  }

  u64 Seek(i64 offset, SeekMode whence) override {
    const i64 new_pos = whence == SeekMode::kSeekSet   ? offset
                        : whence == SeekMode::kSeekCur ? offset + pos
                        : whence == SeekMode::kSeekEnd ? offset + GetSize()
                                                       : (0);

    if (new_pos < 0) {
      return -1;
    }

    pos = new_pos;
    return pos;
  }

  u64 GetSize() override { return obj.size(); }

  NativeHandle GetNativeHandle() override { return nullptr; }

  u64 Tell() override { return pos; }
};

template <typename T>
File MakeStream(T&& container = T{}) {
  File result(
      base::MakeUnique<ContainerStream<T>>(base::forward<T>(container)));
  return result;
}
}  // namespace io
