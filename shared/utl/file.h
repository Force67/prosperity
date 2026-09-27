#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"

#include <base/containers/vector.h>
#include <base/memory/unique_pointer.h>
#include <base/strings/xstring.h>
#include <base/algorithm.h>
#include <base/math/value_bounds.h>
#include <base/memory/move.h>
#include <base/memory/shared_pointer.h>
#include <base/meta/traits.h>

namespace utl {
// Use void* uniformly: on Windows we stash the HANDLE, on POSIX we stash
// either a FILE* or nullptr. Callers that actually need an int fd can
// reach down to the FILE* themselves.
using native_handle = void *;

enum class fileMode { read, write, append, create, trunc, readWrite };

enum class seekMode : u32 {
  seek_set,
  seek_cur,
  seek_end,
};

class fileBase {
public:
  virtual ~fileBase() = default;

  virtual void Close(){};
  virtual bool IsOpen() { return true; }
  virtual u64 Read(void *, size_t) = 0;
  virtual u64 Write(const void *, size_t) = 0;
  virtual void Flush() {}
  virtual u64 Seek(i64, seekMode) = 0;
  virtual u64 Tell() = 0;
  virtual u64 GetSize() = 0;
  virtual native_handle GetNativeHandle() = 0;
};

class File {
  base::UniquePointer<fileBase> file{};

public:
  File() = default;
  File(const base::String &, fileMode mode = fileMode::read);
  File(const void *, size_t);
  File(base::UniquePointer<fileBase> &&);
  ~File();

  // move
  File(File &rhs) : file(rhs.GetBase()) {}

  void Close() {
    if (file)
      file = {};
  }

  inline void Reset(base::UniquePointer<fileBase> &&ptr) { file = base::move(ptr); }

  inline base::UniquePointer<fileBase> GetBase() { return base::move(file); }

  inline u64 Read(void *ptr, size_t size) { return file->Read(ptr, size); }
  inline u64 Write(const void *ptr, size_t size) {
    return file->Write(ptr, size);
  }
  inline void Flush() {
    if (file)
      file->Flush();
  }
  inline u64 Seek(u64 ofs, seekMode mods) {
    return file->Seek(ofs, mods);
  }
  inline u64 GetSize() { return file->GetSize(); }
  inline u64 Tell() { return file->Tell(); }
  inline native_handle GetNativeHandle() { return file->GetNativeHandle(); }
  inline bool IsOpen() { return file->IsOpen(); }
  inline bool Exists() { return static_cast<bool>(file); }

  // POD to base::Vector
  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>, bool>
  Read(base::Vector<T> &vec, mem_size size) {
    vec.resize(size);
    return this->Read(vec.data(), sizeof(T) * size) == sizeof(T) * size;
  }

  // Read POD vector, size set via resize() externally.
  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>, bool>
  Read(base::Vector<T> &vec) {
    return this->Read(vec.data(), sizeof(T) * vec.size()) ==
           sizeof(T) * vec.size();
  }

  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>, bool>
  Read(T &data) {
    return Read(&data, sizeof(T)) == sizeof(T);
  }

  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                   const File &>
  Write(const T &data) {
    Write(base::AddressOf(data), sizeof(T));
    return *this;
  }

  template <typename T>
  base::enable_if_t<base::is_trivially_copyable_v<T> && !base::is_pointer_v<T>,
                   const File &>
  Write(const base::Vector<T> &vec) {
    Write(vec.data(), vec.size() * sizeof(T));
    return *this;
  }
};

using FileHandle = base::SharedPointer<File>;

template <typename T> struct ContainerStream final : fileBase {
  using value_type = typename base::remove_reference_t<T>::value_type;

  T obj;
  u64 pos;

  ContainerStream(T &&obj) : obj(base::forward<T>(obj)), pos(0) {}

  ~ContainerStream() override {}

  u64 Read(void *buffer, u64 size) override {
    const u64 end = obj.size();

    if (pos < end) {
      if (const u64 max = base::Min<u64>(size, end - pos)) {
        base::Copy(obj.begin() + pos, obj.begin() + pos + max,
                  static_cast<value_type *>(buffer));
        pos = pos + max;
        return max;
      }
    }

    return 0;
  }

  u64 Write(const void *buffer, u64 size) override {
    const u64 old_size = obj.size();
    (void)old_size;

    if (pos > obj.size()) {
      obj.resize(pos);
    }

    const auto src = static_cast<const value_type *>(buffer);

    const u64 overlap = base::Min<u64>(obj.size() - pos, size);
    base::Copy(src, src + overlap, obj.begin() + pos);

    obj.insert(obj.end(), src + overlap, src + size);
    pos += size;

    return size;
  }

  u64 Seek(i64 offset, seekMode whence) override {
    const i64 new_pos =
        whence == seekMode::seek_set
            ? offset
            : whence == seekMode::seek_cur
                  ? offset + pos
                  : whence == seekMode::seek_end ? offset + GetSize() : (0);

    if (new_pos < 0) {
      return -1;
    }

    pos = new_pos;
    return pos;
  }

  u64 GetSize() override { return obj.size(); }

  native_handle GetNativeHandle() override { return nullptr; }

  u64 Tell() override { return pos; }
};

template <typename T> File make_stream(T &&container = T{}) {
  File result(base::MakeUnique<ContainerStream<T>>(base::forward<T>(container)));
  return result;
}
}
