#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

#include <base/containers/vector.h>
#include <base/memory/unique_pointer.h>
#include <base/strings/xstring.h>

namespace utl {
// Use void* uniformly: on Windows we stash the HANDLE, on POSIX we stash
// either a FILE* or nullptr. Callers that actually need an int fd can
// reach down to the FILE* themselves.
using native_handle = void *;

enum class fileMode { read, write, append, create, trunc, readWrite };

enum class seekMode : uint32_t {
  seek_set,
  seek_cur,
  seek_end,
};

class fileBase {
public:
  virtual ~fileBase() = default;

  virtual void Close(){};
  virtual bool IsOpen() { return true; }
  virtual uint64_t Read(void *, size_t) = 0;
  virtual uint64_t Write(const void *, size_t) = 0;
  virtual void Flush() {}
  virtual uint64_t Seek(int64_t, seekMode) = 0;
  virtual uint64_t Tell() = 0;
  virtual uint64_t GetSize() = 0;
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

  inline void Reset(base::UniquePointer<fileBase> &&ptr) { file = std::move(ptr); }

  inline base::UniquePointer<fileBase> GetBase() { return std::move(file); }

  inline uint64_t Read(void *ptr, size_t size) { return file->Read(ptr, size); }
  inline uint64_t Write(const void *ptr, size_t size) {
    return file->Write(ptr, size);
  }
  inline void Flush() {
    if (file)
      file->Flush();
  }
  inline uint64_t Seek(uint64_t ofs, seekMode mods) {
    return file->Seek(ofs, mods);
  }
  inline uint64_t GetSize() { return file->GetSize(); }
  inline uint64_t Tell() { return file->Tell(); }
  inline native_handle GetNativeHandle() { return file->GetNativeHandle(); }
  inline bool IsOpen() { return file->IsOpen(); }
  inline bool Exists() { return static_cast<bool>(file); }

  // POD to base::Vector
  template <typename T>
  std::enable_if_t<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>, bool>
  Read(base::Vector<T> &vec, std::size_t size) {
    vec.resize(size);
    return this->Read(vec.data(), sizeof(T) * size) == sizeof(T) * size;
  }

  // Read POD vector, size set via resize() externally.
  template <typename T>
  std::enable_if_t<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>, bool>
  Read(base::Vector<T> &vec) {
    return this->Read(vec.data(), sizeof(T) * vec.size()) ==
           sizeof(T) * vec.size();
  }

  template <typename T>
  std::enable_if_t<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>, bool>
  Read(T &data) {
    return Read(&data, sizeof(T)) == sizeof(T);
  }

  template <typename T>
  std::enable_if_t<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>,
                   const File &>
  Write(const T &data) {
    Write(std::addressof(data), sizeof(T));
    return *this;
  }

  template <typename T>
  std::enable_if_t<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>,
                   const File &>
  Write(const base::Vector<T> &vec) {
    Write(vec.data(), vec.size() * sizeof(T));
    return *this;
  }
};

using FileHandle = std::shared_ptr<File>;

template <typename T> struct ContainerStream final : fileBase {
  using value_type = typename std::remove_reference_t<T>::value_type;

  T obj;
  uint64_t pos;

  ContainerStream(T &&obj) : obj(std::forward<T>(obj)), pos(0) {}

  ~ContainerStream() override {}

  uint64_t Read(void *buffer, uint64_t size) override {
    const uint64_t end = obj.size();

    if (pos < end) {
      if (const uint64_t max = std::min<uint64_t>(size, end - pos)) {
        std::copy(obj.begin() + pos, obj.begin() + pos + max,
                  static_cast<value_type *>(buffer));
        pos = pos + max;
        return max;
      }
    }

    return 0;
  }

  uint64_t Write(const void *buffer, uint64_t size) override {
    const uint64_t old_size = obj.size();
    (void)old_size;

    if (pos > obj.size()) {
      obj.resize(pos);
    }

    const auto src = static_cast<const value_type *>(buffer);

    const uint64_t overlap = std::min<uint64_t>(obj.size() - pos, size);
    std::copy(src, src + overlap, obj.begin() + pos);

    obj.insert(obj.end(), src + overlap, src + size);
    pos += size;

    return size;
  }

  uint64_t Seek(int64_t offset, seekMode whence) override {
    const int64_t new_pos =
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

  uint64_t GetSize() override { return obj.size(); }

  native_handle GetNativeHandle() override { return nullptr; }

  uint64_t Tell() override { return pos; }
};

template <typename T> File make_stream(T &&container = T{}) {
  File result(base::MakeUnique<ContainerStream<T>>(std::forward<T>(container)));
  return result;
}
}
