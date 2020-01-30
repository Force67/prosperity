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
#include <string>
#include <vector>

namespace utl {
#if defined(_WIN32) || defined(__linux__)
using native_handle = void *;
#else
using native_handle = int;
#endif

enum class fileMode { read, write, append, create, trunc };

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
  virtual uint64_t Seek(int64_t, seekMode) = 0;
  virtual uint64_t Tell() = 0;
  virtual uint64_t GetSize() = 0;
  virtual native_handle GetNativeHandle() = 0;
};

class File {
  std::unique_ptr<fileBase> file{};

public:
  File() = default;
  File(const std::string &, fileMode mode = fileMode::read);
  File(const void *, size_t);
  File(std::unique_ptr<fileBase> &&);
  ~File();

  // move
  File(File &rhs) : file(rhs.GetBase()) {}

  void Close() {
    if (file)
      file.reset();
  }

  void Reset(std::unique_ptr<fileBase> &&ptr) { file = std::move(ptr); }

  inline std::unique_ptr<fileBase> GetBase() { return std::move(file); }

  inline uint64_t Read(void *ptr, size_t size) { return file->Read(ptr, size); }
  inline uint64_t Write(const void *ptr, size_t size) {
    return file->Write(ptr, size);
  }
  inline uint64_t Seek(uint64_t ofs, seekMode mods) {
    return file->Seek(ofs, mods);
  }
  inline uint64_t GetSize() { return file->GetSize(); }
  inline uint64_t Tell() { return file->Tell(); }
  inline native_handle GetNativeHandle() { return file->GetNativeHandle(); }
  inline bool IsOpen() { return file->IsOpen(); }
  inline bool Exists() { return file.get(); }

  // POD to std::vector
  template <typename T>
  std::enable_if_t<std::is_pod<T>::value && !std::is_pointer<T>::value, bool>
  Read(std::vector<T> &vec, std::size_t size) {
    vec.resize(size);
    return this->Read(vec.data(), sizeof(T) * size) == sizeof(T) * size;
  }

  // Read POD std::vector, size must be set by resize() method
  template <typename T>
  std::enable_if_t<std::is_pod<T>::value && !std::is_pointer<T>::value, bool>
  Read(std::vector<T> &vec) {
    return this->Read(vec.data(), sizeof(T) * vec.size()) ==
           sizeof(T) * vec.size();
  }

  // Read POD, sizeof(T) is used
  template <typename T>
  std::enable_if_t<std::is_pod<T>::value && !std::is_pointer<T>::value, bool>
  Read(T &data) {
    return Read(&data, sizeof(T)) == sizeof(T);
  }

  // Write POD unconditionally
  template <typename T>
  std::enable_if_t<std::is_pod<T>::value && !std::is_pointer<T>::value,
                   const File &>
  Write(const T &data) {
    Write(std::addressof(data), sizeof(T));
    return *this;
  }

  // Write POD std::vector unconditionally
  template <typename T>
  std::enable_if_t<std::is_pod<T>::value && !std::is_pointer<T>::value,
                   const File &>
  Write(const std::vector<T> &vec) {
    /*if (*/ Write(vec.data(),
                   vec.size() *
                       sizeof(T)); // != vec.size() * sizeof(T);//) //xfail();
    return *this;
  }
};

using FileHandle = std::shared_ptr<File>;

template <typename T> struct ContainerStream final : fileBase {
  // T can be a reference, but this is not recommended
  using value_type = typename std::remove_reference_t<T>::value_type;

  T obj;
  uint64_t pos;

  ContainerStream(T &&obj) : obj(std::forward<T>(obj)), pos(0) {}

  ~ContainerStream() override {}

  uint64_t Read(void *buffer, uint64_t size) override {
    const uint64_t end = obj.size();

    if (pos < end) {
      // Get readable size
      if (const uint64_t max = std::min<uint64_t>(size, end - pos)) {
        std::copy(obj.cbegin() + pos, obj.cbegin() + pos + max,
                  static_cast<value_type *>(buffer));
        pos = pos + max;
        return max;
      }
    }

    return 0;
  }

  uint64_t Write(const void *buffer, uint64_t size) override {
    const uint64_t old_size = obj.size();

    if (old_size + size < old_size) {
      // fmt::raw_error("fs::container_stream<>::write(): overflow");
    }

    if (pos > old_size) {
      // Fill gap if necessary (default-initialized)
      obj.resize(pos);
    }

    const auto src = static_cast<const value_type *>(buffer);

    // Overwrite existing part
    const uint64_t overlap = std::min<uint64_t>(obj.size() - pos, size);
    std::copy(src, src + overlap, obj.begin() + pos);

    // Append new data
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
      // fs::g_tls_error = fs::error::inval;
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
  File result(std::make_unique<ContainerStream<T>>(std::forward<T>(container)));
  return result;
}
}