/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <utl/path.h>

#ifdef _WIN32
#include <Windows.h>
#include <cwchar>
#include <cstring>
#else
#include <climits>
#include <cstring>
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

namespace utl {

#ifdef _WIN32
base::StringW make_abs_path(const base::StringW &rel) {
  static base::StringW filePath;
  if (filePath.empty()) {
    wchar_t buf[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    wchar_t *dirPtr = std::wcsrchr(buf, L'\\');
    if (dirPtr) dirPtr[1] = L'\0';
    filePath = base::StringW(buf);
  }

  base::StringW out = filePath;
  out += rel;

  // backslashes everywhere
  for (auto &c : out) if (c == L'/') c = L'\\';
  return out;
}

base::String make_abs_path(const base::String &rel) {
  static base::String filePath;
  if (filePath.empty()) {
    char buf[MAX_PATH]{};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    char *dirPtr = std::strrchr(buf, '\\');
    if (dirPtr) dirPtr[1] = '\0';
    filePath = base::String(buf);
  }

  base::String out = filePath;
  out += rel;

  for (auto &c : out) if (c == '/') c = '\\';
  return out;
}

#else  // POSIX

static const base::String& exe_dir() {
  static base::String filePath;
  if (filePath.empty()) {
    char buf[PATH_MAX]{};
    ssize_t n = ::readlink("/proc/self/exe", buf, PATH_MAX - 1);
    if (n > 0) {
      buf[n] = '\0';
      char* slash = std::strrchr(buf, '/');
      if (slash)
        slash[1] = '\0';
      filePath = base::String(buf);
    } else {
      filePath = base::String("./");
    }
  }
  return filePath;
}

base::String make_abs_path(const base::String &rel) {
  base::String out = exe_dir();
  out += rel;
  return out;
}

base::StringW make_abs_path(const base::StringW &rel) {
  const auto& dir = exe_dir();
  base::StringW out;
  out.reserve(static_cast<base::StringW::size_type>(dir.size() + rel.size()));
  for (const char* p = dir.c_str(); *p; ++p)
    out.push_back(static_cast<wchar_t>(*p));
  out += rel;
  return out;
}

#endif

}  // namespace utl
