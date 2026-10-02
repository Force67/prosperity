#include "main/game_directory.h"

#include <sys/stat.h>

namespace cli {
namespace {
bool HasType(const base::String& path, int type) {
  struct stat info{};
  return ::stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFMT) == type;
}

base::String ParentPath(const base::String& path) {
  const auto slash = path.find_last_of("/\\", base::String::npos, 2);
  return slash == base::String::npos ? base::String(".")
                                     : path.substr(0, slash == 0 ? 1 : slash);
}

base::String Filename(const base::String& path) {
  const auto slash = path.find_last_of("/\\", base::String::npos, 2);
  return slash == base::String::npos ? path : path.substr(slash + 1);
}
}  // namespace

GameDirectory FindGameDirectory(const base::String& path) {
  base::String root = path;
  while (root.size() > 1 && (root.back() == '/' || root.back() == '\\'))
    root.pop_back();
  if (HasType(root, S_IFREG)) {
    if (Filename(root) != "eboot.bin")
      return {};
    root = ParentPath(root);
  } else if (!HasType(root, S_IFDIR)) {
    return {};
  }
  if (Filename(root) == "decrypted" &&
      HasType(ParentPath(root) + "/sce_sys", S_IFDIR))
    root = ParentPath(root);
  for (const char* module : {"/decrypted/eboot.bin", "/eboot.bin"}) {
    if (HasType(root + module, S_IFREG))
      return {root, base::String("/app0") + module};
  }
  return {};
}
}  // namespace cli
