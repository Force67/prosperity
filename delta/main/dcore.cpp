/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "dcore.h"
#include <logger/logger.h>
#include <utl/file.h>

#include <kern/vfs.h>

#include "formats/pkg_object.h"
#include "formats/pup_object.h"

deltaCore::deltaCore() = default;
deltaCore::~deltaCore() = default;

bool deltaCore::init() {
  LOG_INFO("Initializing deltaCore " rsc_copyright);
  return true;
}

namespace {
// Bridges a PkgFilesystem into the kernel VFS as an on-demand virtual mount.
class PkgProvider : public krnl::vfs::VirtualProvider {
public:
  explicit PkgProvider(const base::String &path) : fs_(path) {}
  bool valid() const { return fs_.valid(); }

  std::unique_ptr<krnl::vfs::VirtualFile> open(const char *rel) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return nullptr;
    return std::make_unique<PkgFile>(&fs_, *node);
  }
  bool stat(const char *rel, int64_t &size) override {
    const auto *node = fs_.find(rel);
    if (!node)
      return false;
    size = static_cast<int64_t>(node->size);
    return true;
  }
  bool list(const char *rel, std::vector<krnl::vfs::DirEntry> &out) override {
    // Build "prefix/" so we match only paths inside this directory. Root ("" or
    // "/") -> "/". The pkg stores absolute paths with a leading '/'.
    std::string prefix(rel ? rel : "");
    while (!prefix.empty() && prefix.back() == '/')
      prefix.pop_back();
    prefix += "/";
    if (prefix.empty() || prefix[0] != '/')
      prefix.insert(prefix.begin(), '/');

    std::vector<std::string> all;
    fs_.paths(all);
    std::set<std::string> seen;
    for (const auto &p : all) {
      if (p.size() <= prefix.size() || p.compare(0, prefix.size(), prefix) != 0)
        continue;
      std::string rest = p.substr(prefix.size());
      auto slash = rest.find('/');
      bool isDir = slash != std::string::npos;
      std::string child = isDir ? rest.substr(0, slash) : rest;
      if (!child.empty() && seen.insert(child).second)
        out.push_back({child, isDir});
    }
    return !out.empty();
  }

private:
  struct PkgFile : krnl::vfs::VirtualFile {
    vfs::PkgFilesystem *fs;
    vfs::PkgFilesystem::Node node;
    PkgFile(vfs::PkgFilesystem *f, const vfs::PkgFilesystem::Node &n)
        : fs(f), node(n) {}
    int64_t read(void *buf, int64_t off, int64_t len) override {
      return fs->read(node, buf, off, len);
    }
    int64_t size() override { return static_cast<int64_t>(node.size); }
  };

  vfs::PkgFilesystem fs_;
};

bool endsWithIgnoreCase(const base::String &s, const char *ext) {
  size_t n = s.length(), e = std::strlen(ext);
  if (n < e)
    return false;
  const char *p = s.c_str() + (n - e);
  for (size_t i = 0; i < e; ++i) {
    char a = p[i];
    if (a >= 'A' && a <= 'Z')
      a += 'a' - 'A';
    if (a != ext[i])
      return false;
  }
  return true;
}
} // namespace

void deltaCore::boot(const base::String &xdir) {
  base::String path = xdir;

#ifdef _WIN32
  for (auto &c : path)
    if (c == '/')
      c = '\\';
#endif

  const bool isPkg = endsWithIgnoreCase(xdir, ".pkg");
  base::String mainModule = path;

  if (isPkg) {
    auto provider = std::make_shared<PkgProvider>(path);
    if (!provider->valid()) {
      LOG_ERROR("failed to load pkg {}", path.c_str());
      return;
    }
    krnl::vfs::mountVirtual("/app0", provider);
    mainModule = base::String("/app0/eboot.bin");
  }

  std::thread ctx([mainModule = std::move(mainModule), isPkg]() {
    auto p = base::MakeUnique<krnl::proc>();
    if (!p->create(mainModule, isPkg))
      return;

    p->start();
  });

  ctx.detach();
}
