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
#include <thread>

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
