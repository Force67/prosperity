/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstring>

#include <base/containers/vector.h>

#include "vfs.h"

namespace krnl::vfs {
struct mountPoint {
  base::String guest;
  base::String host;
};

static base::Vector<mountPoint> g_mounts;

void mount(const char *guest, const char *host) {
  g_mounts.push_back({base::String(guest), base::String(host)});
}

base::String resolve(const char *path) {
  if (!path)
    return {};

  const mountPoint *best = nullptr;
  size_t bestLen = 0;
  for (auto &m : g_mounts) {
    size_t len = m.guest.length();
    if (std::strncmp(path, m.guest.c_str(), len) == 0 && len >= bestLen) {
      best = &m;
      bestLen = len;
    }
  }
  if (!best)
    return {};

  base::String out(best->host);
  const char *rest = path + bestLen; // remainder after the mount prefix
  if (*rest && *rest != '/')
    out += "/";
  out += rest;
  return out;
}
} // namespace krnl::vfs
