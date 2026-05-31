#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base/strings/xstring.h>

namespace krnl::vfs {
// Map a guest path prefix (e.g. "/app0") onto a host directory. Longest prefix
// wins at resolve time.
void mount(const char *guestPrefix, const char *hostDir);

// Resolve a guest path to a host path, or empty if nothing is mounted for it.
base::String resolve(const char *guestPath);
} // namespace krnl::vfs
