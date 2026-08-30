/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Mounting a game container at /app0.
//
// Each supported container (a PS4 .pkg, a PS5 .ffpkg UFS2 image, a .rar/.zip
// holding either console's game directory) is adapted onto the guest VFS by a
// provider that reads it on demand -- nothing is ever extracted. The providers
// themselves are private to this unit; a caller says which container it has and
// gets back the mount plus the handful of title facts the boot path needs.
//
// This lives in kern rather than delta_formats because a provider implements
// krnl::vfs::VirtualProvider, and formats depends on nothing in delta -- worth
// keeping that way, since it is the module whose tests stand alone. It lives
// here rather than in main because choosing how a container becomes a guest
// filesystem is a decision, and the composition root holds wiring.

#include "base/arch.h"
#include <memory>
#include <string>
#include <vector>

#include <base/strings/xstring.h>

#include "kern/vfs.h"

namespace krnl::vfs {

struct TitleMount {
  // Null when the container could not be opened; nothing else is then set.
  std::shared_ptr<VirtualProvider> provider;
  std::string titleId;
  std::string title;
  std::vector<u8> icon;  // empty unless `wantIcon`
  u32 attributes = 0;    // PS4 param.sfo ATTRIBUTE
  u32 sdkVersion = 0;    // PS5 param.json sdkVersion, top half
  bool isPs5 = false;
  // Whether the container carries a decrypted/ tree of plaintext ELFs. When it
  // does not, the top-level eboot.bin is a still-encrypted SELF.
  bool hasDecrypted = false;

  explicit operator bool() const { return provider != nullptr; }
};

// `wantIcon` reads sce_sys/icon0.png into the result; it is up to 16 MiB and
// only the desktop window title bar uses it.
TitleMount mountPkg(const base::String &path, bool wantIcon);
TitleMount mountFfpkg(const base::String &path, bool wantIcon);
TitleMount mountArchive(const base::String &path, bool wantIcon);

}  // namespace krnl::vfs
