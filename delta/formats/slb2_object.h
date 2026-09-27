#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <io/file.h>
#include "base/arch.h"

namespace formats {
struct slb2_header {
  u32 magic;
  u32 version;
  u32 flags;
  u32 fileCount;
  u32 blockCount;
  u32 unk[3];
};

struct slb2_entry {
  u32 offset;
  u32 fileSize;
  u32 unk[2];
  char fileName[32];
};

class slb2Object {
public:
  bool load(io::File &);

private:
  slb2_header header{};
};
}