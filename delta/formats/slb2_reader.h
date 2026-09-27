#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "io/file.h"

namespace formats {
struct Slb2Header {
  u32 magic;
  u32 version;
  u32 flags;
  u32 file_count;
  u32 block_count;
  u32 unk[3];
};

struct Slb2Entry {
  u32 offset;
  u32 file_size;
  u32 unk[2];
  char file_name[32];
};

class Slb2Object {
 public:
  bool Load(io::File&);

 private:
  Slb2Header header_{};
};
}  // namespace formats