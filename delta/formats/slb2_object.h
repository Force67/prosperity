#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <utl/file.h>

namespace formats {
struct slb2_header {
  uint32_t magic;
  uint32_t version;
  uint32_t flags;
  uint32_t fileCount;
  uint32_t blockCount;
  uint32_t unk[3];
};

struct slb2_entry {
  uint32_t offset;
  uint32_t fileSize;
  uint32_t unk[2];
  char fileName[32];
};

class slb2Object {
public:
  bool load(utl::File &);

private:
  slb2_header header{};
};
}