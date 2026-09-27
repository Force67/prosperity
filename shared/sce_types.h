#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Sony Elf custom extensions

#include "base/arch.h"

// ELF and SCE spec names keep their canonical spelling.
// NOLINTBEGIN(readability-identifier-naming)

// increments by 0x10 for each new
// revision
enum class SELFProductType : u8 {
  PUP,
  K = 0xC,
  SL = 0xF,
  SM = 0xE,
  SELF = 0x8,  //< applies to EBOOT, ELF and SELF
  SPRX = 0x9   //< applies to SPRX, SDLL and SEXE
};

enum class SELFContentType : u8 { SELF = 1, PUP = 4 };

enum SegFlags {
  SF_ORDR = 0x1,    //< ordered?
  SF_ENCR = 0x2,    //< encrypted
  SF_SIGN = 0x4,    //< signed
  SF_DFLG = 0x8,    //< deflated
  SF_BFLG = 0x800,  //< block segment
};

static constexpr u32 SELF_MAGIC = 0x1D3D154F;
// PS5 titles wrap the same header layout in a different container magic.
static constexpr u32 SELF_MAGIC_PS5 = 0xEEF51454;

inline bool isSelfMagic(u32 magic) {
  return magic == SELF_MAGIC || magic == SELF_MAGIC_PS5;
}

struct SELFHeader {
  u32 magic;
  u8 version;
  u8 mode;
  u8 endian;
  u8 attr;

  // actually keyType
  SELFContentType contentType;
  SELFProductType productType;
  u16 pad;

  u16 headerSize;
  u16 metaSize;  // < sce Special
  u32 sizeSELF;  // < unrounded img size
  u32 fileSize;

  u16 numSegments;
  u16 flags;  //< always 0x22
  u32 pad2;   //<alignment
};

struct SELFSegmentTable {
  u64 flags;
  u64 offset;
  u64 fileSize;
  u64 memSize;

  u32 Id() { return static_cast<u32>((u64)flags >> 20); }
};

struct SCEContentId {
  char pad[0x20];
};

// is it really called "SCESPECIAL"?
struct SCESpecial {
  u64 authId;
  u64 productType;
  u64 version1;
  u64 version2;
  SCEContentId contentId;
  char shaSum[0x20];
};

// similar to MS PDB_CODEVIEW
struct SCEComment {
  u32 magic;  // "PATH"
  u32 unk;
  u32 pathLength;  // length of the following path
};

static_assert(sizeof(SELFHeader) == 32, "header size mismatch");
static_assert(sizeof(SELFSegmentTable) == 32, "segment table size mismatch");

#define SCE_OK 0
// NOLINTEND(readability-identifier-naming)
