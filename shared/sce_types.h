#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Sony Elf custom extensions

#include <cstdint>

// increments by 0x10 for each new
// revision
enum class SELFProductType : uint8_t {
  PUP,
  K = 0xC,
  SL = 0xF,
  SM = 0xE,
  SELF = 0x8, //< applies to EBOOT, ELF and SELF
  SPRX = 0x9  //< applies to SPRX, SDLL and SEXE
};

enum class SELFContentType : uint8_t { SELF = 1, PUP = 4 };

enum SegFlags {
  SF_ORDR = 0x1,   //< ordered?
  SF_ENCR = 0x2,   //< encrypted
  SF_SIGN = 0x4,   //< signed
  SF_DFLG = 0x8,   //< deflated
  SF_BFLG = 0x800, //< block segment
};

static constexpr uint32_t SELF_MAGIC = 0x1D3D154F;

struct SELFHeader {
  uint32_t magic;
  uint8_t version;
  uint8_t mode;
  uint8_t endian;
  uint8_t attr;

  // actually keyType
  SELFContentType contentType;
  SELFProductType productType;
  uint16_t pad;

  uint16_t headerSize;
  uint16_t metaSize; // < sce Special
  uint32_t sizeSELF; // < unrounded img size
  uint32_t fileSize;

  uint16_t numSegments;
  uint16_t flags; //< always 0x22
  uint32_t pad2;  //<alignment
};

struct SELFSegmentTable {
  uint64_t flags;
  uint64_t offset;
  uint64_t fileSize;
  uint64_t memSize;

  uint32_t Id() { return static_cast<uint32_t>((uint64_t)flags >> 20); }
};

struct SCEContentId {
  char pad[0x20];
};

// is it really called "SCESPECIAL"?
struct SCESpecial {
  uint64_t authId;
  uint64_t productType;
  uint64_t version1;
  uint64_t version2;
  SCEContentId contentId;
  char shaSum[0x20];
};

// similar to MS PDB_CODEVIEW
struct SCEComment {
  uint32_t magic; // "PATH"
  uint32_t unk;
  uint32_t pathLength; // length of the following path
};

static_assert(sizeof(SELFHeader) == 32, "header size mismatch");
static_assert(sizeof(SELFSegmentTable) == 32, "segment table size mismatch");

#define SCE_OK 0