
// Copyright (C) 2019 Force67

#include "base/arch.h"
#include <cstdio>
#include <base/memory/unique_pointer.h>

struct SELFHeader {
  u32 magic;
  u32 unk;
  u8 contentType;
  u8 productType;
  u16 pad;
  u16 headerSize;
  u16 signatureSize;
  u32 sizeSELF; // < unrounded img size
  u32 pad1;
  u16 numSegments;
  u16 unk1; //< always 0x22
  u32 pad2;
};

struct ELFHeader {
  u8 ident[16];
  u16 type;
  u16 machine;
  u32 version;
  u64 entry;
  u64 phoff;
  u64 shoff; //< null
  u32 flags;
  u16 ehsize;
  u16 phentsize;
  u16 phnum;
  u16 shentsize;
  u16 shnum;
  u16 shstrndx;
};

struct ELFPgHeader {
  u32 type;
  u32 flags;
  u64 offset;
  u64 vaddr;
  u64 paddr;
  u64 filesz;
  u64 memsz;
  u32 flags1;
  u16 align;
};

int main(int argc, char **argv) {
  std::puts("SELF_Edit - Copyright (C) Force67 2019");

  if (argc < 2) {

    std::puts(">Usage: sedit.exe <filename> -options");
    return -1;
  }

  // open input file
  FILE *file = fopen(argv[1], "rb");
  if (file) {

    // determine the size
    fseek(file, 0, SEEK_END);
    u32 len = ftell(file);
    auto data = base::MakeUnique<u8[]>(len);

    // read file in buffer
    fseek(file, 0, SEEK_SET);
    fread(data.get(), 1, len, file);
    fclose(file);

    SELFHeader *self = (SELFHeader *)data.get();
    if (self->magic == 0x1D3D154F) {

      char newName[512];
      strcpy(newName, argv[1]);

      size_t len = strlen(newName);
      strcpy(&newName[len - 4], ".elf");

      file = fopen(newName, "wb");
      if (!file)
        return -2;

      // dump elf header
      ELFHeader *elf = (ELFHeader *)(data.get() + sizeof(SELFHeader) +
                                     (self->numSegments * 32));
      fwrite(elf, 64, 1, file);

      auto *segments = (ELFPgHeader *)((u8 *)elf + elf->phoff);
      fwrite(segments, sizeof(ELFPgHeader) * elf->phnum, 1, file);
      fclose(file);
    }
  }

  return 0;
}