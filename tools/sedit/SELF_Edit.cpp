
// Copyright (C) 2019 Force67

#include <cstdint>
#include <cstdio>
#include <memory>

struct SELFHeader {
  uint32_t magic;
  uint32_t unk;
  uint8_t contentType;
  uint8_t productType;
  uint16_t pad;
  uint16_t headerSize;
  uint16_t signatureSize;
  uint32_t sizeSELF; // < unrounded img size
  uint32_t pad1;
  uint16_t numSegments;
  uint16_t unk1; //< always 0x22
  uint32_t pad2;
};

struct ELFHeader {
  uint8_t ident[16];
  uint16_t type;
  uint16_t machine;
  uint32_t version;
  uint64_t entry;
  uint64_t phoff;
  uint64_t shoff; //< null
  uint32_t flags;
  uint16_t ehsize;
  uint16_t phentsize;
  uint16_t phnum;
  uint16_t shentsize;
  uint16_t shnum;
  uint16_t shstrndx;
};

struct ELFPgHeader {
  uint32_t type;
  uint32_t flags;
  uint64_t offset;
  uint64_t vaddr;
  uint64_t paddr;
  uint64_t filesz;
  uint64_t memsz;
  uint32_t flags1;
  uint16_t align;
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
    uint32_t len = ftell(file);
    auto data = std::make_unique<uint8_t[]>(len);

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

      auto *segments = (ELFPgHeader *)((uint8_t *)elf + elf->phoff);
      fwrite(segments, sizeof(ELFPgHeader) * elf->phnum, 1, file);
      fclose(file);
    }
  }

  return 0;
}