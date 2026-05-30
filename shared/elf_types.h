#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Elf spec with Sony additions

#include <cstdint>

enum ELFDynTag {
  DT_NULL,   //< end of dyn section
  DT_NEEDED, //< required library
  DT_PLTRELSZ,
  DT_PLTGOT,
  DT_HASH,
  DT_STRTAB,
  DT_SYMTAB,
  DT_RELA,
  DT_RELASZ,
  DT_RELAENT,
  DT_STRSZ,
  DT_SYMENT,
  DT_INIT,
  DT_FINI,
  DT_SONAME,
  DT_RPATH,
  DT_SYMBOLIC,
  DT_REL,
  DT_RELSZ,
  DT_RELENT,
  DT_PLTREL,
  DT_DEBUG,
  DT_TEXTREL,
  DT_JMPREL,
  DT_BIND_NOW,
  DT_INIT_ARRAY,
  DT_FINI_ARRAY,
  DT_INIT_ARRAYSZ,
  DT_FINI_ARRAYSZ,
  DT_RUNPATH,
  DT_FLAGS,
  DT_ENCODING,
  DT_PREINIT_ARRAY,
  DT_PREINIT_ARRAYSZ,
  DT_SCE_FINGERPRINT = 0x61000007,
  DT_SCE_ORIGFILENAME = 0x61000009,
  DT_SCE_MODULEINFO = 0x6100000d,
  DT_SCE_NEEDED_MODULE = 0x6100000f,
  DT_SCE_MODULE_ATTR = 0x61000011,
  DT_SCE_EXPLIB = 0x61000013,
  DT_SCE_IMPLIB = 0x61000015,
  DT_SCE_EXPORT_LIB_ATTR = 0x61000017,
  DT_SCE_IMPORT_LIB_ATTR = 0x61000019,
  DT_SCE_STRTAB = 0x61000035,
  DT_SCE_STRSIZE = 0x61000037,
  DT_SCE_JMPREL = 0x61000029,
  DT_SCE_PLTREL = 0x6100002b,
  DT_SCE_PLTRELSZ = 0x6100002d,
  DT_SCE_SYMTAB = 0x61000039,
  DT_SCE_SYMTABSZ = 0x6100003f,
  DT_SCE_RELA = 0x6100002f,
  DT_SCE_RELASZ = 0x61000031,
  DT_SCE_HASH = 0x61000025,
};

enum sceModuleAttr {
  NO_ATTR,
  SCE_CANT_STOP,
  SCE_EXCLUSIVE_LOAD,
  SCE_EXCLUSIVE_START = 4,
  SCE_CAN_RESTART = 8,
  SCE_CAN_RELOCATE = 0x10,
  SCE_CANT_SHARE = 0x20
};

enum ElfSegType {
  PT_NULL,
  PT_LOAD,
  PT_DYNAMIC,
  PT_INTERP,
  PT_NOTE,
  PT_SHLIB,
  PT_PHDR,
  PT_TLS,
  PT_LOPROC = 0x70000000,
  PT_HIPROC = 0x7FFFFFFF,
  PT_SCE_DYNLIBDATA = 0x61000000,
  PT_SCE_PROCPARAM = 0x61000001,
  PT_SCE_MODULEPARAM = 0x61000002,
  PT_SCE_RELRO = 0x61000010,
  PT_SCE_COMMENT = 0x6FFFFF00,
  PT_SCE_LIBVERSION = 0x6FFFFF01,
  PT_GNU_EH_FRAME = 0x6474E550,
  PT_GNU_STACK = 0x6474E551,
};

enum ElfRelType { R_X86_64_JUMP_SLOT = 7 };

enum ElfFlags { PF_X = 1, PF_W = 2, PF_R = 4, PF_MASKPROC = 0xF0000000 };

enum ELFTypeSCE : uint16_t {
  ET_SCE_EXEC = 0xfe00,    // PS4 Executable
  ET_SCE_DYNEXEC = 0xfe10, // PS4 Main module
  ET_SCE_RELEXEC = 0xfe04, // PS4 Relocatable PRX
  ET_SCE_STUBLIB = 0xfe0c, // PS4 Stub library
  ET_SCE_DYNAMIC = 0xfe18, // PS4 Dynamic PRX
};

static constexpr uint32_t ELF_MAGIC = 0x464C457F;
static constexpr uint16_t ELF_MACHINE_X86_64 = 0x3E;

struct ELFHeader {
  uint32_t magic;
  uint8_t ident[12]; // < actually 16
  ELFTypeSCE type;   //< sony custom
  uint16_t machine;
  uint32_t version;
  uint64_t entry;
  uint64_t phoff; //< program header offset
  uint64_t shoff; //< does sometimes not exist
  uint32_t flags;
  uint16_t ehsize;
  uint16_t phentsize; //< size in bytes of entries in pg table
  uint16_t phnum;
  uint16_t shentsize;
  uint16_t shnum;
  uint16_t shstrndx;
};

struct ELFPgHeader {
  uint32_t type;  //< data info
  uint32_t flags; //< memory protection flags
  uint64_t offset;
  uint64_t vaddr;
  uint64_t paddr;
  uint64_t filesz;
  uint64_t memsz;
  uint64_t align;
};

struct ELFDyn {
  int64_t tag;
  union {
    uint64_t value;
    uint64_t ptr;
  } un;
};

struct ElfRel {
  uint64_t offset;
  uint64_t info;
  int64_t addend;
};

struct ElfSym {
  uint32_t st_name;
  uint8_t st_info;
  uint8_t st_other;
  uint16_t st_shndx;
  uint64_t st_value;
  uint64_t st_size;
};

static_assert(sizeof(ELFPgHeader) == 56, "Elf program header size mismatch");
static_assert(sizeof(ELFHeader) == 64, "Elf header size mismatch");

inline const char *ElfTypeToString(uint32_t type) {
  switch (type) {
  case ET_SCE_EXEC:
    return "Executable";
  case ET_SCE_DYNEXEC:
    return "Main module";
  case ET_SCE_RELEXEC:
    return "Relocatable PRX";
  case ET_SCE_STUBLIB:
    return "Stub library";
  case ET_SCE_DYNAMIC:
    return "Dynamic PRX";
  default:
    return "Unknown";
  }
}

inline const char *SegTypeToString(uint32_t type) {
#define AS_STR(idx)                                                            \
  if (type == idx)                                                             \
    return #idx;

  AS_STR(PT_LOAD)
  AS_STR(PT_DYNAMIC)
  AS_STR(PT_INTERP)
  AS_STR(PT_NOTE)
  AS_STR(PT_SHLIB)
  AS_STR(PT_PHDR)
  AS_STR(PT_TLS)
  AS_STR(PT_LOPROC)
  AS_STR(PT_HIPROC)
  AS_STR(PT_SCE_DYNLIBDATA)
  AS_STR(PT_SCE_PROCPARAM)
  AS_STR(PT_SCE_MODULEPARAM)
  AS_STR(PT_SCE_RELRO)
  AS_STR(PT_SCE_COMMENT)
  AS_STR(PT_SCE_LIBVERSION)
  AS_STR(PT_GNU_EH_FRAME)
  AS_STR(PT_GNU_STACK)

#undef AS_STR

  return "Unknown";
}

#define ELF64_R_SYM(i) ((i) >> 32)
#define ELF64_R_TYPE(i) ((i)&0xffffffff)

/* Relocation types for AMD x86-64 architecture */
#define R_X86_64_NONE 0      /* No reloc */
#define R_X86_64_64 1        /* Direct 64 bit  */
#define R_X86_64_PC32 2      /* PC relative 32 bit signed */
#define R_X86_64_GOT32 3     /* 32 bit GOT entry */
#define R_X86_64_PLT32 4     /* 32 bit PLT address */
#define R_X86_64_COPY 5      /* Copy symbol at runtime */
#define R_X86_64_GLOB_DAT 6  /* Create GOT entry */
#define R_X86_64_JUMP_SLOT 7 /* Create PLT entry */
#define R_X86_64_RELATIVE 8  /* Adjust by program base */
#define R_X86_64_GOTPCREL 9  /* 32 bit signed pc relative offset to GOT */
#define R_X86_64_32 10       /* Direct 32 bit zero extended */
#define R_X86_64_32S 11      /* Direct 32 bit sign extended */
#define R_X86_64_16 12       /* Direct 16 bit zero extended */
#define R_X86_64_PC16 13     /* 16 bit sign extended pc relative */
#define R_X86_64_8 14        /* Direct 8 bit sign extended  */
#define R_X86_64_PC8 15      /* 8 bit sign extended pc relative */
#define R_X86_64_DTPMOD64 16 /* ID of module containing symbol */
#define R_X86_64_DTPOFF64 17 /* Offset in module's TLS block */
#define R_X86_64_TPOFF64 18  /* Offset in initial TLS block */
#define R_X86_64_TLSGD                                                         \
  19 /* 32 bit signed PC relative offset                                       \
                                                 to two GOT entries for GD     \
        symbol */
#define R_X86_64_TLSLD                                                                             \
  20                         /* 32 bit signed PC relative offset                                   \
                                                                         to two GOT entries for LD \
                                symbol */
#define R_X86_64_DTPOFF32 21 /* Offset in TLS block */
#define R_X86_64_GOTTPOFF                                                      \
  22                           /* 32 bit signed PC relative offset             \
                                                                           to GOT entry for IE symbol */
#define R_X86_64_TPOFF32 23    /* Offset in initial TLS block */
#define R_X86_64_PC64 24       /* PC relative 64 bit */
#define R_X86_64_GOTOFF64 25   /* 64 bit offset to GOT */
#define R_X86_64_GOTPC32 26    /* 32 bit signed pc relative offset to GOT */
#define R_X86_64_GOT64 27      /* 64-bit GOT entry offset */
#define R_X86_64_GOTPCREL64 28 /* 64-bit PC relative offset to GOT entry */
#define R_X86_64_GOTPC64 29    /* 64-bit PC relative offset to GOT */
#define R_X86_64_GOTPLT64 30   /* like GOT64, says PLT entry needed */
#define R_X86_64_PLTOFF64 31   /* 64-bit GOT relative offset to PLT entry */
#define R_X86_64_SIZE32 32     /* Size of symbol plus 32-bit addend */
#define R_X86_64_SIZE64 33     /* Size of symbol plus 64-bit addend */
#define R_X86_64_GOTPC32_TLSDESC 34 /* GOT offset for TLS descriptor */
#define R_X86_64_TLSDESC_CALL 35    /* Marker for call through TLS descriptor */
#define R_X86_64_TLSDESC 36         /* TLS descriptor */
#define R_X86_64_IRELATIVE 37       /* Adjust indirectly by program base */
#define R_X86_64_RELATIVE64 38      /* 64bit adjust by program base */
#define R_X86_64_ORBIS_GOTPCREL_LOAD 40

#define ELF_ST_BIND(x) ((x) >> 4)
#define ELF_ST_TYPE(x) (((unsigned int)x) & 0xf)
#define ELF32_ST_BIND(x) ELF_ST_BIND(x)
#define ELF32_ST_TYPE(x) ELF_ST_TYPE(x)
#define ELF64_ST_BIND(x) ELF_ST_BIND(x)
#define ELF64_ST_TYPE(x) ELF_ST_TYPE(x)
#define ELF_ST_VISIBILITY(o) ((o)&0x3)

#define STB_LOCAL 0
#define STB_GLOBAL 1
#define STB_WEAK 2