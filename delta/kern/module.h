#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "proc.h"
#include <elf_types.h>
#include <sce_types.h>

#include <base/containers/vector.h>
#include <base/strings/xstring.h>
#include <base/memory/unique_pointer.h>

namespace utl {
class File;
}

namespace krnl {
struct moduleSeg {
  uint8_t *addr;
  uint32_t size;
};

struct moduleInfo {
  base::String name;
  uint32_t handle;
  uint8_t *base;
  uint8_t *entry;
  uint16_t tlsSlot;
  uint32_t codeSize;

  uint8_t *ripZone;
  size_t ripZoneSize;

  uint8_t *procParam;
  uint32_t procParamSize;

  // per-library SCE module param (PT_SCE_MODULEPARAM); queried via
  // sys_dynlib_get_obj_member index 8 to validate the module's SDK version.
  uint8_t *moduleParam;
  uint32_t moduleParamSize;

  uint8_t *initAddr;
  uint8_t *finiAddr;
  bool initRan = false;  // DT_INIT already executed (loader runs it for some PRX)

  moduleSeg textSeg;
  moduleSeg dataSeg;

  uint8_t *tlsAddr;
  size_t tlsSizeMem;
  size_t tlsSizeFile;
  uint32_t tlsalign;

  uint8_t *ehFrameheaderAddr;
  uint8_t *ehFrameAddr;
  uint32_t ehFrameheaderSize;
  uint32_t ehFrameSize;

  uint8_t fingerprint[20];
};

class smodule {
  friend class proc;

public:
  explicit smodule(proc *);

  bool fromFile(const base::String &);
  // Load a module from a guest VFS path (host or virtual mount). Converts a
  // fake SELF to an ELF on the fly, so it works for a pkg's eboot.bin.
  bool fromVfs(const base::String &);
  bool fromMem(base::UniquePointer<uint8_t[]>);

  uintptr_t getSymbol(uint64_t);
  uintptr_t getSymbolFullName(const char *name);
  uintptr_t getSymbol2(const char *name);
  // Resolve an exported symbol by its 11-char NID prefix (export strtab names
  // are "<nid>#<libid>#<modid>"; dlsym only knows the NID, not the inner ids).
  uintptr_t getSymbolByNid(const char *nid);
  bool resolveObfSymbol(const char *name, uintptr_t &ptrOut);

  bool applyRelocations();
  bool resolveImports();

  bool unload();

  inline moduleInfo &getInfo() { return info; }

  // DT_NEEDED module names (".prx" suffix stripped), for dependency-ordered
  // module enumeration (sys_dynlib_get_list).
  inline const base::Vector<base::String> &neededObjects() const {
    return sharedObjects;
  }

  inline bool isDynlib() { return elf->type == ET_SCE_DYNAMIC; }

  /*traits -> object_ref TODO: properly implement*/
  void release(){};
  void retain(){};

private:
  moduleInfo info{};

  void digestDynamic();
  // PS5 modules drop PT_SCE_DYNLIBDATA and use standard ELF dynamic tags; parsed
  // on this separate path so PS4 handling stays byte-identical.
  void digestDynamicPs5(const ELFPgHeader *dynS);
  void logDbgInfo();
  void installEHFrame();
  bool setupTLS();
  bool mapImage();

  template <typename Type, typename TAdd> Type *getOffset(const TAdd dist) {
    return (Type *)(data.Get_UseOnlyIfYouKnowWhatYouareDoing() + dist);
  }

  template <typename Type, typename TAdd> Type *getAddress(const TAdd dist) {
    return (Type *)(info.base + dist);
  }

  template <typename Type, typename TAdd> Type getAddressNPTR(const TAdd dist) {
    return (Type)(info.base + dist);
  }

  template <typename Type = ELFPgHeader> Type *getSegment(ElfSegType type) {
    for (uint16_t i = 0; i < elf->phnum; i++) {
      auto s = &segments[i];
      if (s->type == type)
        return reinterpret_cast<Type *>(s);
    }

    return nullptr;
  }

private:
  base::UniquePointer<uint8_t[]> data;

private:
  proc *process;
  ELFHeader *elf;
  ELFPgHeader *segments;

  struct libInfo {
    const char *name;
    int32_t id;
    uint16_t attr;
    bool exported;
  };

  struct modInfo {
    const char *name;
    int32_t id;
    uint16_t attr;
  };

  base::Vector<modInfo> impModules;
  base::Vector<libInfo> impLibs;
  base::Vector<base::String> sharedObjects;

  // True for a PS5 (Prospero) module: standard-ELF dynamic layout, no
  // PT_SCE_DYNLIBDATA. Set by digestDynamic(); gates the PS5-only code path.
  bool ps5Layout = false;

  // filled in by digestDynamic() from DT_ entries. must default to zero: a
  // module that omits one would otherwise relocate against garbage.
  ElfRel *jmpslots = nullptr;
  ElfRel *rela = nullptr;
  ElfSym *symbols = nullptr;
  uint8_t *hashes = nullptr;

  struct table {
    char *ptr = nullptr;
    size_t size = 0;
  };

  table strtab;
  table symtab;

  uint32_t numJmpSlots = 0;
  uint32_t numSymbols = 0;
  uint32_t numRela = 0;

  // applyRelocations must run at most once: the TLS relocs (DTPMOD64/DTPOFF)
  // are additive (+=), so a second pass (the harness relocates, then the guest
  // libkernel calls syscall 599 too) would double the module's TLS index.
  bool relocated = false;
};
}
