#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "proc.h"
#include "base/arch.h"
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
  u8 *addr;
  u32 size;
};

struct moduleInfo {
  base::String name;
  u32 handle;
  u8 *base;
  u8 *entry;
  u16 tlsSlot;
  u32 codeSize;

  u8 *ripZone;
  size_t ripZoneSize;

  u8 *procParam;
  u32 procParamSize;

  // per-library SCE module param (PT_SCE_MODULEPARAM); queried via
  // sys_dynlib_get_obj_member index 8 to validate the module's SDK version.
  u8 *moduleParam;
  u32 moduleParamSize;

  u8 *initAddr;
  u8 *finiAddr;
  bool initRan = false;  // DT_INIT already executed (loader runs it for some PRX)

  moduleSeg textSeg;
  moduleSeg dataSeg;

  u8 *tlsAddr;
  size_t tlsSizeMem;
  size_t tlsSizeFile;
  u32 tlsalign;

  u8 *ehFrameheaderAddr;
  u8 *ehFrameAddr;
  u32 ehFrameheaderSize;
  u32 ehFrameSize;

  u8 fingerprint[20];
};

class smodule {
  friend class proc;

public:
  explicit smodule(proc *);

  bool fromFile(const base::String &);
  // Load a module from a guest VFS path (host or virtual mount). Converts a
  // fake SELF to an ELF on the fly, so it works for a pkg's eboot.bin.
  bool fromVfs(const base::String &);
  bool fromMem(base::UniquePointer<u8[]>);

  uintptr_t getSymbol(u64);
  // LLE-only export lookup by NID (getSymbol without the HLE/vprx override), for
  // the PS5 global-NID resolver.
  uintptr_t getExport(u64 nid);
  uintptr_t getSymbolFullName(const char *name);
  uintptr_t getSymbol2(const char *name);
  // Resolve an exported symbol by its 11-char NID prefix (export strtab names
  // are "<nid>#<libid>#<modid>"; dlsym only knows the NID, not the inner ids).
  uintptr_t getSymbolByNid(const char *nid);
  bool resolveObfSymbol(const char *name, uintptr_t &ptrOut);

  bool applyRelocations();
  bool resolveImports();

  // Imports that landed on the badcall stub the last time resolveImports ran,
  // i.e. slots a module loaded later could still satisfy.
  inline bool hasUnresolvedImports() const { return unresolvedImports != 0; }

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
  void startModuleWatch();
  void plantGuestBreakpoints();

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
    for (u16 i = 0; i < elf->phnum; i++) {
      auto s = &segments[i];
      if (s->type == type)
        return reinterpret_cast<Type *>(s);
    }

    return nullptr;
  }

private:
  base::UniquePointer<u8[]> data;

private:
  proc *process;
  ELFHeader *elf;
  ELFPgHeader *segments;

  struct libInfo {
    const char *name;
    i32 id;
    u16 attr;
    bool exported;
  };

  struct modInfo {
    const char *name;
    i32 id;
    u16 attr;
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
  u8 *hashes = nullptr;

  struct table {
    char *ptr = nullptr;
    size_t size = 0;
  };

  table strtab;
  table symtab;

  u32 numJmpSlots = 0;
  u32 numSymbols = 0;
  u32 numRela = 0;

  // applyRelocations must run at most once: the TLS relocs (DTPMOD64/DTPOFF)
  // are additive (+=), so a second pass (the harness relocates, then the guest
  // libkernel calls syscall 599 too) would double the module's TLS index.
  bool relocated = false;

  u32 unresolvedImports = 0;
  bool importsBound = false;
};
}
