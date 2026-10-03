#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "base/atomic.h"
#include "elf_types.h"
#include "sce_types.h"

#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"

namespace io {
class File;
}

namespace kern {
class Process;
struct ModuleSeg {
  u8* addr;
  u32 size;
};

struct ModuleInfo {
  base::String name;
  u32 handle;
  u8* base;
  u8* entry;
  u16 tls_slot;
  u32 code_size;

  u8* rip_zone;
  size_t rip_zone_size;

  u8* proc_param;
  u32 proc_param_size;

  // per-library SCE module param (PT_SCE_MODULEPARAM); queried via
  // sys_dynlib_get_obj_member index 8 to validate the module's SDK version.
  u8* module_param;
  u32 module_param_size;

  u8* init_addr;
  u8* fini_addr;
  bool init_ran =
      false;  // DT_INIT already executed (loader runs it for some PRX)

  ModuleSeg text_seg;
  ModuleSeg data_seg;

  u8* tls_addr;
  size_t tls_size_mem;
  size_t tls_size_file;
  u32 tlsalign;

  u8* eh_frameheader_addr;
  u8* eh_frame_addr;
  u32 eh_frameheader_size;
  u32 eh_frame_size;

  u8 fingerprint[20];
};

class Module {
  friend class Process;

 public:
  explicit Module(Process*);

  bool FromFile(const base::String&);
  // Load a module from a guest VFS path (host or virtual mount). Converts a
  // fake SELF to an ELF on the fly, so it works for a pkg's eboot.bin.
  bool FromVfs(const base::String&);
  bool FromMem(base::UniquePointer<u8[]>);

  uintptr_t GetSymbol(u64);
  // LLE-only export lookup by NID (GetSymbol without the HLE/vprx override),
  // for the PS5 global-NID resolver.
  uintptr_t GetExport(u64 nid);
  uintptr_t GetSymbolFullName(const char* name);
  uintptr_t GetSymbol2(const char* name);
  // Resolve an exported symbol by its 11-char NID prefix (export strtab names
  // are "<nid>#<libid>#<modid>"; dlsym only knows the NID, not the inner ids).
  uintptr_t GetSymbolByNid(const char* nid);
  bool ResolveObfSymbol(const char* name, uintptr_t& ptr_out);

  bool ApplyRelocations();
  bool ResolveImports();

  // Imports that landed on the badcall stub the last time ResolveImports ran,
  // i.e. slots a module loaded later could still satisfy.
  inline bool HasUnresolvedImports() const { return unresolved_imports_ != 0; }

  bool Unload();

  inline ModuleInfo& GetInfo() { return info_; }

  // DT_NEEDED module names (".prx" suffix stripped), for dependency-ordered
  // module enumeration (sys_dynlib_get_list).
  inline const base::Vector<base::String>& NeededObjects() const {
    return shared_objects_;
  }

  inline bool IsDynlib() { return elf_->type == ET_SCE_DYNAMIC; }

  /*traits -> ObjectRef TODO: properly implement*/
  void Release() {
    if (references_.fetch_sub(1) == 1)
      delete this;
  }
  void Retain() { references_.fetch_add(1); }

 private:
  base::Atomic<u32> references_{1};
  ModuleInfo info_{};

  void DigestDynamic();
  // PS5 modules drop PT_SCE_DYNLIBDATA and use standard ELF dynamic tags;
  // parsed on this separate path so PS4 handling stays byte-identical.
  void DigestDynamicPs5(const ELFPgHeader* dyn_s);
  void LogDbgInfo();
  void InstallEhFrame();
  bool SetupTls();
  bool MapImage();
  void StartModuleWatch();
  void PlantGuestBreakpoints();

  template <typename Type, typename TAdd>
  Type* GetOffset(const TAdd dist) {
    return (Type*)(data_.Get_UseOnlyIfYouKnowWhatYouareDoing() + dist);
  }

  template <typename Type, typename TAdd>
  Type* GetAddress(const TAdd dist) {
    return (Type*)(info_.base + dist);
  }

  template <typename Type, typename TAdd>
  Type GetAddressNptr(const TAdd dist) {
    return (Type)(info_.base + dist);
  }

  template <typename Type = ELFPgHeader>
  Type* GetSegment(ElfSegType type) {
    for (u16 i = 0; i < elf_->phnum; i++) {
      auto s = &segments_[i];
      if (s->type == type)
        return reinterpret_cast<Type*>(s);
    }

    return nullptr;
  }

 private:
  base::UniquePointer<u8[]> data_;

 private:
  Process* process_;
  ELFHeader* elf_;
  ELFPgHeader* segments_;

  struct LibInfo {
    const char* name;
    i32 id;
    u16 attr;
    bool exported;
  };

  struct ModInfo {
    const char* name;
    i32 id;
    u16 attr;
  };

  base::Vector<ModInfo> imp_modules_;
  base::Vector<LibInfo> imp_libs_;
  base::Vector<base::String> shared_objects_;

  // True for a PS5 (Prospero) module: standard-ELF dynamic layout, no
  // PT_SCE_DYNLIBDATA. Set by DigestDynamic(); gates the PS5-only code path.
  bool ps5_layout_ = false;

  // filled in by DigestDynamic() from DT_ entries. must default to zero: a
  // module that omits one would otherwise relocate against garbage.
  ElfRel* jmpslots_ = nullptr;
  ElfRel* rela_ = nullptr;
  ElfSym* symbols_ = nullptr;
  u8* hashes_ = nullptr;

  struct Table {
    char* ptr = nullptr;
    size_t size = 0;
  };

  Table strtab_;
  Table symtab_;

  // NID -> st_value of every defined symbol, built at the first GetExport:
  // every import of every module asks every loaded module.
  base::HashMap<u64, u64> export_index_;
  bool export_index_built_ = false;

  u32 num_jmp_slots_ = 0;
  u32 num_symbols_ = 0;
  u32 num_rela_ = 0;

  // ApplyRelocations must run at most once: the TLS relocs (DTPMOD64/DTPOFF)
  // are additive (+=), so a second pass (the harness relocates, then the guest
  // libkernel calls syscall 599 too) would double the module's TLS index.
  bool relocated_ = false;

  u32 unresolved_imports_ = 0;
  bool imports_bound_ = false;
};
}  // namespace kern
