
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <sys/mman.h>
#include "base/arch.h"
#include "base/logging.h"
#include "base/math/alignment.h"
#include "host_memory/host_memory.h"
#include "io/file.h"

#if defined(DELTA_BACKEND_NATIVE)
#include "runtime/code_lift.h"
#endif
#include "cpu/backend.h"
#include "runtime/vprx/vprx.h"

#include "formats/fself.h"

#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"
#include "base/threading/thread.h"
#include "kern/module.h"
#include "kern/probe/probe.h"
#include "kern/process.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kDumpModule, "DELTA_DUMP_MODULE", nullptr);
DELTA_OPTION(const char*, kGuestBrk, "DELTA_GUEST_BRK", nullptr);
DELTA_OPTION(u64, kBrkAfter, "DELTA_GUEST_BRK_AFTER", 0);
DELTA_OPTION(const char*, kNoExec, "DELTA_GUEST_NOEXEC", nullptr);
DELTA_OPTION(const char*, kModCheck, "DELTA_MODCHECK", nullptr);
DELTA_OPTION(bool, kLibkDebug, "DELTA_LIBK_DEBUG", false);
DELTA_OPTION(bool, kImplibTrace, "DELTA_IMPLIB_TRACE", false);
DELTA_OPTION(bool, kRelocTrace, "DELTA_RELOC_TRACE", false);
}  // namespace

namespace kern {
Module::Module(Process* process) : process_(process) {
  /*-1 = no tls used*/
  info_.handle = -1;
  info_.tls_slot = -1;

  /*set size from process config*/
  info_.rip_zone_size = process->GetEnv().rip_zone_size;
}

bool Module::FromFile(const base::String& path) {
  io::File file(path);
  if (!file.IsOpen()) {
    // missing dep on disk; fail soft so the caller can keep going
    LOG_ERROR("smodule: cannot open {}", path.c_str());
    return false;
  }

  ELFHeader disk_header{};
  file.Read(disk_header);

  if (disk_header.magic == ELF_MAGIC &&
      disk_header.machine == ELF_MACHINE_X86_64) {
    file.Seek(0, io::SeekMode::kSeekSet);

    auto sz = file.GetSize();
    data_ = base::MakeUnique<u8[]>(static_cast<mem_size>(sz));
    file.Read(data_.Get_UseOnlyIfYouKnowWhatYouareDoing(), sz);
    return FromMem(base::move(data_));
  }

  // Not a raw x86-64 (SCE) ELF, e.g. still SELF-encrypted, or a different
  // arch. We don't decrypt here, so reject it.
  LOG_ERROR("smodule: {} is not a decrypted x86-64 ELF", path.c_str());
  return false;
}

bool Module::FromVfs(const base::String& guest_path) {
  io::File f = vfs::OpenRead(guest_path.c_str());
  if (!f.Exists()) {
    LOG_ERROR("smodule: cannot open vfs path {}", guest_path.c_str());
    return false;
  }

  auto sz = f.GetSize();
  if (sz < sizeof(ELFHeader))
    return false;

  base::Vector<u8> raw;
  raw.resize(static_cast<mem_size>(sz));
  f.Read(raw.data(), sz);

  const u8* src = raw.data();
  size_t src_size = static_cast<size_t>(sz);

  // eboot.bin / prx in a pkg are fake SELFs: rebuild the plain ELF in memory.
  base::Vector<u8> elf_img;
  u32 magic = static_cast<u32>(src[0]) | (static_cast<u32>(src[1]) << 8) |
              (static_cast<u32>(src[2]) << 16) |
              (static_cast<u32>(src[3]) << 24);
  if (isSelfMagic(magic)) {
    elf_img = formats::FselfToElf(src, src_size);
    if (elf_img.empty()) {
      LOG_ERROR("smodule: self2elf failed for {}", guest_path.c_str());
      return false;
    }
    src = elf_img.data();
    src_size = elf_img.size();
  } else if (magic != ELF_MAGIC) {
    LOG_ERROR("smodule: {} is neither SELF nor ELF", guest_path.c_str());
    return false;
  }

  if (const char* dd = kDumpModule) {
    if (guest_path.find(dd) != base::String::npos) {
      if (FILE* f = std::fopen("/tmp/dumped_module.elf", "wb")) {
        std::fwrite(src, 1, src_size, f);
        std::fclose(f);
        LOG_INFO("dumped decrypted {} -> /tmp/dumped_module.elf ({} bytes)",
                 guest_path.c_str(), src_size);
      }
    }
  }

  auto out = base::MakeUnique<u8[]>(static_cast<mem_size>(src_size));
  std::memcpy(out.Get_UseOnlyIfYouKnowWhatYouareDoing(), src, src_size);
  return FromMem(base::move(out));
}

bool Module::FromMem(base::UniquePointer<u8[]> data) {
  this->data_ = base::move(data);

  elf_ = GetOffset<ELFHeader>(0);
  segments_ = GetOffset<ELFPgHeader>(elf_->phoff);

  if (!MapImage()) {
    LOG_ERROR("smodule: Failed to map image");
    __builtin_trap();
    return false;
  }

  DigestDynamic();

#ifdef _DEBUG
  LOG_TRACE("mapped {} at {}", info.name.c_str(),
            static_cast<const void*>(info.base));
#endif
  SetupTls();

  if (!IsDynlib()) {
    auto* seg = GetSegment(ElfSegType::PT_SCE_PROCPARAM);
    if (seg) {
      info_.proc_param = GetAddress<u8>(seg->vaddr);
      info_.proc_param_size = seg->filesz;
    }
  } else {
    auto* seg = GetSegment(ElfSegType::PT_SCE_MODULEPARAM);
    if (seg) {
      info_.module_param = GetAddress<u8>(seg->vaddr);
      info_.module_param_size = seg->filesz;
    }
  }

  InstallEhFrame();

  PlantGuestBreakpoints();
  StartModuleWatch();

  if (elf_->entry == 0)
    info_.entry = nullptr;
  else
    info_.entry = GetAddress<u8>(elf_->entry);

  for (auto& it : shared_objects_) {
    process_->LoadModule(it);
  }

  return true;
}

bool Module::Unload() {
  data_ = {};

  return true;
}

void Module::DigestDynamic() {
  const auto* dyn_s = GetSegment(ElfSegType::PT_DYNAMIC);
  if (!dyn_s)
    return;
  const auto* dyld_s = GetSegment(ElfSegType::PT_SCE_DYNLIBDATA);

  // PS5 modules have no PT_SCE_DYNLIBDATA: their tables are standard ELF
  // dynamic tags as vaddrs into the mapped image. Route to the PS5 path; PS4
  // untouched.
  if (!dyld_s) {
    ps5_layout_ = true;
    DigestDynamicPs5(dyn_s);
    return;
  }

  u8* dynld_ptr = GetOffset<u8>(dyld_s->offset);
  u8* dynld_addr = GetAddress<u8>(dyld_s->vaddr);
  // std::printf("addr = %p\n", dynldAddr);
  ELFDyn* dynamics = GetOffset<ELFDyn>(dyn_s->offset);
  for (i32 i = 0; i < (dyn_s->filesz / sizeof(ELFDyn)); i++) {
    auto* d = &dynamics[i];

    switch (d->tag) {
      case DT_SCE_HASH:
        hashes_ = reinterpret_cast<u8*>(dynld_ptr + d->un.value);
        break;
      case DT_INIT:
        info_.init_addr = reinterpret_cast<u8*>(dynld_addr + d->un.ptr);
        break;
      case DT_FINI:
        info_.fini_addr = reinterpret_cast<u8*>(dynld_addr + d->un.ptr);
        break;
      case DT_SCE_JMPREL:
        jmpslots_ = (ElfRel*)(dynld_ptr + d->un.ptr);
        break;
      case DT_PLTRELSZ:
      case DT_SCE_PLTRELSZ:
        num_jmp_slots_ = static_cast<u32>(d->un.value / sizeof(ElfRel));
        break;
      case DT_SCE_STRTAB:
        strtab_.ptr = (char*)(dynld_ptr + d->un.ptr);
        break;
      case DT_STRSZ:
      case DT_SCE_STRSIZE:
        strtab_.size = d->un.value;
        break;
      case DT_SCE_SYMTAB:
        symbols_ = reinterpret_cast<ElfSym*>(dynld_ptr + d->un.ptr);
        break;
      case DT_SCE_SYMTABSZ:
        num_symbols_ = static_cast<u32>(d->un.value / sizeof(ElfSym));
        break;
      case DT_SCE_RELA:
        rela_ = reinterpret_cast<ElfRel*>(dynld_ptr + d->un.ptr);
        break;
      case DT_NEEDED: {
        auto name = (const char*)(strtab_.ptr + (d->un.value & 0xFFFFFFFF));
        if (name) {
          base::String xname(name);
          /*quick but (valid?) hack for determining if an object is exported*/
          auto pos = xname.find(".prx");
          if (pos != base::String::npos) {
            shared_objects_.push_back(xname.substr(0, pos));
          }
        }

        break;
      }
      case DT_RELASZ:
      case DT_SCE_RELASZ:
        num_rela_ = static_cast<u32>(d->un.value / sizeof(ElfRel));
        break;
      case DT_SCE_EXPLIB:
      case DT_SCE_IMPLIB: {
        auto& e = imp_libs_.emplace_back();
        e.id = d->un.value >> 48;
        e.exported = d->tag == DT_SCE_EXPLIB;
        e.name = (const char*)(strtab_.ptr + (d->un.value & 0xFFFFFFFF));
        break;
      }
      case DT_SCE_EXPORT_LIB_ATTR:
      case DT_SCE_IMPORT_LIB_ATTR: {
        u16 id = d->un.value >> 48;
        u16 idx = d->un.value & 0xFFF;

        for (auto& mod : imp_libs_) {
          if (mod.id == id) {
            mod.attr = idx;
            break;
          }
        }
        break;
      }
      case DT_SCE_NEEDED_MODULE: {
        auto& e = imp_modules_.emplace_back();
        e.id = d->un.value >> 48;
        e.name = (const char*)(strtab_.ptr + (d->un.value & 0xFFFFFFFF));
        break;
      }
      case DT_SCE_MODULE_ATTR: {
        u16 id = d->un.value >> 48;
        u16 idx = d->un.value & 0xFFF;

        for (auto& mod : imp_modules_) {
          if (mod.id == id) {
            mod.attr = idx;
            break;
          }
        }
        break;
      }
      case DT_SCE_MODULEINFO:
        info_.name = (const char*)(strtab_.ptr + (d->un.value & 0xFFFFFFFF));
        break;
      case DT_SCE_FINGERPRINT:
        std::memcpy(info_.fingerprint, GetOffset<void>(d->un.value), 20);
        break;
    }
  }

  if (kImplibTrace) {
    for (auto& l : imp_libs_)
      BASE_LOGI("implib", "{} id={} {}", info_.name.c_str(), l.id,
                l.name ? l.name : "?");
    for (auto& m : imp_modules_)
      BASE_LOGI("impmod", "{} id={} {}", info_.name.c_str(), m.id,
                m.name ? m.name : "?");
  }
}

// PS5-only dynamic parser: DT_STRTAB/SYMTAB/RELA/JMPREL are vaddrs into the
// mapped image (unlike PS4's DT_SCE_* offsets into PT_SCE_DYNLIBDATA).
// Symbol/reloc format is identical, so ResolveImports/applyRelocations are
// reused unchanged.
void Module::DigestDynamicPs5(const ELFPgHeader* dyn_s) {
  ELFDyn* dynamics = GetOffset<ELFDyn>(dyn_s->offset);
  const int count = static_cast<int>(dyn_s->filesz / sizeof(ELFDyn));

  for (int i = 0; i < count; i++) {
    auto* d = &dynamics[i];
    switch (d->tag) {
      case DT_STRTAB:
        strtab_.ptr = GetAddress<char>(d->un.ptr);
        break;
      case DT_STRSZ:
        strtab_.size = d->un.value;
        break;
      case DT_SYMTAB:
        symbols_ = GetAddress<ElfSym>(d->un.ptr);
        break;
      // PS5 keeps the SCE symbol-table size tag even with a standard DT_SYMTAB.
      case DT_SCE_SYMTABSZ:
        num_symbols_ = static_cast<u32>(d->un.value / sizeof(ElfSym));
        break;
      case DT_RELA:
        rela_ = GetAddress<ElfRel>(d->un.ptr);
        break;
      case DT_RELASZ:
        num_rela_ = static_cast<u32>(d->un.value / sizeof(ElfRel));
        break;
      case DT_JMPREL:
        jmpslots_ = GetAddress<ElfRel>(d->un.ptr);
        break;
      case DT_PLTRELSZ:
        num_jmp_slots_ = static_cast<u32>(d->un.value / sizeof(ElfRel));
        break;
      case DT_HASH:
        hashes_ = GetAddress<u8>(d->un.ptr);
        break;
      case DT_INIT:
        info_.init_addr = GetAddress<u8>(d->un.ptr);
        break;
      case DT_FINI:
        info_.fini_addr = GetAddress<u8>(d->un.ptr);
        break;
      case DT_SCE_PS5_IMPORT_LIB: {
        auto& e = imp_libs_.emplace_back();
        e.id = static_cast<i32>(d->un.value >> 48);
        e.exported = false;
        e.name = nullptr;  // strtab may not be known yet; filled in below
        break;
      }
      case DT_SCE_PS5_IMPORT_MODULE: {
        auto& e = imp_modules_.emplace_back();
        e.id = static_cast<i32>(d->un.value >> 48);
        e.name = nullptr;
        break;
      }
      default:
        break;
    }
  }

  // Both tables index the string table, which DT_STRTAB may only have announced
  // after them; resolve the names in a second pass.
  if (strtab_.ptr) {
    size_t li = 0, mi = 0;
    for (int i = 0; i < count; i++) {
      auto* d = &dynamics[i];
      if (d->tag == DT_SCE_PS5_IMPORT_LIB && li < imp_libs_.size())
        imp_libs_[li++].name = strtab_.ptr + (d->un.value & 0xFFFFFFFF);
      else if (d->tag == DT_SCE_PS5_IMPORT_MODULE && mi < imp_modules_.size())
        imp_modules_[mi++].name = strtab_.ptr + (d->un.value & 0xFFFFFFFF);
    }
  }

  // DT_NEEDED entries usually precede DT_STRTAB, so resolve their names only
  // once the string table pointer is known.
  if (strtab_.ptr) {
    for (int i = 0; i < count; i++) {
      auto* d = &dynamics[i];
      if (d->tag != DT_NEEDED)
        continue;
      const char* name = strtab_.ptr + (d->un.value & 0xFFFFFFFF);
      base::String xname(name);
      auto pos = xname.find(".prx");
      if (pos != base::String::npos)
        shared_objects_.push_back(xname.substr(0, pos));
    }
  }

  // Standard ELF has no dynamic tag for the symbol count; fall back to the SysV
  // hash chain count ([nbucket u32][nchain u32], nchain == #dynsyms).
  if (num_symbols_ == 0 && hashes_)
    num_symbols_ = reinterpret_cast<u32*>(hashes_)[1];

  if (kImplibTrace)
    BASE_LOGI("ps5dyn", "{} strtab={:p} sz={} syms={} rela={} jmp={} needed={}",
              info_.name.c_str(), (void*)strtab_.ptr,
              (unsigned long long)strtab_.size, num_symbols_, num_rela_,
              num_jmp_slots_, (unsigned long long)shared_objects_.size());
}

// DELTA_GUEST_BRK=<name>:<hex offset>[,...]: ud2 at a guest address once
// mapped, so the crash handler reports registers AT that instruction (the only
// way to see state feeding a fault inside a stripped module).
// DELTA_GUEST_NOEXEC=<addr>:<size>: <secs>: take X off a guest range; faults
// then happen at the ENTRY, not where the mapping ends.
static void StartNoExecWatch() {
  const char* spec = kNoExec;
  if (!spec)
    return;
  static const bool kOnce = ([spec] {
    const uintptr_t addr = std::strtoull(spec, nullptr, 16);
    const char *c1 = std::strchr(spec, ':');
    const size_t size = c1 ? std::strtoull(c1 + 1, nullptr, 16) : 0x1000;
    const char *c2 = c1 ? std::strchr(c1 + 1, ':') : nullptr;
    const u64 delay = c2 ? std::strtoull(c2 + 1, nullptr, 10) : 60;
    base::SpawnDetachedThread("module", [addr, size, delay] {
      base::SleepForMilliseconds((delay) * 1000);
      const int r = ::mprotect(reinterpret_cast<void *>(addr), size,
                               PROT_READ | PROT_WRITE);
      BASE_LOGI("noexec", "{:#x}+{:#x} -> rw ({})", (unsigned long long)addr,
                size, r);
    });
  }(), true);
  (void)kOnce;
}

void Module::PlantGuestBreakpoints() {
  StartNoExecWatch();
  const char* spec = kGuestBrk;
  if (!spec)
    return;
  for (base::String rest(spec); !rest.empty();) {
    const size_t comma = rest.find(',');
    base::String tok = rest.substr(0, comma);
    rest =
        comma == base::String::npos ? base::String() : rest.substr(comma + 1);
    const size_t colon = tok.find(':');
    if (colon == base::String::npos)
      continue;
    if (info_.name.find(tok.substr(0, colon)) == base::String::npos)
      continue;
    const u64 off = std::strtoull(tok.c_str() + colon + 1, nullptr, 16);
    if (off >= info_.code_size)
      continue;
    u8* at = GetAddress<u8>(off);
    // The segment is already protected by MapImage, so open the page first.
    const uintptr_t pg = reinterpret_cast<uintptr_t>(at) & ~uintptr_t(0x3FFF);
    ::mprotect(reinterpret_cast<void*>(pg), 0x8000,
               PROT_READ | PROT_WRITE | PROT_EXEC);
    // DELTA_GUEST_BRK_AFTER=<seconds>: arm the trap later instead of at load.
    // A site on a hot path traps on its first execution, which is rarely the
    // one being investigated; delaying past the earlier ones reaches it.
    if (const u64 delay = kBrkAfter) {
      const base::String name = info_.name;
      base::SpawnDetachedThread("module", [at, off, delay, name] {
        base::SleepForMilliseconds((delay) * 1000);
        at[0] = 0x0F;
        at[1] = 0x0B;  // ud2
        BASE_LOGI("guestbrk", "{} +{:#x} -> ud2 at {:p} (armed after {}s)",
                  name.c_str(), (unsigned long long)off, (void*)at,
                  (unsigned long long)delay);
      });
      continue;
    }
    at[0] = 0x0F;
    at[1] = 0x0B;  // ud2
    BASE_LOGI("guestbrk", "{} +{:#x} -> ud2 at {:p}", info_.name.c_str(),
              (unsigned long long)off, (void*)at);
  }
}

// DELTA_MODCHECK=<name>: watch a module's read-only segments for corruption; a
// moving digest means something scribbled on the image (libcohtml's V8
// snapshot).
void Module::StartModuleWatch() {
  const char* want = kModCheck;
  if (!want || info_.name.find(want) == base::String::npos)
    return;
  struct Range {
    const u8* addr;
    size_t size;
  };
  auto ranges = base::MakeShared<base::Vector<Range>>();
  for (u16 i = 0; i < elf_->phnum; ++i) {
    const auto* p = &segments_[i];
    if (p->type != PT_LOAD || (p->flags & PF_W) || !p->filesz)
      continue;
    const u8* a = elf_->type == ET_SCE_EXEC
                      ? reinterpret_cast<const u8*>(p->vaddr)
                      : GetAddress<const u8>(p->paddr);
    ranges->push_back({a, static_cast<size_t>(p->filesz)});
  }
  if (ranges->empty())
    return;
  const base::String name = info_.name;
  base::SpawnDetachedThread("module", [ranges, name] {
    base::Vector<u64> last(ranges->size(), 0);
    for (bool first = true;; first = false) {
      for (size_t i = 0; i < ranges->size(); i++) {
        u64 h = 1469598103934665603ull;
        const auto& r = (*ranges)[i];
        for (size_t k = 0; k < r.size; k += 64)
          h = (h ^ r.addr[k]) * 1099511628211ull;
        if (first) {
          BASE_LOGI("modcheck", "{} seg{} {:p}+{:#x} digest={:#x}",
                    name.c_str(), i, (const void*)r.addr, r.size,
                    (unsigned long long)h);
        } else if (h != last[i]) {
          BASE_LOGI("modcheck", "{} seg{} {:p}+{:#x} CHANGED {:#x} -> {:#x}",
                    name.c_str(), i, (const void*)r.addr, r.size,
                    (unsigned long long)last[i], (unsigned long long)h);
        }
        last[i] = h;
      }
      base::SleepForMilliseconds((2) * 1000);
    }
  });
}

bool Module::MapImage() {
  // size is the highest segment end, not the sum: segments map at their paddr,
  // which can be sparse, so summing under-reserves and a later segment ends up
  // writing into the unmapped part of the reservation.
  u64 code_size = 0;
  for (u16 i = 0; i < elf_->phnum; ++i) {
    const auto* p = &segments_[i];
    if (p->type == PT_LOAD || p->type == PT_SCE_RELRO) {
      u64 align = p->align ? p->align : 0x1000;
      u64 base = elf_->type == ET_SCE_EXEC ? p->vaddr : p->paddr;
      u64 end = base::Align<u64>(base + p->memsz, align);
      if (end > code_size)
        code_size = end;
    }
  }

  // could also check if INTERP exists
  if (code_size == 0)
    return false;

  // reserve a region from xxxxxxxx00000000 - xxxxxxxxFFFFFFFF
  constexpr size_t kOneMb = 1024ull * 1024ull;
  constexpr size_t kEightGb = 8ull * 1024ull * kOneMb;

  // A fixed (non-PIC) ET_SCE_EXEC embeds absolute references (segments at vaddr
  // 0x400000) and must map IN PLACE with zero bias (GetAddress(vaddr) ==
  // vaddr). Relocatable modules (ET_SCE_DYNEXEC / ET_SCE_DYNAMIC) get a
  // sequential high reservation. codeSize is the absolute image end for
  // ET_SCE_EXEC, else the image size.
  if (elf_->type == ET_SCE_EXEC) {
    u64 lo_vaddr = UINT64_MAX;
    for (u16 i = 0; i < elf_->phnum; ++i) {
      const auto* p = &segments_[i];
      if (p->type == PT_LOAD || p->type == PT_SCE_RELRO) {
        u64 align = p->align ? p->align : 0x1000;
        lo_vaddr = base::Min<u64>(lo_vaddr, p->vaddr & ~(align - 1));
      }
    }
    if (lo_vaddr == UINT64_MAX)
      lo_vaddr = 0;

    // The rip zone (x86 lifter scratch) trails the image. It is unused on the
    // FEX/aarch64 path but still reserved + filled so the layout matches.
    info_.rip_zone_size =
        base::Max<size_t>(info_.rip_zone_size, code_size - lo_vaddr);
    size_t span = (code_size - lo_vaddr) + info_.rip_zone_size;

    void* got = host_memory::AllocMem(reinterpret_cast<void*>(lo_vaddr), span,
                                      host_memory::PageProtection::kW,
                                      host_memory::AllocationType::kReserve);
    if (!got || reinterpret_cast<uintptr_t>(got) != lo_vaddr) {
      LOG_ERROR(
          "mapImage: could not reserve fixed-exec range at {:#x} (+{:#x})",
          lo_vaddr, span);
      return false;
    }
    host_memory::AllocMem(reinterpret_cast<void*>(lo_vaddr), span,
                          host_memory::PageProtection::kW,
                          host_memory::AllocationType::kCommit);

    info_.base = nullptr;  // zero load bias: image lives at its absolute vaddrs
    info_.code_size = code_size;
    info_.rip_zone = reinterpret_cast<u8*>(code_size);  // base(0) + codeSize

    std::memset(info_.rip_zone, 0xCC, info_.rip_zone_size);
    host_memory::ProtectMem(info_.rip_zone, info_.rip_zone_size,
                            host_memory::PageProtection::kRwx);
  } else {
    // ASLR off: fixed sequential bases so a guest crash reproduces at the same
    // address while the boot is being worked on; switch to kernel-chosen later.
#ifdef __ANDROID__
    // Android user VA is 39-bit; the x86 layout's 32 TiB base is unmappable.
    // Pack modules in tight 2 GiB slots from 64 GiB: above the GNM driver's
    // fixed PS4 regions (~63.5 GiB) and internal memory (8 GiB), below the
    // guest arena/FEX heap.
    constexpr size_t moduleSlot = 2ull * 1024ull * one_mb;   // 2 GiB
    static uintptr_t s_nextBase = 0x0000'0010'0000'0000ull;  // 64 GiB
#else
    constexpr size_t kModuleSlot = kEightGb;
    static uintptr_t s_next_base = 0x0000200000000000ull;
#endif
    info_.base = static_cast<u8*>(
        host_memory::AllocMem(reinterpret_cast<void*>(s_next_base), kModuleSlot,
                              host_memory::PageProtection::kW,
                              host_memory::AllocationType::kReserve));
    s_next_base += kModuleSlot;

    if (!info_.base)
      return false;

    // The lifter emits a per-fs-access stub into the rip-zone; linear-sweep
    // lifts the whole segment, so size the zone to the code (stubs ~32 B,
    // capped under the 8 GiB slot / rel32 reach), not the old fixed 5 KiB.
    info_.rip_zone_size = base::Max<size_t>(info_.rip_zone_size, code_size);

    // immediately take module memory + rip Zone memory
    host_memory::AllocMem(info_.base, code_size + info_.rip_zone_size,
                          host_memory::PageProtection::kW,
                          host_memory::AllocationType::kCommit);

    info_.code_size = code_size;
    info_.rip_zone = info_.base + code_size;

    std::memset(info_.rip_zone, 0xCC, info_.rip_zone_size);
    host_memory::ProtectMem(info_.rip_zone, info_.rip_zone_size,
                            host_memory::PageProtection::kRwx);
  }

  // map data
  for (u16 i = 0; i < elf_->phnum; i++) {
    const auto* s = &segments_[i];
    if (s->type == PT_LOAD || s->type == PT_SCE_RELRO) {
      void* target = elf_->type == ET_SCE_EXEC
                         ? reinterpret_cast<void*>(s->vaddr)
                         : GetAddress<void>(s->paddr);

      auto* seg = s->flags & PF_X ? &info_.text_seg : &info_.data_seg;
      seg->addr = static_cast<u8*>(target);
      seg->size = s->memsz;

      std::memcpy(target, GetOffset<void>(s->offset), s->filesz);
    }
  }

  // PS5 (Prospero) marks its code segments PF_X only (no PF_R), unlike PS4's
  // PF_R|PF_X. Both the lifter gate and the page-protection map must account
  // for that; gate the relaxation to PS5 so PS4 handling is unchanged.
  const bool ps5 = process_->GetPlatform() == Process::Platform::kPs5;

  // Lift code (x86 host only). The lifter rewrites syscall/int/fs reads
  // in place so raw guest x86-64 runs natively. On aarch64 the FEXCore JIT
  // handles all three, so the image is left byte-for-byte intact.
#if defined(DELTA_BACKEND_NATIVE)
  u8* rip_end = info_.base + code_size + info_.rip_zone_size;
  for (u16 i = 0; i < elf_->phnum; i++) {
    const auto* s = &segments_[i];
    u32 perm = s->flags & (PF_R | PF_W | PF_X);
    const bool exec = ps5 ? (perm & PF_X) != 0 : perm == (PF_R | PF_X);
    if (s->type == PT_LOAD && exec) {
      runtime::codeLift lift(info_.rip_zone, rip_end);
      LOG_ASSERT(lift.init());

      lift.transform(GetAddress<u8>(s->vaddr), s->filesz);
    }
  }
#endif

#if 1
  // temp hack: raise the 5.05 libkernel debug level. offset is fw-specific and
  // handle==1 is only libkernel on a real boot; bounds-check so a smaller
  // handle-1 image can't get written out of range.
  constexpr u32 kLibkernelDbgOff = 0x68264;
  if (kLibkDebug && info_.handle == 1 &&
      kLibkernelDbgOff + sizeof(u32) <= info_.code_size) {
    *GetAddress<u32>(kLibkernelDbgOff) = UINT32_MAX;
    LOG_WARNING("Enabling libkernel debug messages");
  }
#endif

  // apply page protections
  for (u16 i = 0; i < elf_->phnum; i++) {
    const auto* s = &segments_[i];
    if (s->type == PT_LOAD) {
      u32 perm = s->flags & (PF_R | PF_W | PF_X);
      auto trans_perm = [ps5](u32 op) {
        // PS5 code is PF_X only; map any executable segment rx (x86 has no
        // execute-without-read), writable rw, else r. PS4 handling unchanged.
        if (ps5) {
          if (op & PF_X)
            return host_memory::PageProtection::kRx;
          if (op & PF_W)
            return host_memory::PageProtection::kW;
          return host_memory::PageProtection::kR;
        }
        switch (op) {
          case (PF_R | PF_X):
            return host_memory::PageProtection::kRx;
          case (PF_R | PF_W):
            return host_memory::PageProtection::kW;
          case (PF_R):
            return host_memory::PageProtection::kR;
          default:
            return host_memory::PageProtection::kPriv;
        }
      };

      host_memory::ProtectMem(GetAddress<void>(s->vaddr), s->filesz,
                              trans_perm(perm));
    }
  }

  // Tell the backend the image is in place: native no-op (lifting done above),
  // FEX registers [base, base+codeSize) as an executable range for the JIT.
  cpu::GetBackend().OnImageMapped(info_);

  return true;
}

bool Module::SetupTls() {
  auto* p = GetSegment(PT_TLS);
  // Only modules with a real TLS template get an index; empty PT_TLS (memsz 0)
  // would inflate indices away from libkernel's dense numbering.
  if (p && p->memsz) {
    info_.tls_addr = GetAddress<u8>(p->vaddr);
    info_.tlsalign = p->align;
    info_.tls_size_file = p->filesz;
    info_.tls_size_mem = p->memsz;
    info_.tls_slot = process_->NextFreeTls();
  }

  return true;
}

static bool DecodeNid(const char* name, u64& lid, u64& mid) {
  // Obfuscated imports: "<11-char nid>#<libid>#<modid>", ids variable-length
  // base64 (two chars past 63), so the ids can't be read at fixed offsets.
  const char* h1 = std::strchr(name, '#');
  if (!h1)
    return false;
  const char* h2 = std::strchr(h1 + 1, '#');
  if (!h2 || h2 == h1 + 1 || !h2[1])
    return false;
  lid = 0;
  mid = 0;
  if (!runtime::decode_nid(h1 + 1, static_cast<size_t>(h2 - (h1 + 1)), lid))
    return false;
  if (!runtime::decode_nid(h2 + 1, std::strlen(h2 + 1), mid))
    return false;
  return true;
}

bool Module::ResolveObfSymbol(const char* name, uintptr_t& ptr_out) {
  // PS5: impLibs/impModules aren't populated, so resolve by global NID across
  // all loaded modules. LLE only: PS4 HLE stubs must not hijack a Prospero
  // import.
  if (ps5_layout_) {
    u64 hid = 0;
    if (!runtime::decode_nid(name, 11, hid))
      return false;
    // System libraries forced to HLE on PS5 because their LLE backend needs a
    // daemon we don't host: libSceVideoOut (port table never registers; real
    // Open returns 0x802900ff, null DCB crash), libSceUserService (spins on the
    // IPMI daemon), libScePad (reads pad-daemon state; HLE feeds SDL),
    // libSceSaveData (IPMI session; Skyrim's boot state machine spins at 100%
    // CPU), libSceIme/ libSceSystemService (abort or fail fatally). NIDs are
    // globally unique, so probing by name is safe; everything else (incl.
    // GnmDriver/AGC) stays LLE.
    static const char* const kPs5ForcedHle[] = {
        "libSceVideoOut",   "libSceUserService",   "libScePad",
        "libSceSaveData",   "libSceSystemService", "libSceIme",
        "libSceAppContent", "libSceVideodec2"};
    auto bind_hle = [&](const char* lib, uintptr_t hle) {
      char tn[64];
      std::snprintf(tn, sizeof(tn), "%s!%.11s", lib, name);
      ptr_out = cpu::MakeHostThunk(reinterpret_cast<void*>(hle), tn);
    };
    for (const char* lib : kPs5ForcedHle) {
      if (uintptr_t hle = runtime::vprx_get_forced(lib, hid)) {
        bind_hle(lib, hle);
        return true;
      }
    }
    // Bind to the module the import names: a title shipping SDK modules gets
    // the same NIDs from its own libc.prx and the firmware's
    // libSceLibcInternal, but only the named one has SceLibcMallocReplace
    // installed (else Skyrim's malloc lands in the firmware's 16 MiB arena
    // instead of the game's manager).
    u64 libid = 0, modid = 0;
    if (DecodeNid(name, libid, modid)) {
      for (auto& m : imp_modules_) {
        if (m.id != static_cast<i32>(modid) || !m.name)
          continue;
        if (auto named = process_->GetModule(base::StringRef(m.name)))
          if (uintptr_t a = named->GetExport(hid)) {
            ptr_out = a;
            return true;
          }
        break;
      }
    }
    for (auto& mod : process_->GetModuleList())
      if (uintptr_t a = mod->GetExport(hid)) {
        ptr_out = a;
        return true;
      }

    // Shims for exports a given firmware lacks (see vprx/ps5/*_ps5.cpp);
    // consulted only when no loaded module exports the NID. The seven AGC shims
    // are real exports from firmware 13.60 on; forcing them there would shadow
    // working code.
    static const char* const kPs5MissingExportShims[] = {
        "libkernel", "libSceAgcDriver", "libSceAgc", "libSceNgs2",
        "libSceFiber"};
    for (const char* lib : kPs5MissingExportShims) {
      if (uintptr_t hle = runtime::vprx_get_forced(lib, hid)) {
        bind_hle(lib, hle);
        return true;
      }
    }
    return false;
  }

  u64 libid = 0, modid = 0;
  if (!DecodeNid(name, libid, modid)) {
    // Not an obfuscated NID import (or a malformed one): let the caller route
    // it to the badcall stub instead of taking the whole process down.
    LOG_ERROR("resolveObfSymbol: can't decode symbol '{}'", name);
    return false;
  }

  const char* libname = nullptr;

  for (auto& mod : imp_libs_) {
    if (mod.id == static_cast<i32>(libid)) {
      libname = mod.name;
      break;
    }
  }

  if (!libname)
    return false;

  // HLE override: a registered vprx module wins over the loaded LLE module
  // (e.g. libSceVideoOut's .bss device table is never populated here).
  {
    u64 hid = 0;
    if (runtime::decode_nid(name, 11, hid)) {
      if (uintptr_t hle = runtime::vprx_get(libname, hid)) {
        // The HLE handler is a native host function; on FEX the guest can't
        // jump to it directly, so bind a guest trampoline. Native returns it
        // as-is.
        char tn[64];
        std::snprintf(tn, sizeof(tn), "%s!%.11s", libname, name);
        ptr_out = cpu::MakeHostThunk(reinterpret_cast<void*>(hle), tn);
        return true;
      }
    }
  }

  for (auto& mod : imp_modules_) {
    if (mod.id == static_cast<i32>(modid)) {
      auto xmod = process_->GetModule(mod.name);
      if (!xmod) {
        LOG_ERROR("resolveObfSymbol: Unknown module {} ({}) requestd", mod.name,
                  mod.id);
        return false;
      }

      char nameenc[12]{};  // name + null terminator
      std::strncpy(nameenc, name, 11);

      base::String long_name(nameenc);
      long_name += "#";
      long_name += libname;
      long_name += "#";
      long_name += mod.name;
      ptr_out = xmod->GetSymbolFullName(long_name.c_str());

      // libkernel forwards some exports to libkernel_sys under the name
      // "libkernel"; on a miss, search the other loaded modules for the bare
      // NID before badcall.
      if (!ptr_out) {
        for (auto& other : process_->GetModuleList()) {
          if (other.get() == xmod || other.get() == this)
            continue;
          if (uintptr_t a = other->GetSymbolByNid(nameenc)) {
            ptr_out = a;
            break;
          }
        }
      }
      return true;
    }
  }

  return false;
}

/*invoked by sys_dynlib_process_needed_and_relocate*/
bool Module::ResolveImports() {
  /*unpatched functioncall*/
  uintptr_t addr_bad_call = 0;
  if (auto kmod = process_->GetModule("libkernel"))
    addr_bad_call = kmod->GetSymbolFullName("M0z6Dr6TNnM#libkernel#libkernel");

  // A re-run only revisits the slots still sitting on the badcall stub: binding
  // an already-bound slot again would allocate a second host thunk for it, and
  // the miss has already been reported once.
  const bool retry = imports_bound_;
  imports_bound_ = true;
  unresolved_imports_ = 0;

  for (u32 i = 0; i < num_jmp_slots_; i++) {
    const auto* r = &jmpslots_[i];

    i32 type = ELF64_R_TYPE(r->info);
    i32 isym = ELF64_R_SYM(r->info);

    ElfSym* sym = &symbols_[isym];

    if (retry && *GetAddress<uintptr_t>(r->offset) != addr_bad_call)
      continue;

    if (type != R_X86_64_JUMP_SLOT) {
      LOG_WARNING("resolveImports: bad jump slot {}", i);
      continue;
    }

    if ((u32)isym >= num_symbols_ || sym->st_name >= strtab_.size) {
      LOG_WARNING("resolveImports: bad symbol index {} for relocation {}", isym,
                  i);
      continue;
    }

    i32 binding = ELF64_ST_BIND(sym->st_info);
    if (binding == STB_LOCAL) {
      *GetAddress<uintptr_t>(r->offset) =
          GetAddressNptr<uintptr_t>(sym->st_value);
      continue;
    }

    uintptr_t addr = 0;
    const char* name = &strtab_.ptr[sym->st_name];

    // DELTA_RELOC_TRACE: dump every PLT import (module, GOT offset, obfuscated
    // NID#lib#mod). Lets us pin which symbol a given GOT slot resolves to when
    // an LLE module calls an import we mis-emulate.
    if (kRelocTrace)
      BASE_LOGI("reloc", "{} jmpslot@{:#x} -> {}", info_.name.c_str(),
                (unsigned long)r->offset, name);

    // unresolved import (missing dep): point at the badcall stub, don't fail
    if (!ResolveObfSymbol(name, addr) || !addr) {
      addr = addr_bad_call;
      unresolved_imports_++;
      if (!retry)
        LOG_WARNING("unresolved import {} in {} (jmpslot@{:#x})", name,
                    info_.name.c_str(), r->offset);
    } else if (retry) {
      BASE_LOGI("reloc", "late-bound {} in {}", name, info_.name.c_str());
    }

    if (kRelocTrace)
      BASE_LOGI("reloc", "  {} @{:#x} resolved -> {:#x}", name,
                (unsigned long)r->offset, (unsigned long)addr);

    // DELTA_FIOS_TRACE: substitute a return-capturing guest wrapper for the
    // libSceFios2 whole-file APIs so the SotC world-container's
    // FHGetSize/FHRead can be traced on aarch64 (int3 hooks are x86-host-only).
    // No-op when unset.
    addr = probe::WrapImport(name, addr);

    *GetAddress<uintptr_t>(r->offset) = addr;
  }

  return true;
}

/*invoked by sys_dynlib_process_needed_and_relocate*/
bool Module::ApplyRelocations() {
  if (relocated_)
    return true;
  relocated_ = true;

  for (size_t i = 0; i < num_rela_; i++) {
    auto* r = &rela_[i];

    u32 isym = ELF64_R_SYM(r->info);
    i32 type = ELF64_R_TYPE(r->info);

    // check the index before indexing symbols[] below
    if (isym >= num_symbols_) {
      LOG_ERROR("Invalid symbol index {}", isym);
      continue;
    }

    ElfSym* sym = &symbols_[isym];
    i32 bind = ELF64_ST_BIND(sym->st_info);

    uintptr_t sym_val = 0;

    if (bind == STB_LOCAL)
      sym_val = sym->st_value;
    else if (bind == STB_GLOBAL || bind == STB_WEAK) {
      if (sym->st_value)
        sym_val = GetAddressNptr<uintptr_t>(sym->st_value);
      else {
        const char* name = &strtab_.ptr[sym->st_name];

        // unresolved import (missing dep): skip, leave the slot zeroed
        if (!ResolveObfSymbol(name, sym_val) || !sym_val)
          continue;
      }
    }

    switch (type) {
      case R_X86_64_64:
        *GetAddress<u64>(r->offset) = sym_val + r->addend;
        break;
      case R_X86_64_RELATIVE: /* base + ofs*/
        *GetAddress<i64>(r->offset) = GetAddressNptr<i64>(r->addend);
        break;
      case R_X86_64_GLOB_DAT:
        *GetAddress<u64>(r->offset) = sym_val;
        break;
      case R_X86_64_PC32:
        *GetAddress<u32>(r->offset) = static_cast<u32>(
            sym_val + r->addend - GetAddressNptr<u64>(r->offset));
        break;
      case R_X86_64_DTPMOD64:
        *GetAddress<u64>(r->offset) += info_.tls_slot;
        break;
      case R_X86_64_DTPOFF32:
        *GetAddress<u32>(r->offset) += static_cast<u32>(sym_val + r->addend);
        break;
      case R_X86_64_DTPOFF64:
        *GetAddress<u64>(r->offset) += sym_val + r->addend;
        break;
      case R_X86_64_NONE:
        break;
      default:
        continue;
    }
  }

  return true;
}

uintptr_t Module::GetSymbol(u64 nid) {
  // are there any overrides for me?
  auto imp = runtime::vprx_get(info_.name.c_str(), nid);
  if (imp != 0)
    return imp;

  for (u32 i = 0; i < num_symbols_; i++) {
    const auto* s = &symbols_[i];

    if (!s->st_value)
      continue;

    // if the symbol is exported
    // i32 binding = ELF64_ST_BIND(s->st_info);

    const char* name = &strtab_.ptr[s->st_name];

    u64 hid = 0;
    if (!runtime::decode_nid(name, 11, hid)) {
      LOG_ERROR("resolveExport: cant handle NID");
      return 0;
    }

    if (nid == hid) {
      return GetAddressNptr<uintptr_t>(s->st_value);
    }
  }

  return 0;
}

uintptr_t Module::GetExport(u64 nid) {
  for (u32 i = 0; i < num_symbols_; i++) {
    const auto* s = &symbols_[i];
    if (!s->st_value)
      continue;
    const char* name = &strtab_.ptr[s->st_name];
    u64 hid = 0;
    if (runtime::decode_nid(name, 11, hid) && nid == hid)
      return GetAddressNptr<uintptr_t>(s->st_value);
  }
  return 0;
}

uintptr_t Module::GetSymbolFullName(const char* name) {
  // no export hash table (module exports nothing)
  if (!hashes_ || !symbols_ || !strtab_.ptr)
    return 0;

  auto elf_hash = [](const char* name) {
    auto p = (const u8*)name;
    u32 h = 0;
    u32 g;
    while (*p != '\0') {
      h = (h << 4) + *p++;
      if ((g = h & 0xF0000000ull) != 0) {
        h ^= g >> 24;
      }
      h &= ~g;
    }
    return h;
  };

  auto hash = elf_hash(name);

  auto* htab = reinterpret_cast<u32*>(hashes_);
  u32 nbucket = htab[0];
  u32 nchain = htab[1];
  u32* bucket = &htab[2];
  u32* chain = &bucket[nbucket];

  /*char nameOut[11]{};
  runtime::encode_nid("module_start", reinterpret_cast<u8*>(&nameOut));*/

  for (u32 i = bucket[hash % nbucket]; i; i = chain[i]) {
    const auto* s = &symbols_[i];

    if (i > nchain)
      return 0;

    if (!s->st_value)
      continue;

    const char* sname = &strtab_.ptr[s->st_name];
    if (std::strncmp(sname, name, 11) == 0) {
      return GetAddressNptr<uintptr_t>(s->st_value);
    }
  }

  return 0;
}

uintptr_t Module::GetSymbol2(const char* name) {
  for (u32 i = 0; i < num_symbols_; i++) {
    const auto* s = &symbols_[i];

    if (!s->st_value)
      continue;

    const char* sname = &strtab_.ptr[s->st_name];

    if (std::strcmp(sname, name) == 0) {
      return GetAddressNptr<uintptr_t>(s->st_value);
    }
  }

  return 0;
}

uintptr_t Module::GetSymbolByNid(const char* nid) {
  for (u32 i = 0; i < num_symbols_; i++) {
    const auto* s = &symbols_[i];

    // exports are defined (st_value != 0); imports are undefined (== 0).
    if (!s->st_value || s->st_name >= strtab_.size)
      continue;

    const char* sname = &strtab_.ptr[s->st_name];
    if (std::strncmp(sname, nid, 11) == 0)
      return GetAddressNptr<uintptr_t>(s->st_value);
  }

  return 0;
}

// taken from idc's "uplift" project
void Module::InstallEhFrame() {
  const auto* p = GetSegment(PT_GNU_EH_FRAME);
  if (!p)
    return;  // no eh_frame_hdr segment
  if (p->filesz > p->memsz)
    return;

  info_.eh_frame_addr = GetAddress<u8>(p->vaddr);
  info_.eh_frame_size = p->memsz;

  // custom struct for eh_frame_hdr
  struct GnuExceptionInfo {
    u8 version;
    u8 encoding;
    u8 fde_count;
    u8 encoding_table;
    u8 first;
  };

  auto* exinfo = GetOffset<GnuExceptionInfo>(p->offset);

  if (exinfo->version != 1)
    return;

  u8* data_buffer = nullptr;
  u8* current = &exinfo->first;

  if (exinfo->encoding == 0x03)  // relative to base address
  {
    auto offset = *reinterpret_cast<u32*>(current);
    current += 4;

    data_buffer = (u8*)&info_.base[offset];
  } else if (exinfo->encoding == 0x1B)  // pc-relative
  {
    auto offset = *reinterpret_cast<i32*>(current);
    // pc-relative means relative to the field's position in the MAPPED image;
    // exinfo points into the on-disk buffer, so using it as the pc failed the
    // in-image check (why no module ever got an .eh_frame).
    const size_t field_off =
        static_cast<size_t>(current - reinterpret_cast<u8*>(exinfo));
    current += 4;
    data_buffer = GetAddress<u8>(p->vaddr) + field_off + offset;
  } else {
    return;
  }

  if (!data_buffer) {
    return;
  }

  // the FDE table sits in the mapped image. some modules (webkit/jsc) have an
  // eh_frame_hdr whose pointer doesn't walk to a clean terminator, so keep
  // every read inside the image and give up if we miss it. eh_frame is
  // optional.
  u8* const image_begin = info_.base;
  u8* const image_end = info_.base + info_.code_size;
  if (data_buffer < image_begin || data_buffer >= image_end)
    return;

  u8* data_buffer_end = data_buffer;
  bool terminated = false;
  while (data_buffer_end + sizeof(u32) <= image_end) {
    // CFI length is unsigned: 0 ends the table, 0xffffffff means a 64-bit len
    u32 len = *reinterpret_cast<u32*>(data_buffer_end);
    if (len == 0) {
      data_buffer_end += sizeof(u32);
      terminated = true;
      break;
    }

    size_t advance;
    if (len == 0xFFFFFFFFu) {
      if (data_buffer_end + 12 > image_end)
        break;
      advance = 12u + *reinterpret_cast<u64*>(data_buffer_end + 4);
    } else {
      advance = 4u + len;
    }

    // garbage length could overflow or stall the walk; bail if we'd not advance
    if (advance < sizeof(u32) || data_buffer_end + advance <= data_buffer_end)
      break;
    data_buffer_end += advance;
  }
  // A terminating zero-length CFI is optional and trailing encodings vary; none
  // of it changes where .eh_frame starts (the guest unwinder finds FDEs through
  // the header's table, not by walking). Requiring a terminator left
  // eh_frame_addr 0 for EVERY module, so a guest C++ throw hit std::terminate
  // (Minecraft world creation, inside libcohtml). Fall back to the rest of the
  // image.
  info_.eh_frameheader_addr = data_buffer;
  info_.eh_frameheader_size = static_cast<u32>(
      (terminated ? data_buffer_end : image_end) - data_buffer);
}

void Module::LogDbgInfo() {
  for (u16 i = 0; i < elf_->phnum; i++) {
    auto s = &segments_[i];
    switch (s->type) {
      case PT_SCE_COMMENT: {
        // this is similar to the windows pdb path
        auto* comment = GetOffset<SCEComment>(s->offset);

        base::String name;
        name.resize(comment->pathLength);
        memcpy(name.data(), GetOffset<void>(s->offset + sizeof(SCEComment)),
               comment->pathLength);

        LOG_INFO("Starting: {}", name.c_str());
        break;
      }
#if 0
			case PT_SCE_LIBVERSION:
			{
				u8* sec = GetOffset<u8>(s->offset);

				// count entries
				i32 index = 0;
				while (index <= s->filesz) {

					i8 cb = sec[index];

					// skip control byte
					index++;

					for (int i = index; i < (index + cb); i++)
					{
						if (sec[i] == 0x3A) {

							size_t length = i - index;

							base::String name;
							name.resize(length);
							memcpy(name.data(), &sec[index], length);

							u32 version = *(u32*)& sec[i + 1];
							u8* vptr = (u8*)& version;

							std::printf("lib <%s>, version %x.%x.%x.%x\n", name.c_str(), vptr[0], vptr[1], vptr[2], vptr[3]);
							break;
						}
					}

					// skip forward
					index += cb;
				}
				break;
			}
#endif
    }
  }
}
}  // namespace kern
