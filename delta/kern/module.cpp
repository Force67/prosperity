
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <utl/file.h>
#include <utl/mem.h>

#if defined(DELTA_BACKEND_NATIVE)
#include "runtime/code_lift.h"
#endif
#include "runtime/vprx/vprx.h"
#include "cpu/cpu_backend.h"

#include <crypto/UnSELF.h>

#include "module.h"
#include "proc.h"
#include "vfs.h"

namespace krnl {
smodule::smodule(proc *process) : process(process) {
  /*-1 = no tls used*/
  info.handle = -1;
  info.tlsSlot = -1;

  /*set size from process config*/
  info.ripZoneSize = process->getEnv().ripZoneSize;
}

bool smodule::fromFile(const base::String &path) {
  utl::File file(path);
  if (!file.IsOpen()) {
    // missing dep on disk; fail soft so the caller can keep going
    LOG_ERROR("smodule: cannot open {}", path.c_str());
    return false;
  }

  ELFHeader diskHeader{};
  file.Read(diskHeader);

  if (diskHeader.magic == ELF_MAGIC &&
      diskHeader.machine == ELF_MACHINE_X86_64) {
    file.Seek(0, utl::seekMode::seek_set);

    auto sz = file.GetSize();
    data = base::MakeUnique<uint8_t[]>(static_cast<mem_size>(sz));
    file.Read(data.Get_UseOnlyIfYouKnowWhatYouareDoing(), sz);
    return fromMem(std::move(data));
  }

  // Not a raw x86-64 (SCE) ELF, e.g. still SELF-encrypted, or a different
  // arch. We don't decrypt here, so reject it.
  LOG_ERROR("smodule: {} is not a decrypted x86-64 ELF", path.c_str());
  return false;
}

bool smodule::fromVfs(const base::String &guestPath) {
  utl::File f = vfs::openRead(guestPath.c_str());
  if (!f.Exists()) {
    LOG_ERROR("smodule: cannot open vfs path {}", guestPath.c_str());
    return false;
  }

  auto sz = f.GetSize();
  if (sz < sizeof(ELFHeader))
    return false;

  base::Vector<uint8_t> raw;
  raw.resize(static_cast<mem_size>(sz));
  f.Read(raw.data(), sz);

  const uint8_t *src = raw.data();
  size_t srcSize = static_cast<size_t>(sz);

  // eboot.bin / prx in a pkg are fake SELFs: rebuild the plain ELF in memory.
  base::Vector<uint8_t> elfImg;
  uint32_t magic = static_cast<uint32_t>(src[0]) |
                   (static_cast<uint32_t>(src[1]) << 8) |
                   (static_cast<uint32_t>(src[2]) << 16) |
                   (static_cast<uint32_t>(src[3]) << 24);
  if (magic == SELF_MAGIC) {
    elfImg = crypto::self2elf(src, srcSize);
    if (elfImg.empty()) {
      LOG_ERROR("smodule: self2elf failed for {}", guestPath.c_str());
      return false;
    }
    src = elfImg.data();
    srcSize = elfImg.size();
  } else if (magic != ELF_MAGIC) {
    LOG_ERROR("smodule: {} is neither SELF nor ELF", guestPath.c_str());
    return false;
  }

  if (const char *dd = std::getenv("DELTA_DUMP_MODULE")) {
    if (guestPath.find(dd) != base::String::npos) {
      if (FILE *f = std::fopen("/tmp/dumped_module.elf", "wb")) {
        std::fwrite(src, 1, srcSize, f);
        std::fclose(f);
        LOG_INFO("dumped decrypted {} -> /tmp/dumped_module.elf ({} bytes)",
                 guestPath.c_str(), srcSize);
      }
    }
  }

  auto out = base::MakeUnique<uint8_t[]>(static_cast<mem_size>(srcSize));
  std::memcpy(out.Get_UseOnlyIfYouKnowWhatYouareDoing(), src, srcSize);
  return fromMem(std::move(out));
}

bool smodule::fromMem(base::UniquePointer<uint8_t[]> data) {
  /*TODO: figure out a way of getting rid of the back buffer*/
  this->data = std::move(data);

  elf = getOffset<ELFHeader>(0);
  segments = getOffset<ELFPgHeader>(elf->phoff);

  if (!mapImage()) {
    LOG_ERROR("smodule: Failed to map image");
    __debugbreak();
    return false;
  }

  digestDynamic();

#ifdef _DEBUG
  LOG_TRACE("mapped {} at {}", info.name.c_str(), fmt::ptr(info.base));
#endif
  setupTLS();

  if (!isDynlib()) {
    auto *seg = getSegment(ElfSegType::PT_SCE_PROCPARAM);
    if (seg) {
      info.procParam = getAddress<uint8_t>(seg->vaddr);
      info.procParamSize = seg->filesz;
    }
  } else {
    auto *seg = getSegment(ElfSegType::PT_SCE_MODULEPARAM);
    if (seg) {
      info.moduleParam = getAddress<uint8_t>(seg->vaddr);
      info.moduleParamSize = seg->filesz;
    }
  }

  installEHFrame();

  if (elf->entry == 0)
    info.entry = nullptr;
  else
    info.entry = getAddress<uint8_t>(elf->entry);

  for (auto &it : sharedObjects) {
      process->loadModule(it);
  }

  return true;
}

bool smodule::unload() {
  data = {};

  // todo: unload memory from VMA
  return true;
}

void smodule::digestDynamic() {
  const auto *dynS = getSegment(ElfSegType::PT_DYNAMIC);
  if (!dynS)
    return;
  const auto *dyldS = getSegment(ElfSegType::PT_SCE_DYNLIBDATA);

  // PS5 (Prospero) modules have no PT_SCE_DYNLIBDATA: their string/symbol/rela
  // tables are described by standard ELF dynamic tags as vaddrs into the mapped
  // image, not DT_SCE_* offsets into a data segment. Route them to the PS5-only
  // path and leave the PS4 handling below untouched.
  if (!dyldS) {
    ps5Layout = true;
    digestDynamicPs5(dynS);
    return;
  }

  uint8_t *dynldPtr = getOffset<uint8_t>(dyldS->offset);
  uint8_t *dynldAddr = getAddress<uint8_t>(dyldS->vaddr);
  // std::printf("addr = %p\n", dynldAddr);
  ELFDyn *dynamics = getOffset<ELFDyn>(dynS->offset);
  for (int32_t i = 0; i < (dynS->filesz / sizeof(ELFDyn)); i++) {
    auto *d = &dynamics[i];

    switch (d->tag) {
    case DT_SCE_HASH:
      hashes = reinterpret_cast<uint8_t *>(dynldPtr + d->un.value);
      break;
    case DT_INIT:
      info.initAddr = reinterpret_cast<uint8_t *>(dynldAddr + d->un.ptr);
      break;
    case DT_FINI:
      info.finiAddr = reinterpret_cast<uint8_t *>(dynldAddr + d->un.ptr);
      break;
    case DT_SCE_JMPREL:
      jmpslots = (ElfRel *)(dynldPtr + d->un.ptr);
      break;
    case DT_PLTRELSZ:
    case DT_SCE_PLTRELSZ:
      numJmpSlots = static_cast<uint32_t>(d->un.value / sizeof(ElfRel));
      break;
    case DT_SCE_STRTAB:
      strtab.ptr = (char *)(dynldPtr + d->un.ptr);
      break;
    case DT_STRSZ:
    case DT_SCE_STRSIZE:
      strtab.size = d->un.value;
      break;
    case DT_SCE_SYMTAB:
      symbols = reinterpret_cast<ElfSym *>(dynldPtr + d->un.ptr);
      break;
    case DT_SCE_SYMTABSZ:
      numSymbols = static_cast<uint32_t>(d->un.value / sizeof(ElfSym));
      break;
    case DT_SCE_RELA:
      rela = reinterpret_cast<ElfRel *>(dynldPtr + d->un.ptr);
      break;
    case DT_NEEDED: {
      auto name = (const char *)(strtab.ptr + (d->un.value & 0xFFFFFFFF));
      if (name) {
        base::String xname(name);
        /*quick but (valid?) hack for determining if an object is exported*/
        auto pos = xname.find(".prx");
        if (pos != base::String::npos) {
          sharedObjects.push_back(xname.substr(0, pos));
        }
      }

      break;
    }
    case DT_RELASZ:
    case DT_SCE_RELASZ:
      numRela = static_cast<uint32_t>(d->un.value / sizeof(ElfRel));
      break;
    case DT_SCE_EXPLIB:
    case DT_SCE_IMPLIB: {
      /*for now, in the future we also want to store explibs*/
      auto &e = impLibs.emplace_back();
      e.id = d->un.value >> 48;
      e.exported = d->tag == DT_SCE_EXPLIB;
      e.name = (const char *)(strtab.ptr + (d->un.value & 0xFFFFFFFF));
      break;
    }
    case DT_SCE_EXPORT_LIB_ATTR:
    case DT_SCE_IMPORT_LIB_ATTR: {
      uint16_t id = d->un.value >> 48;
      uint16_t idx = d->un.value & 0xFFF;

      for (auto &mod : impLibs) {
        if (mod.id == id) {
          mod.attr = idx;
          break;
        }
      }
      break;
    }
    case DT_SCE_NEEDED_MODULE: {
      auto &e = impModules.emplace_back();
      e.id = d->un.value >> 48;
      e.name = (const char *)(strtab.ptr + (d->un.value & 0xFFFFFFFF));
      break;
    }
    case DT_SCE_MODULE_ATTR: {
      uint16_t id = d->un.value >> 48;
      uint16_t idx = d->un.value & 0xFFF;

      for (auto &mod : impModules) {
        if (mod.id == id) {
          mod.attr = idx;
          break;
        }
      }
      break;
    }
    case DT_SCE_MODULEINFO:
      info.name = (const char *)(strtab.ptr + (d->un.value & 0xFFFFFFFF));
      break;
    case DT_SCE_FINGERPRINT:
      std::memcpy(info.fingerprint, getOffset<void>(d->un.value), 20);
      break;
    }
  }

  if (std::getenv("DELTA_IMPLIB_TRACE")) {
    for (auto &l : impLibs)
      std::printf("[implib] %s id=%u %s\n", info.name.c_str(), l.id,
                  l.name ? l.name : "?");
    for (auto &m : impModules)
      std::printf("[impmod] %s id=%u %s\n", info.name.c_str(), m.id,
                  m.name ? m.name : "?");
  }
}

// PS5-only dynamic parser. A Prospero module uses standard ELF dynamic tags:
// DT_STRTAB/SYMTAB/RELA/JMPREL are vaddrs into the mapped image (mapImage() has
// already run), unlike PS4 where DT_SCE_* tags are offsets into
// PT_SCE_DYNLIBDATA. The symbol/reloc format itself (Elf64_Sym, Elf64_Rela,
// x86-64 reloc types, NID#lib#mod mangling) is identical, so resolveImports()
// and applyRelocations() are reused unchanged.
void smodule::digestDynamicPs5(const ELFPgHeader *dynS) {
  ELFDyn *dynamics = getOffset<ELFDyn>(dynS->offset);
  const int count = static_cast<int>(dynS->filesz / sizeof(ELFDyn));

  for (int i = 0; i < count; i++) {
    auto *d = &dynamics[i];
    switch (d->tag) {
    case DT_STRTAB:
      strtab.ptr = getAddress<char>(d->un.ptr);
      break;
    case DT_STRSZ:
      strtab.size = d->un.value;
      break;
    case DT_SYMTAB:
      symbols = getAddress<ElfSym>(d->un.ptr);
      break;
    // PS5 keeps the SCE symbol-table size tag even with a standard DT_SYMTAB.
    case DT_SCE_SYMTABSZ:
      numSymbols = static_cast<uint32_t>(d->un.value / sizeof(ElfSym));
      break;
    case DT_RELA:
      rela = getAddress<ElfRel>(d->un.ptr);
      break;
    case DT_RELASZ:
      numRela = static_cast<uint32_t>(d->un.value / sizeof(ElfRel));
      break;
    case DT_JMPREL:
      jmpslots = getAddress<ElfRel>(d->un.ptr);
      break;
    case DT_PLTRELSZ:
      numJmpSlots = static_cast<uint32_t>(d->un.value / sizeof(ElfRel));
      break;
    case DT_HASH:
      hashes = getAddress<uint8_t>(d->un.ptr);
      break;
    case DT_INIT:
      info.initAddr = getAddress<uint8_t>(d->un.ptr);
      break;
    case DT_FINI:
      info.finiAddr = getAddress<uint8_t>(d->un.ptr);
      break;
    default:
      break;
    }
  }

  // DT_NEEDED entries usually precede DT_STRTAB, so resolve their names only
  // once the string table pointer is known.
  if (strtab.ptr) {
    for (int i = 0; i < count; i++) {
      auto *d = &dynamics[i];
      if (d->tag != DT_NEEDED)
        continue;
      const char *name = strtab.ptr + (d->un.value & 0xFFFFFFFF);
      base::String xname(name);
      auto pos = xname.find(".prx");
      if (pos != base::String::npos)
        sharedObjects.push_back(xname.substr(0, pos));
    }
  }

  // Standard ELF has no dynamic tag for the symbol count; fall back to the SysV
  // hash chain count ([nbucket u32][nchain u32], nchain == #dynsyms).
  if (numSymbols == 0 && hashes)
    numSymbols = reinterpret_cast<uint32_t *>(hashes)[1];

  if (std::getenv("DELTA_IMPLIB_TRACE"))
    std::printf("[ps5dyn] %s strtab=%p sz=%llu syms=%u rela=%u jmp=%u needed=%llu\n",
                info.name.c_str(), (void *)strtab.ptr,
                (unsigned long long)strtab.size, numSymbols, numRela, numJmpSlots,
                (unsigned long long)sharedObjects.size());
}

bool smodule::mapImage() {
  // size is the highest segment end, not the sum: segments map at their paddr,
  // which can be sparse, so summing under-reserves and a later segment ends up
  // writing into the unmapped part of the reservation.
  uint64_t codeSize = 0;
  for (uint16_t i = 0; i < elf->phnum; ++i) {
    const auto *p = &segments[i];
    if (p->type == PT_LOAD || p->type == PT_SCE_RELRO) {
      uint64_t align = p->align ? p->align : 0x1000;
      uint64_t base = elf->type == ET_SCE_EXEC ? p->vaddr : p->paddr;
      uint64_t end = align_up(base + p->memsz, align);
      if (end > codeSize)
        codeSize = end;
    }
  }

  // could also check if INTERP exists
  if (codeSize == 0)
    return false;

  // reserve a region from xxxxxxxx00000000 - xxxxxxxxFFFFFFFF
  constexpr size_t one_mb = 1024ull * 1024ull;
  constexpr size_t eight_gb = 8ull * 1024ull * one_mb;

  // A fixed (non-PIC) PS4 executable (ET_SCE_EXEC) links against its absolute
  // load addresses: its segments start at vaddr 0x400000 and it embeds absolute
  // references the loader cannot rebase. It must be mapped *in place*, with a
  // zero load bias so getAddress(vaddr) == vaddr. Relocatable modules
  // (ET_SCE_DYNEXEC main module, ET_SCE_DYNAMIC PRX) are position-independent
  // and get a sequential high reservation instead. codeSize above is already the
  // absolute image end for ET_SCE_EXEC (base = vaddr) and the image size for the
  // relocatable types (base = paddr, which starts near 0).
  if (elf->type == ET_SCE_EXEC) {
    uint64_t loVaddr = UINT64_MAX;
    for (uint16_t i = 0; i < elf->phnum; ++i) {
      const auto *p = &segments[i];
      if (p->type == PT_LOAD || p->type == PT_SCE_RELRO) {
        uint64_t align = p->align ? p->align : 0x1000;
        loVaddr = std::min<uint64_t>(loVaddr, p->vaddr & ~(align - 1));
      }
    }
    if (loVaddr == UINT64_MAX)
      loVaddr = 0;

    // The rip zone (x86 lifter scratch) trails the image. It is unused on the
    // FEX/aarch64 path but still reserved + filled so the layout matches.
    info.ripZoneSize = std::max<size_t>(info.ripZoneSize, codeSize - loVaddr);
    size_t span = (codeSize - loVaddr) + info.ripZoneSize;

    void *got = utl::allocMem(reinterpret_cast<void *>(loVaddr), span,
                              utl::pageProtection::w,
                              utl::allocationType::reserve);
    if (!got || reinterpret_cast<uintptr_t>(got) != loVaddr) {
      LOG_ERROR("mapImage: could not reserve fixed-exec range at {:#x} (+{:#x})",
                loVaddr, span);
      return false;
    }
    utl::allocMem(reinterpret_cast<void *>(loVaddr), span,
                  utl::pageProtection::w, utl::allocationType::commit);

    info.base = nullptr;  // zero load bias: image lives at its absolute vaddrs
    info.codeSize = codeSize;
    info.ripZone = reinterpret_cast<uint8_t *>(codeSize);  // base(0) + codeSize

    std::memset(info.ripZone, 0xCC, info.ripZoneSize);
    utl::protectMem(info.ripZone, info.ripZoneSize, utl::pageProtection::rwx);
  } else {
    // ASLR off: hand out fixed, sequential bases so a guest crash lands at the
    // same address every run and is easy to reproduce while the boot is being
    // worked on. Switch back to the nullptr (kernel-chosen) reservation below
    // once the boot is stable.
    // info.base = static_cast<uint8_t *>(utl::allocMem(
    //     nullptr, eight_gb, utl::pageProtection::w,
    //     utl::allocationType::reserve));
#ifdef __ANDROID__
    // Android user VA is 39-bit (~512 GiB); the x86 layout's 32 TiB base is
    // unmappable. Pack modules with a tight slot (modules are << 2 GiB, ripZone
    // is 5 KiB), based at 64 GiB: above the fixed PS4 guest regions the GNM
    // driver maps (SceGnm* at ~0xfe0000000 / 63.5 GiB) and
    // SceKernelInternalMemory at 8 GiB, and below the guest arena (lv2/sys_mem,
    // 256 GiB) and FEX heap.
    constexpr size_t moduleSlot = 2ull * 1024ull * one_mb;  // 2 GiB
    static uintptr_t s_nextBase = 0x0000'0010'0000'0000ull; // 64 GiB
#else
    constexpr size_t moduleSlot = eight_gb;
    static uintptr_t s_nextBase = 0x0000200000000000ull;
#endif
    info.base = static_cast<uint8_t *>(utl::allocMem(
        reinterpret_cast<void *>(s_nextBase), moduleSlot,
        utl::pageProtection::w, utl::allocationType::reserve));
    s_nextBase += moduleSlot;

    if (!info.base)
      return false;

    // The lifter emits a per-fs-access stub into the rip-zone. The linear-sweep
    // resync lifts the whole segment (not just the prefix before the first
    // rodata blob), so a large module can need far more than the old fixed 5
    // KiB. Size the zone to the code (a generous bound: stubs are ~32 B, fs
    // accesses are sparser than that), capped well under the 8 GiB module slot /
    // rel32 reach.
    info.ripZoneSize = std::max<size_t>(info.ripZoneSize, codeSize);

    // immediately take module memory + rip Zone memory
    utl::allocMem(info.base, codeSize + info.ripZoneSize, utl::pageProtection::w,
                  utl::allocationType::commit);

    info.codeSize = codeSize;
    info.ripZone = info.base + codeSize;

    std::memset(info.ripZone, 0xCC, info.ripZoneSize);
    utl::protectMem(info.ripZone, info.ripZoneSize, utl::pageProtection::rwx);
  }

  // step 0: map data
  for (uint16_t i = 0; i < elf->phnum; i++) {
    const auto *s = &segments[i];
    if (s->type == PT_LOAD || s->type == PT_SCE_RELRO) {
      void *target = elf->type == ET_SCE_EXEC
                         ? reinterpret_cast<void *>(s->vaddr)
                         : getAddress<void>(s->paddr);

      auto *seg = s->flags & PF_X ? &info.textSeg : &info.dataSeg;
      seg->addr = static_cast<uint8_t *>(target);
      seg->size = s->memsz;

      std::memcpy(target, getOffset<void>(s->offset), s->filesz);
    }
  }

  // PS5 (Prospero) marks its code segments PF_X only (no PF_R), unlike PS4's
  // PF_R|PF_X. Both the lifter gate and the page-protection map must account for
  // that; gate the relaxation to PS5 so PS4 handling is unchanged.
  const bool ps5 = process->getPlatform() == proc::platform::ps5;

  // step 1: lift code (x86 host only). The lifter rewrites syscall/int/fs reads
  // in place so raw guest x86-64 runs natively. On aarch64 the FEXCore JIT
  // handles all three, so the image is left byte-for-byte intact.
#if defined(DELTA_BACKEND_NATIVE)
  uint8_t *ripEnd = info.base + codeSize + info.ripZoneSize;
  for (uint16_t i = 0; i < elf->phnum; i++) {
    const auto *s = &segments[i];
    uint32_t perm = s->flags & (PF_R | PF_W | PF_X);
    const bool exec = ps5 ? (perm & PF_X) != 0 : perm == (PF_R | PF_X);
    if (s->type == PT_LOAD && exec) {
      runtime::codeLift lift(info.ripZone, ripEnd);
      LOG_ASSERT(lift.init());

      /*TODO: we should really introduce a cache here*/
      lift.transform(getAddress<uint8_t>(s->vaddr), s->filesz);
    }
  }
#endif

#if 1
  // temp hack: raise the 5.05 libkernel debug level. offset is fw-specific and
  // handle==1 is only libkernel on a real boot; bounds-check so a smaller
  // handle-1 image can't get written out of range.
  constexpr uint32_t kLibkernelDbgOff = 0x68264;
  if (std::getenv("DELTA_LIBK_DEBUG") && info.handle == 1 &&
      kLibkernelDbgOff + sizeof(uint32_t) <= info.codeSize) {
    *getAddress<uint32_t>(kLibkernelDbgOff) = UINT32_MAX;
    LOG_WARNING("Enabling libkernel debug messages");
  }
#endif

  // step 2: apply page protections
  for (uint16_t i = 0; i < elf->phnum; i++) {
    const auto *s = &segments[i];
    if (s->type == PT_LOAD) {
      uint32_t perm = s->flags & (PF_R | PF_W | PF_X);
      auto trans_perm = [ps5](uint32_t op) {
        // PS5 code is PF_X only; map any executable segment rx (x86 has no
        // execute-without-read), writable rw, else r. PS4 handling unchanged.
        if (ps5) {
          if (op & PF_X)
            return utl::pageProtection::rx;
          if (op & PF_W)
            return utl::pageProtection::w;
          return utl::pageProtection::r;
        }
        switch (op) {
        case (PF_R | PF_X):
          return utl::pageProtection::rx;
        case (PF_R | PF_W):
          return utl::pageProtection::w;
        case (PF_R):
          return utl::pageProtection::r;
        default:
          return utl::pageProtection::priv;
          /*todo: invalid parameter bugcheck*/
        }
      };

      utl::protectMem(getAddress<void>(s->vaddr), s->filesz, trans_perm(perm));
    }
  }

  // Tell the backend the image is in place: native no-op (lifting done above),
  // FEX registers [base, base+codeSize) as an executable range for the JIT.
  cpu::backend().onImageMapped(info);

  return true;
}

bool smodule::setupTLS() {
  auto *p = getSegment(PT_TLS);
  // Only modules with an actual TLS template get a module index. Many modules
  // ship an empty PT_TLS (memsz 0); handing those a slot inflates the indices
  // so they no longer match libkernel's own (dense) TLS-module numbering, and
  // __tls_get_addr then can't find a real module's block.
  if (p && p->memsz) {
    info.tlsAddr = getAddress<uint8_t>(p->vaddr);
    info.tlsalign = p->align;
    info.tlsSizeFile = p->filesz;
    info.tlsSizeMem = p->memsz;
    info.tlsSlot = process->nextFreeTLS();
  }

  return true;
}

static bool decodeNid(const char *name, uint64_t &lid, uint64_t &mid) {
  /*nid's are always 16 bytes long so we should get away
   with hard coding the access offset*/

  bool decodeSuccess = true;
  if (!runtime::decode_nid(&name[12], 1, lid))
    decodeSuccess = false;
  if (!runtime::decode_nid(&name[14], 1, mid))
    decodeSuccess = false;
  if (!decodeSuccess) {
    LOG_ERROR("decodeNid: can't decode symbol");
    return false;
  }
  return decodeSuccess;
}

bool smodule::resolveObfSymbol(const char *name, uintptr_t &ptrOut) {
  // PS5: the symbol's #lib#mod ids use different import metadata than PS4
  // (impLibs/impModules aren't populated), so resolve by the global NID - a
  // unique hash - across all loaded modules. LLE only: PS4 HLE stubs must not
  // hijack a Prospero import.
  if (ps5Layout) {
    uint64_t hid = 0;
    if (!runtime::decode_nid(name, 11, hid))
      return false;
    // A few system libraries must run HLE on PS5 because their LLE backend needs a
    // service daemon we don't host:
    //   - libSceVideoOut: its .bss port table never registers, so the real
    //     sceVideoOutOpen returns 0x802900ff and the renderer bails before creating
    //     its command buffers (null AGC DrawCommandBuffer crash).
    //   - libSceUserService: the real sceUserServiceInitialize spins allocating
    //     buffers forever waiting on the SceUserService IPMI daemon, stalling the
    //     engine's RenderInit before it ever submits GPU work.
    // NIDs are globally unique, so probing each forced-HLE table by name is safe
    // (a userService NID only ever matches the userService table). Everything else
    // (incl. libSceGnmDriver/AGC, which run LLE fine) stays LLE.
    static const char *const kPs5ForcedHle[] = {"libSceVideoOut",
                                                "libSceUserService"};
    for (const char *lib : kPs5ForcedHle) {
      if (uintptr_t hle = runtime::vprx_get_forced(lib, hid)) {
        char tn[64];
        std::snprintf(tn, sizeof(tn), "%s!%.11s", lib, name);
        ptrOut = cpu::makeHostThunk(reinterpret_cast<void *>(hle), tn);
        return true;
      }
    }
    for (auto &mod : process->getModuleList())
      if (uintptr_t a = mod->getExport(hid)) {
        ptrOut = a;
        return true;
      }
    return false;
  }

  uint64_t libid = 0, modid = 0;
  if (!decodeNid(name, libid, modid))
    __debugbreak();

  const char *libname = nullptr;

  // TODO: could be done nicer
  for (auto &mod : impLibs) {
    if (mod.id == static_cast<int32_t>(libid)) {
      libname = mod.name;
      break;
    }
  }

  if (!libname)
    return false;

  // HLE override: if a vprx module is registered for this library, it wins over
  // the loaded LLE module (e.g. libSceVideoOut, whose real .bss device table is
  // never populated in our env). The 11-char NID prefix decodes to the same hid
  // the HLE table is keyed on.
  {
    uint64_t hid = 0;
    if (runtime::decode_nid(name, 11, hid)) {
      if (uintptr_t hle = runtime::vprx_get(libname, hid)) {
        // The HLE handler is a native host function; on FEX the guest can't jump
        // to it directly, so bind a guest trampoline. Native returns it as-is.
        char tn[64];
        std::snprintf(tn, sizeof(tn), "%s!%.11s", libname, name);
        ptrOut = cpu::makeHostThunk(reinterpret_cast<void *>(hle), tn);
        return true;
      }
    }
  }

  for (auto &mod : impModules) {
    if (mod.id == static_cast<int32_t>(modid)) {
      auto xmod = process->getModule(mod.name);
      if (!xmod) {
        LOG_ERROR("resolveObfSymbol: Unknown module {} ({}) requestd", mod.name,
                  mod.id);
        return false;
      }

      char nameenc[12]{}; // name + null terminator
      std::strncpy(nameenc, name, 11);

      base::String longName(nameenc);
      longName += "#";
      longName += libname;
      longName += "#";
      longName += mod.name;
      ptrOut = xmod->getSymbolFullName(longName.c_str());

      // libkernel forwards a set of its exports to libkernel_sys; the import
      // still names "libkernel", so a miss there means we search the rest of
      // the loaded modules for the bare NID before falling back to the badcall
      // stub. (Fixes libSceSaveData's libkernel_sys memory-pool imports.)
      if (!ptrOut) {
        for (auto &other : process->getModuleList()) {
          if (other.get() == xmod || other.get() == this)
            continue;
          if (uintptr_t a = other->getSymbolByNid(nameenc)) {
            ptrOut = a;
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
bool smodule::resolveImports() {
  /*unpatched functioncall*/
  uintptr_t addrBadCall = 0;
  if (auto kmod = process->getModule("libkernel"))
    addrBadCall = kmod->getSymbolFullName("M0z6Dr6TNnM#libkernel#libkernel");

  for (uint32_t i = 0; i < numJmpSlots; i++) {
    const auto *r = &jmpslots[i];

    int32_t type = ELF64_R_TYPE(r->info);
    int32_t isym = ELF64_R_SYM(r->info);

    ElfSym *sym = &symbols[isym];

    if (type != R_X86_64_JUMP_SLOT) {
      LOG_WARNING("resolveImports: bad jump slot {}", i);
      continue;
    }

    if ((uint32_t)isym >= numSymbols || sym->st_name >= strtab.size) {
      LOG_WARNING("resolveImports: bad symbol index {} for relocation {}", isym,
                  i);
      continue;
    }

    int32_t binding = ELF64_ST_BIND(sym->st_info);
    if (binding == STB_LOCAL) {
      *getAddress<uintptr_t>(r->offset) =
          getAddressNPTR<uintptr_t>(sym->st_value);
      continue;
    }

    uintptr_t addr = 0;
    const char *name = &strtab.ptr[sym->st_name];

    // DELTA_RELOC_TRACE: dump every PLT import (module, GOT offset, obfuscated
    // NID#lib#mod). Lets us pin which symbol a given GOT slot resolves to when an
    // LLE module calls an import we mis-emulate.
    if (std::getenv("DELTA_RELOC_TRACE"))
      std::printf("[reloc] %s jmpslot@%#lx -> %s\n", info.name.c_str(),
                  (unsigned long)r->offset, name);

    // unresolved import (missing dep): point at the badcall stub, don't fail
    if (!resolveObfSymbol(name, addr) || !addr) {
      addr = addrBadCall;
      LOG_WARNING("unresolved import {} in {} (jmpslot@{:#x})", name,
                  info.name.c_str(), r->offset);
    }

    if (std::getenv("DELTA_RELOC_TRACE"))
      std::printf("[reloc]   %s @%#lx resolved -> %#lx\n", name,
                  (unsigned long)r->offset, (unsigned long)addr);

    *getAddress<uintptr_t>(r->offset) = addr;
  }

  return true;
}

/*invoked by sys_dynlib_process_needed_and_relocate*/
bool smodule::applyRelocations() {
  if (relocated)
    return true;
  relocated = true;

  for (size_t i = 0; i < numRela; i++) {
    auto *r = &rela[i];

    uint32_t isym = ELF64_R_SYM(r->info);
    int32_t type = ELF64_R_TYPE(r->info);

    // check the index before indexing symbols[] below
    if (isym >= numSymbols) {
      LOG_ERROR("Invalid symbol index {}", isym);
      continue;
    }

    ElfSym *sym = &symbols[isym];
    int32_t bind = ELF64_ST_BIND(sym->st_info);

    uint64_t symVal = 0;

    if (bind == STB_LOCAL)
      symVal = sym->st_value;
    else if (bind == STB_GLOBAL || bind == STB_WEAK) {
      /*relative offset*/ // TODO (force): should we check MID here?
      if (sym->st_value)
        symVal = getAddressNPTR<uintptr_t>(sym->st_value);
      else {
        const char *name = &strtab.ptr[sym->st_name];

        // unresolved import (missing dep): skip, leave the slot zeroed
        if (!resolveObfSymbol(name, symVal) || !symVal)
          continue;
      }
    }

    switch (type) {
    case R_X86_64_64:
      *getAddress<uint64_t>(r->offset) = symVal + r->addend;
      break;
    case R_X86_64_RELATIVE: /* base + ofs*/
      *getAddress<int64_t>(r->offset) = getAddressNPTR<int64_t>(r->addend);
      break;
    case R_X86_64_GLOB_DAT:
      *getAddress<uint64_t>(r->offset) = symVal;
      break;
    case R_X86_64_PC32:
      *getAddress<uint32_t>(r->offset) = static_cast<uint32_t>(
          symVal + r->addend - getAddressNPTR<uint64_t>(r->offset));
      break;
    case R_X86_64_DTPMOD64:
      *getAddress<uint64_t>(r->offset) += info.tlsSlot;
      break;
    case R_X86_64_DTPOFF32:
      *getAddress<uint32_t>(r->offset) +=
          static_cast<uint32_t>(symVal + r->addend);
      break;
    case R_X86_64_DTPOFF64:
      *getAddress<uint64_t>(r->offset) += symVal + r->addend;
      break;
    case R_X86_64_NONE:
      break;
    default:
      continue;
    }
  }

  return true;
}

uintptr_t smodule::getSymbol(uint64_t nid) {
  // are there any overrides for me?
  auto imp = runtime::vprx_get(info.name.c_str(), nid);
  if (imp != 0)
    return imp;

  for (uint32_t i = 0; i < numSymbols; i++) {
    const auto *s = &symbols[i];

    if (!s->st_value)
      continue;

    // if the symbol is exported
    // int32_t binding = ELF64_ST_BIND(s->st_info);

    const char *name = &strtab.ptr[s->st_name];

    uint64_t hid = 0;
    if (!runtime::decode_nid(name, 11, hid)) {
      LOG_ERROR("resolveExport: cant handle NID");
      return 0;
    }

    if (nid == hid) {
      return getAddressNPTR<uintptr_t>(s->st_value);
    }
  }

  return 0;
}

uintptr_t smodule::getExport(uint64_t nid) {
  for (uint32_t i = 0; i < numSymbols; i++) {
    const auto *s = &symbols[i];
    if (!s->st_value)
      continue;
    const char *name = &strtab.ptr[s->st_name];
    uint64_t hid = 0;
    if (runtime::decode_nid(name, 11, hid) && nid == hid)
      return getAddressNPTR<uintptr_t>(s->st_value);
  }
  return 0;
}

uintptr_t smodule::getSymbolFullName(const char *name) {
  // TODO: fix elf hash lookup

  // no export hash table (module exports nothing)
  if (!hashes || !symbols || !strtab.ptr)
    return 0;

  auto elfHash = [](const char *name) {
    auto p = (const uint8_t *)name;
    uint32_t h = 0;
    uint32_t g;
    while (*p != '\0') {
      h = (h << 4) + *p++;
      if ((g = h & 0xF0000000ull) != 0) {
        h ^= g >> 24;
      }
      h &= ~g;
    }
    return h;
  };

  auto hash = elfHash(name);

  auto *htab = reinterpret_cast<uint32_t *>(hashes);
  uint32_t nbucket = htab[0];
  uint32_t nchain = htab[1];
  uint32_t *bucket = &htab[2];
  uint32_t *chain = &bucket[nbucket];

  /*char nameOut[11]{};
  runtime::encode_nid("module_start", reinterpret_cast<uint8_t*>(&nameOut));*/

  for (uint32_t i = bucket[hash % nbucket]; i; i = chain[i]) {
    const auto *s = &symbols[i];

    if (i > nchain)
      return 0;

    if (!s->st_value)
      continue;

    const char *sname = &strtab.ptr[s->st_name];
    if (std::strncmp(sname, name, 11) == 0) {
      return getAddressNPTR<uintptr_t>(s->st_value);
    }
  }

  return 0;
}

uintptr_t smodule::getSymbol2(const char *name) {
  for (uint32_t i = 0; i < numSymbols; i++) {
    const auto *s = &symbols[i];

    if (!s->st_value)
      continue;

    const char *sname = &strtab.ptr[s->st_name];

    if (std::strcmp(sname, name) == 0) {
      return getAddressNPTR<uintptr_t>(s->st_value);
    }
  }

  return 0;
}

uintptr_t smodule::getSymbolByNid(const char *nid) {
  for (uint32_t i = 0; i < numSymbols; i++) {
    const auto *s = &symbols[i];

    // exports are defined (st_value != 0); imports are undefined (== 0).
    if (!s->st_value || s->st_name >= strtab.size)
      continue;

    const char *sname = &strtab.ptr[s->st_name];
    if (std::strncmp(sname, nid, 11) == 0)
      return getAddressNPTR<uintptr_t>(s->st_value);
  }

  return 0;
}

// taken from idc's "uplift" project
void smodule::installEHFrame() {
  const auto *p = getSegment(PT_GNU_EH_FRAME);
  if (!p)
    return;  // no eh_frame_hdr segment
  if (p->filesz > p->memsz)
    return;

  info.ehFrameAddr = getAddress<uint8_t>(p->vaddr);
  info.ehFrameSize = p->memsz;

  // custom struct for eh_frame_hdr
  struct GnuExceptionInfo {
    uint8_t version;
    uint8_t encoding;
    uint8_t fdeCount;
    uint8_t encodingTable;
    uint8_t first;
  };

  auto *exinfo = getOffset<GnuExceptionInfo>(p->offset);

  if (exinfo->version != 1)
    return;

  uint8_t *data_buffer = nullptr;
  uint8_t *current = &exinfo->first;

  if (exinfo->encoding == 0x03) // relative to base address
  {
    auto offset = *reinterpret_cast<uint32_t *>(current);
    current += 4;

    data_buffer = (uint8_t *)&info.base[offset];
  } else if (exinfo->encoding == 0x1B) // relative to eh_frame
  {
    auto offset = *reinterpret_cast<int32_t *>(current);
    current += 4;
    data_buffer = &current[offset];
  } else {
    return;
  }

  if (!data_buffer) {
    return;
  }

  // the FDE table sits in the mapped image. some modules (webkit/jsc) have an
  // eh_frame_hdr whose pointer doesn't walk to a clean terminator, so keep
  // every read inside the image and give up if we miss it. eh_frame is optional.
  uint8_t *const image_begin = info.base;
  uint8_t *const image_end = info.base + info.codeSize;
  if (data_buffer < image_begin || data_buffer >= image_end)
    return;

  uint8_t *data_buffer_end = data_buffer;
  bool terminated = false;
  while (data_buffer_end + sizeof(uint32_t) <= image_end) {
    // CFI length is unsigned: 0 ends the table, 0xffffffff means a 64-bit len
    uint32_t len = *reinterpret_cast<uint32_t *>(data_buffer_end);
    if (len == 0) {
      data_buffer_end += sizeof(uint32_t);
      terminated = true;
      break;
    }

    size_t advance;
    if (len == 0xFFFFFFFFu) {
      if (data_buffer_end + 12 > image_end)
        break;
      advance = 12u + *reinterpret_cast<uint64_t *>(data_buffer_end + 4);
    } else {
      advance = 4u + len;
    }

    // garbage length could overflow or stall the walk; bail if we'd not advance
    if (advance < sizeof(uint32_t) || data_buffer_end + advance <= data_buffer_end)
      break;
    data_buffer_end += advance;
  }
  if (!terminated)
    return;  // truncated / malformed unwind table

  size_t fde_count;
  if (exinfo->fdeCount == 0x03) // absolute
  {
    fde_count = *reinterpret_cast<uint32_t *>(current);
    current += 4;
  } else {
    return;
  }

  if (exinfo->encodingTable != 0x3B) // relative to eh_frame
  {
    return;
  }

  info.ehFrameheaderAddr = data_buffer;
  info.ehFrameheaderSize = data_buffer_end - data_buffer;
}

void smodule::logDbgInfo() {
  for (uint16_t i = 0; i < elf->phnum; i++) {
    auto s = &segments[i];
    switch (s->type) {
    case PT_SCE_COMMENT: {
      // this is similar to the windows pdb path
      auto *comment = getOffset<SCEComment>(s->offset);

      base::String name;
      name.resize(comment->pathLength);
      memcpy(name.data(), getOffset<void>(s->offset + sizeof(SCEComment)),
             comment->pathLength);

      LOG_INFO("Starting: {}", name.c_str());
      break;
    }
#if 0
			case PT_SCE_LIBVERSION:
			{
				uint8_t* sec = getOffset<uint8_t>(s->offset);

				// count entries
				int32_t index = 0;
				while (index <= s->filesz) {

					int8_t cb = sec[index];

					// skip control byte
					index++;

					for (int i = index; i < (index + cb); i++)
					{
						if (sec[i] == 0x3A) {

							size_t length = i - index;

							std::string name;
							name.resize(length);
							memcpy(name.data(), &sec[index], length);

							uint32_t version = *(uint32_t*)& sec[i + 1];
							uint8_t* vptr = (uint8_t*)& version;

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
} // namespace krnl
