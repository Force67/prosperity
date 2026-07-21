
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <base.h>
#include <logger/logger.h>
#include <utl/mem.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unordered_map>

#include "../../proc.h"
#include "error_table.h"
#include "sys_mem.h"

namespace krnl {
using ppt = utl::pageProtection;
using alt = utl::allocationType;

// PS4 user-space pointers must live below 2^40: libc's sceLibcMspaceCreate (and
// other allocators) reject a base whose bits >= 40 are set. The host kernel
// hands mmap(NULL) addresses far above that ceiling, so when we have to pick an
// address ourselves, carve it from a dedicated low arena instead. Bump-only and
// MAP_FIXED_NOREPLACE so we never clobber an existing mapping.
uint8_t *allocLowGuest(size_t size, size_t align) {
#ifdef __ANDROID__
  // 39-bit user VA: keep the guest arena clear of the module region (32..~224
  // GiB) and the FEX heap / bionic up top, and still under the PS4 2^40 ceiling.
  constexpr uintptr_t kFloor = 0x4000000000ull;   // 256 GiB
  constexpr uintptr_t kCeil = 0x6000000000ull;    // 384 GiB
#else
  // Start the arena at 512 GiB. Titles map their own fixed-address direct/flexible
  // memory pools at round 64 GiB slots (N * 0x10_0000_0000): Uncharted 2 uses
  // 0x10..0x12_0000_0000 (Onion/Garlic/Flexible); GTA:SA's Gameface engine
  // MAP_FIXEDs pools at 0x10/0x20/0x30/0x40_0000_0000, the last being a 128 MB
  // direct-memory pool exactly on our old 256 GiB floor -- it clobbered the
  // primary TCB (fs:0x10 -> 0), which crashed the first scePthreadMutexLock. Our
  // bookkeeping must sit above every slot a title fixed-maps; 512 GiB clears all
  // observed pools while staying under the PS4 2^40 user ceiling.
  constexpr uintptr_t kFloor = 0x8000000000ull;   // 512 GiB
  constexpr uintptr_t kCeil = 0x10000000000ull;   // 2^40, the PS4 user ceiling
#endif
  // Align bases to 64 KiB, not just the 16 KiB page: GNM tiled textures/render
  // targets carry alignment requirements above a page, and titles that allocate
  // a GPU pool here and sub-allocate surfaces from its base assert when the base
  // isn't aligned enough (DOOM's rhiTextureGnm buffer-block alignment check).
  constexpr uintptr_t kAlign = 0x10000;
  static std::atomic<uintptr_t> next{kFloor};
  size = (size + 0x3FFF) & ~uintptr_t(0x3FFF);
  // Caller-requested alignment (MAP_ALIGNED(n) in the mmap flags): the kernel
  // CONTRACTUALLY returns a base aligned to 2^n. Engines size their arena
  // bookkeeping around it -- SotC reserves its streaming arenas with
  // MAP_ALIGNED(20) and indexes them by VA>>20; a 64 KiB-aligned base breaks
  // every lookup (AllocationTracker null-record crash in LoadInitialWorld).
  const uintptr_t al = align > kAlign ? align : kAlign;
  for (int tries = 0; tries < 8192; tries++) {
    uintptr_t raw = next.load(std::memory_order_relaxed);
    uintptr_t base = (raw + (al - 1)) & ~(al - 1);  // align the base up
    if (base + size + 0x4000 > kCeil)
      return nullptr;  // doesn't fit; do NOT poison `next` (CAS, not fetch_add)
    if (!next.compare_exchange_weak(raw, base + size + 0x4000,
                                    std::memory_order_relaxed))
      continue;  // another thread advanced it; reload and retry
    void *p = ::mmap(reinterpret_cast<void *>(base), size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == reinterpret_cast<void *>(base)) {
      if (std::getenv("DELTA_ALLOC_TRACE"))
        std::printf("[lowalloc] %#lx +%#lx\n", (unsigned long)base,
                    (unsigned long)size);
      return static_cast<uint8_t *>(p);
    }
    if (p != MAP_FAILED)
      ::munmap(p, size);  // hint occupied; the CAS already skipped past it
  }
  return nullptr;
}

// POSIX shared memory (shm_open/shm_unlink/ftruncate + fd-backed mmap).
//
// A named shm object is sized with ftruncate then mmap'd to share a region
// between components. In a single guest process "shared" means the same name
// resolves to the same backing block, so every mapper sees one region. The block
// is a low (<2^40) guest allocation; ftruncate or the first mmap allocates it.
namespace {
struct shmBacking {
  uint8_t *base = nullptr;
  size_t size = 0;
};
std::mutex g_shmMutex;
std::unordered_map<std::string, shmBacking> g_shmByName;

class shmObject : public kObject {
public:
  shmObject(proc *p, std::string nm)
      : kObject(p, kObject::oType::shm), shmName(std::move(nm)) {}
  std::string shmName;  // key into g_shmByName
};

// Return the backing block for a shm, allocating/growing it to cover the
// requested range. Caller must NOT hold g_shmMutex. -1 on failure.
uint8_t *shmMap(shmObject *shm, size_t size, size_t offset) {
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto &b = g_shmByName[shm->shmName];
  size_t need = (offset + size + 0x3FFF) & ~size_t(0x3FFF);
  if (!b.base && need) {
    b.base = allocLowGuest(need);
    if (!b.base)
      return reinterpret_cast<uint8_t *>(-1);
    b.size = need;
    proc::getActive()->getVma().add(b.base, need, ppt::w);
  }
  if (!b.base || offset > b.size)
    return reinterpret_cast<uint8_t *>(-1);
  return b.base + offset;
}
}  // namespace

uint8_t *PS4ABI sys_mmap(void *addr, size_t size, uint32_t prot, uint32_t flags,
                         uint32_t fd, size_t offset) {
  auto *proc = proc::getActive();
  if (!proc)
    return reinterpret_cast<uint8_t *>(-1);

  if (flags & mFlags::stack || flags & mFlags::noextend)
    flags |= mFlags::anon;

  /*align the page*/
  size = (size + 0x3FFF) & 0xFFFFFFFFFFFFC000LL;

  // A zero-length mapping is invalid (BSD returns EINVAL). Guests hit this on an
  // error-recovery path, e.g. mmap()ing an fd from a failed physhm_open/fstat.
  // Without this it fell into allocLowGuest(0) -- 8192 failing mmap(len=0) host
  // calls -- and returned (uint8_t*)-1, which the errno convention reports as
  // EPERM (1) rather than EINVAL (22), misleading the guest's fallback.
  if (size == 0)
    return reinterpret_cast<uint8_t *>(-SysError::eINVAL);

  // SCOUT (DELTA_MMAP_CALLER=<minMB>): scan the guest stack for return addresses
  // in a loaded module's .text to pin which guest code requested a big map (e.g.
  // the libc heap). Handler runs on the guest stack on native.
  if (const char *mc = std::getenv("DELTA_MMAP_CALLER")) {
    size_t minB = static_cast<size_t>(std::strtoull(mc, nullptr, 0)) * 1024 * 1024;
    if (minB == 0) minB = 64ull * 1024 * 1024;
    if (size >= minB) {
      std::fprintf(stderr, "[mmap-caller] size=%#zx prot=%#x flags=%#x fd=%u:\n",
                   size, prot, flags, fd);
      auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
      int shown = 0;
      for (int i = 0; i < 8192 && shown < 12; i++) {
        uintptr_t v = sp[i];
        for (auto &m : proc->getModuleList()) {
          auto &mi = m->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && v >= (uintptr_t)t && v < (uintptr_t)t + mi.textSeg.size) {
            std::fprintf(stderr, "  sp+%-4x %s+%#lx\n", i * 8, mi.name.c_str(),
                         v - (uintptr_t)t);
            shown++;
            break;
          }
        }
      }
    }
  }

  // addr is a hint unless MAP_FIXED: relocate it rather than alias an existing map
  if (!(flags & mFlags::fixed)) {
    if (!addr || proc->getVma().overlaps(static_cast<uint8_t *>(addr), size))
      addr = nullptr;
  }

  if (fd != -1) {
    auto *obj = proc->getObjTable().get(fd);
    if (std::getenv("DELTA_MMAPFD_TRACE"))
      std::fprintf(stderr, "[mmapfd] fd=%u addr=%p size=%#zx off=%#zx objType=%d\n",
                   fd, addr, size, offset, obj ? (int)obj->type() : -1);
    if (obj && obj->type() == kObject::oType::shm) {
      // POSIX shared memory: hand back the shared backing so every mapper of
      // this shm sees the same region (sized by ftruncate).
      return shmMap(static_cast<shmObject *>(obj), size, offset);
    }
    if (obj) {
      // Device-backed mmap (e.g. /dev/dce's scanout pool): use the region the
      // device hands back instead of an anonymous fallback, so the guest maps
      // the device's real memory. -1 means "not device-backed"; fall through.
      auto *m = static_cast<device *>(obj)->map(addr, size, prot, flags, offset);
      if (m != reinterpret_cast<uint8_t *>(-1)) {
        proc->getVma().add(
            m, size, static_cast<ppt>(prot & static_cast<uint32_t>(ppt::rwx)),
            prot);
        return m;
      }
    }
  }

  // MAP_ALIGNED(n): bits 31..24 of the flags carry log2 of a base alignment the
  // kernel must honor (FreeBSD 9 semantics; Sony titles rely on it -- SotC
  // reserves streaming arenas with MAP_ALIGNED(20) and keys its allocator
  // bookkeeping on the 1 MiB-aligned base).
  const uint32_t alignLog = (flags >> 24) & 0x1F;
  const size_t mapAlign = (alignLog >= 14 && alignLog < 40)
                              ? (size_t(1) << alignLog)
                              : 0;
  if (mapAlign && addr && !(flags & mFlags::fixed) &&
      (reinterpret_cast<uintptr_t>(addr) & (mapAlign - 1)))
    addr = nullptr;  // misaligned hint: pick our own aligned base instead

  void *ptr = nullptr;
  if (addr) {
    if (flags & mFlags::fixed) {
      // MAP_FIXED: the guest demands this exact address; overlay whatever's there.
      ptr = utl::allocMem(addr, size, ppt::w, alt::reservecommit);
      if (!ptr)
        ptr = utl::allocMem(addr, size, ppt::w, alt::commit);  // maybe pre-reserved
    } else if (utl::allocMem(addr, size, ppt::w, alt::reserve)) {
      // A hint must never alias an existing mapping. reservecommit uses MAP_FIXED
      // and would clobber it, so probe with a NOREPLACE reserve first and only
      // commit if the address was free; otherwise fall through to the low arena.
      // Without this a guest TLS/TCB hint lands on and destroys a loaded module
      // (seen on Android, where the guest hints into the low module region).
      ptr = utl::allocMem(addr, size, ppt::w, alt::commit);
    }
  }
  // No usable hint (or it was taken): pick a low (<2^40) address the guest's
  // own allocators will accept, not whatever high address the host hands out.
  if (!ptr)
    ptr = allocLowGuest(size, mapAlign);
  if (!ptr) {
    return reinterpret_cast<uint8_t *>(-1);
  }

  // Track the prot the guest actually asked for (BSD r=1/w=2/x=4 maps 1:1 onto
  // pageProtection) so sceKernelVirtualQuery / QueryMemoryProtection report the
  // truth instead of a blanket rwx. The host pages stay rwx: FEX reads guest
  // memory directly and we don't deliver protection faults, so restricting them
  // would only risk spurious crashes, not faithful behaviour.
  auto gprot = static_cast<ppt>(prot & static_cast<uint32_t>(ppt::rwx));

  if (flags & mFlags::anon)
    std::memset(ptr, 0, size);

  // File-backed mmap: copy the file's content into the freshly-mapped pages so the
  // guest reads the file it mapped (Doom64 mmaps its asset/WAD files and samples
  // textures straight out of the mapping; an anonymous zero-fill left them black).
  // readAt is a no-op (-1) for non-file devices; the read stops at EOF so a sparse
  // over-sized mapping keeps zeros past the file's end.
  if (fd != static_cast<uint32_t>(-1)) {
    if (auto *o = proc->getObjTable().get(fd))
      if (o->type() == kObject::oType::device) {
        int64_t got = static_cast<device *>(o)->readAt(ptr, size, offset);
        if (got > 0 && std::getenv("DELTA_MMAPFD_TRACE"))
          std::fprintf(stderr, "[mmapfd]   filled %p from fd=%u off=%#zx -> %lld bytes\n",
                       ptr, fd, offset, (long long)got);
      }
  }

  // 0x100 = Sony MAP_VOID: an address-space reservation (titles later commit
  // pieces inside with MAP_FIXED, which punches the reservation apart in the
  // VMA). Virtual query must see it as reserved, not committed memory.
  const bool voidReserve = (flags & 0x100) && prot == 0;
  proc->getVma().add(static_cast<uint8_t *>(ptr), size, gprot, prot,
                     voidReserve);

  utl::protectMem(static_cast<void *>(ptr), size, ppt::rwx);

  std::printf("mmap %p, %x, prot=%x flags=%x, %p -> %p\n", addr, size, prot,
              flags, _ReturnAddress(), ptr);
  // LOG_WARNING("addr={}, len={}, requested by {}", fmt::ptr(addr), len,
  // fmt::ptr(_ReturnAddress()));

  if (flags & mFlags::stack)
    return &static_cast<uint8_t *>(ptr)[size];

  return static_cast<uint8_t *>(ptr);
}

int PS4ABI sys_mprotect(uint8_t *addr, size_t len, int prot) {
  auto *proc = proc::getActive();
  if (!proc)
    return -SysError::eINVAL;

  // BSD prot bits (r=1/w=2/x=4) map 1:1 onto pageProtection. Reflect the change
  // in the region we track so sceKernelVirtualQuery reports the new protection.
  // We don't restrict the host pages (see sys_mmap) and we don't fail on an
  // untracked range: the dynamic linker mprotects its own RELRO segments, which
  // the module loader maps outside this table, and erroring there would break
  // relocation finalisation. So update what we know and report success.
  auto np = static_cast<ppt>(prot & static_cast<uint32_t>(ppt::rwx));
  if (auto *region = proc->getVma().get(addr))
    region->prot = np;
  (void)len;
  return 0;
}

int PS4ABI sys_shm_open(const char *path, uint32_t flags, uint16_t mode) {
  auto *proc = proc::getActive();
  if (!proc || !path)
    return -SysError::eINVAL;

  constexpr uint32_t kO_CREAT = 0x0200, kO_EXCL = 0x0800;
  std::string name(path);
  {
    std::lock_guard<std::mutex> lk(g_shmMutex);
    auto it = g_shmByName.find(name);
    if (it == g_shmByName.end()) {
      if (!(flags & kO_CREAT) && std::getenv("DELTA_SHM_NOAUTO")) {
        // DIAGNOSTIC: restore the pre-LLE behaviour (fail an open of a system shm
        // the guest didn't create) to test whether auto-providing it makes a title
        // block waiting for a ShellCore handshake that never arrives (Doom64).
        std::fprintf(stderr, "[shm_open] NOAUTO: '%s' -> ENOENT\n", name.c_str());
        return -SysError::eNOENT;
      }
      if (!(flags & kO_CREAT)) {
        // A read-only open of a shm that wasn't created by the guest: this is a
        // SYSTEM shared region the kernel would have published at boot (e.g.
        // libSceAvSetting's audio/video settings block). We don't model its
        // contents, so auto-provide a zeroed, pre-sized backing -- the title
        // then fstat()s a real size and mmaps it (reading defaults) instead of
        // failing init with a -ENOENT shm fd it tries to map anyway.
        shmBacking b;
        b.size = 0x10000;  // 64 KiB, ample for a settings block
        b.base = allocLowGuest(b.size);
        if (b.base) {
          std::memset(b.base, 0, b.size);
          proc->getVma().add(b.base, b.size, ppt::w);
        }
        std::fprintf(stderr, "[shm_open] auto-provide system shm '%s' size=%#zx\n",
                     name.c_str(), b.size);
        g_shmByName.emplace(name, b);
      } else {
        g_shmByName.emplace(name, shmBacking{});  // empty; sized later by ftruncate
      }
    } else if ((flags & kO_CREAT) && (flags & kO_EXCL)) {
      return -SysError::eEXIST;
    }
  }

  // A fresh fd per open, all sharing the named backing (POSIX-ish for a single
  // guest process). The ctor registers it in the object table.
  auto *obj = new shmObject(proc, std::move(name));
  std::fprintf(stderr, "[shm_open] '%s' flags=%#x -> fd=%u\n", path, flags,
               obj->handle());
  return obj->handle();
}

int PS4ABI sys_shm_unlink(const char *path) {
  if (!path)
    return -SysError::eINVAL;
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto it = g_shmByName.find(path);
  if (it == g_shmByName.end())
    return -SysError::eNOENT;
  // Drop the name only. Any region already handed to mmap is a raw pointer that
  // stays valid; we keep the host allocation (reclaimed at process exit).
  g_shmByName.erase(it);
  return 0;
}

// Report a shm fd's backing size for fstat (shm objects aren't device-backed,
// so sys_fstat's fdToDevice path can't size them). Returns SIZE_MAX if `fd`
// isn't a shm, so the caller falls through to the normal path.
size_t shmFstatSize(uint32_t fd) {
  auto *proc = proc::getActive();
  if (!proc)
    return SIZE_MAX;
  auto *obj = proc->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::shm)
    return SIZE_MAX;
  auto *shm = static_cast<shmObject *>(obj);
  std::lock_guard<std::mutex> lk(g_shmMutex);
  return g_shmByName[shm->shmName].size;
}

int PS4ABI sys_ftruncate(uint32_t fd, int64_t length) {
  auto *proc = proc::getActive();
  if (!proc || length < 0)
    return -SysError::eINVAL;
  auto *obj = proc->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::shm)
    return -SysError::eBADF;

  auto *shm = static_cast<shmObject *>(obj);
  size_t want = (static_cast<size_t>(length) + 0x3FFF) & ~size_t(0x3FFF);
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto &b = g_shmByName[shm->shmName];
  if (want == 0 || want <= b.size)
    return 0;
  uint8_t *nb = allocLowGuest(want);
  if (!nb)
    return -SysError::eNOMEM;
  if (b.base)
    std::memcpy(nb, b.base, b.size);  // grow before first mmap: preserve contents
  b.base = nb;
  b.size = want;
  proc->getVma().add(b.base, want, ppt::w);
  return 0;
}

int PS4ABI sys_mname(uint8_t *ptr, size_t len, const char *name, void *) {
  auto *proc = proc::getActive();
  if (!proc)
    return -1;

  auto *info = proc->getVma().get(ptr);
  if (!info) {
    LOG_WARNING("attempted to tag unknown memory ({}, {})", fmt::ptr(ptr),
                name);
    return -1;
  }

  LOG_WARNING("tagged {} with name {}", fmt::ptr(ptr), name);
  info->name = name;
  return 0;
}

struct mdbg_property {
  int32_t unk;
  int32_t unk2;
  void *addr;
  size_t areaSize;
  int64_t unk3;
  int64_t unk4;
  char name[32];
};

static_assert(sizeof(mdbg_property) == 72);

int PS4ABI sys_mdbg_service(uint32_t op, void *arg1, void *arg2, void *a3) {
  switch (op) {
  case 1: {
    auto *info = static_cast<mdbg_property *>(arg1);
    LOG_WARNING("set property {} for addr {} with size {}", info->name,
                info->addr, info->areaSize);
    /*TODO: create named object*/

    break;
  }
  }

  return 0;
}

// sceKernelGet/SetDirectMemoryContainer: -1 queries the current container id,
// any other value selects it and returns the previous one. We don't enforce
// separate dmem pools, so just track the selected id (default 0) and never trap.
int PS4ABI sys_dmem_container(uint32_t op) {
  static std::atomic<uint32_t> current{0};
  if (op == 0xFFFFFFFFu)
    return static_cast<int>(current.load());
  return static_cast<int>(current.exchange(op));
}
} // namespace krnl