/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 /dev/dmem mapping: back each mmap with the shared physical-dmem store at the
 * requested physical offset (MAP_SHARED), so every VA that maps a given offset
 * aliases the same bytes -- the direct-memory coherency the AGC command buffers
 * rely on. Placed in the low (<2^40) guest aperture the GPU pointers reference.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstdio>
#include <cstdlib>

#include <mutex>
#include <sys/mman.h>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "dma_dev.h"
#include "kern/guest_vaspace.h"
#include "kern/proc.h"
#include "kern/lv2/sys_mem.h"  // allocLowGuest, mFlags
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
DELTA_OPTION(bool, kNoCarry, "DELTA_DMEM_NOCARRY", false);
}  // namespace

// GPU aperture bridge (delta_gpu, gpu/ps5): direct memory is where a title's
// GPU-visible memory comes from, so the command processor learns the pools from
// here instead of assuming a band.
extern "C" void prosperity_gpu_note_aperture(u64 base, u64 size);

namespace krnl {

namespace {
// What each VA currently maps, so a map that lands on direct memory the guest
// already has can be told apart from a fresh one. Real direct memory is
// recycled physical RAM: a title that maps a new block where an old one was
// gets whatever was in those pages, NOT zeroes. Our backing store is a memfd,
// so a fresh offset reads zero and every such remap silently wipes what was
// there -- which is how V8 lost the read-only heap it had just deserialized
// when it shrank the page holding it.
std::mutex g_dmemVaLock;
std::unordered_map<u64, size_t> g_dmemVaLen;
}  // namespace

void forgetDmemVa(u8 *ptr, size_t size) {
  if (!ptr || !size)
    return;
  const u64 lo = reinterpret_cast<u64>(ptr), hi = lo + size;
  std::lock_guard<std::mutex> lk(g_dmemVaLock);
  for (auto it = g_dmemVaLen.begin(); it != g_dmemVaLen.end();) {
    if (it->first >= lo && it->first < hi)
      it = g_dmemVaLen.erase(it);
    else
      ++it;
  }
}

u8 *dmaDevicePs5::map(void *addr, size_t len, u32 /*prot*/, u32 flags,
                           size_t offset) {
  int fd = dmemBackingFd();
  if (fd < 0 || len == 0 ||
      static_cast<u64>(offset) + len > dmemBackingSize())
    return reinterpret_cast<u8 *>(-1);
  u8 *va = static_cast<u8 *>(addr);
  const bool fixed = (flags & mFlags::fixed) != 0;
  // A fixed map onto the exact base of one the guest already has is it
  // re-pointing its own region, not asking for fresh memory: carry the contents
  // over so the remap behaves like reusing the same physical pages. A map at a
  // DIFFERENT base is a genuine new allocation (a piece committed inside a pool,
  // say) and must still read as fresh memory.
  std::vector<u8> carry;
  const bool systemBase =
      !fixed && reinterpret_cast<uintptr_t>(va) == 0xfe0000000ull;
  if (va && (fixed || systemBase) && !kNoCarry) {
    std::lock_guard<std::mutex> lk(g_dmemVaLock);
    auto it = g_dmemVaLen.find(reinterpret_cast<u64>(va));
    // Carry as much as both mappings have: the system block is mapped twice,
    // 64 KiB by libSceGnmDriver and then 2 MiB by libSceAgcDriver, and the
    // second map must not take back the register tables the first one just
    // wrote at base+0xf000.
    if (it != g_dmemVaLen.end())
      carry.assign(va, va + std::min<size_t>(len, it->second));
  }
  void *p = MAP_FAILED;
  // libSceAgcDriver's init maps a 2 MiB system block with a plain hint at
  // 0xfe0000000, and libSceAgc treats the placement as ABI: it reads its fetch
  // shader table at base+0x40000 and compares that against a hardcoded
  // 0xfe0040000, printing "[Agc] FS Table offset has shifted" and failing
  // sce::Agc::init with 0x8a6c002f when the block moved. The band is guest-owned
  // address space that only our own PROT_NONE placeholder (guest_vaspace)
  // occupies, and the NOREPLACE probe below reads that as taken. Commit the
  // block over the placeholder.
  //
  // libSceGnmDriver wants the SAME base: its own .data starts at 0xfe0000000
  // and it maps a 64 KiB "SceGnmDriver" block there (an OUT-pointer map, so the
  // hint is the pre-set value), then checks that its region 5 -- base+0xf000 --
  // covers 0xfe000f000..0xfe000f300, which only holds at that base. On hardware
  // both drivers name one system area; refusing the second hint printed
  // "GnmDriver Initialization Error: Unsupported memory range" and then
  // "sce::Gnm::Initialize Error: Initialize Embedded Shader Fails", so a title
  // that keeps a Gnm path alongside AGC came up with no embedded shaders. Only
  // the exact base is granted; a hint that merely OVERLAPS the block is still
  // refused, because that is a title's own allocation landing on it by accident.
  constexpr uintptr_t kAgcSystemBase = 0xfe0000000ull;
  constexpr size_t kAgcSystemSize = 0x200000;
  const uintptr_t hint = reinterpret_cast<uintptr_t>(va);
  // ...and once libSceGnmDriver has taken the base, libSceAgcDriver stops
  // hinting it and asks for its 2 MiB with no address at all -- on a console
  // the kernel puts the system block at the one place it lives whatever the
  // caller asks for. Recognise that follow-up map and pin it too, or libSceAgc
  // finds its fetch-shader table away from the 0xfe0040000 it has compiled in
  // and calls it "not survivable".
  static bool s_gnmTookBase = false, s_agcTookBase = false;
  const bool agcSystemBlock =
      !fixed && hint == kAgcSystemBase && len <= kAgcSystemSize;
  const bool agcSystemFollowUp = !fixed && !va && len == kAgcSystemSize &&
                                 s_gnmTookBase && !s_agcTookBase;
  if (agcSystemFollowUp) {
    va = reinterpret_cast<u8 *>(kAgcSystemBase);
    std::lock_guard<std::mutex> lk(g_dmemVaLock);
    auto it = g_dmemVaLen.find(kAgcSystemBase);
    if (it != g_dmemVaLen.end() && !kNoCarry)
      carry.assign(va, va + std::min<size_t>(len, it->second));
  }
  const bool otherHintIntoAgcBlock =
      !fixed && !agcSystemBlock && hint < kAgcSystemBase + kAgcSystemSize &&
      hint + len > kAgcSystemBase;
  // A non-fixed hint is advisory: if the range is taken the host kernel picks an
  // address of its own, which is only page-aligned. Direct memory is 64 KiB
  // aligned on real hardware and titles rely on it -- Dead Cells' HashLink GC
  // fatals ("Page memory is not correctly aligned") on a 4 KiB-aligned page. So
  // probe the hint, and on a miss fall back to our own aperture rather than
  // whatever the kernel hands back.
  if (agcSystemBlock || agcSystemFollowUp) {
    p = ::mmap(va, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
    if (p != MAP_FAILED) {
      (len == kAgcSystemSize ? s_agcTookBase : s_gnmTookBase) = true;
    }
  } else if (va && !fixed && !otherHintIntoAgcBlock) {
    p = ::mmap(va, len, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_FIXED_NOREPLACE, fd, static_cast<off_t>(offset));
    if (p != MAP_FAILED && p != va) {
      ::munmap(p, len);
      p = MAP_FAILED;
    }
  }
  if (p == MAP_FAILED) {
    u8 *base = (va && fixed) ? va : allocLowGuest(len);
    if (!base)
      return reinterpret_cast<u8 *>(-1);
    p = ::mmap(base, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
  }
  if (p == MAP_FAILED)
    return reinterpret_cast<u8 *>(-1);
  noteGuestTaken(static_cast<u8 *>(p), len);
  prosperity_gpu_note_aperture(reinterpret_cast<u64>(p), len);
  if (!carry.empty())
    std::memcpy(p, carry.data(), carry.size());
  {
    std::lock_guard<std::mutex> lk(g_dmemVaLock);
    g_dmemVaLen[reinterpret_cast<u64>(p)] = len;
  }
  if (kDmemTrace)
    BASE_LOGI("dmem",
              "devmap off={:#x} len={:#x} want={:p} fixed={} -> {:p} (shared)",
              offset, len, addr, (int)fixed, p);
  return reinterpret_cast<u8 *>(p);
}
}  // namespace krnl
