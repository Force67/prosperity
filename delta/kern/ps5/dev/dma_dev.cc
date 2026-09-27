/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 /dev/dmem mapping: back each mmap with the shared physical-dmem store at
 * the requested physical offset (MAP_SHARED), so every VA that maps a given
 * offset aliases the same bytes, the direct-memory coherency the AGC command
 * buffers rely on. Placed in the low (<2^40) guest aperture the GPU pointers
 * reference.
 */

#include <cstdio>
#include <cstdlib>
#include "base/arch.h"
#include "base/logging.h"

#include <sys/mman.h>
#include <cstring>

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "kern/guest_va_space.h"
#include "kern/lv2/sys_mem.h"  // AllocLowGuest, MFlags
#include "kern/process.h"
#include "kern/ps5/dev/dma_dev.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
DELTA_OPTION(bool, kNoCarry, "DELTA_DMEM_NOCARRY", false);
}  // namespace

// GPU aperture bridge (delta_gpu, gpu/ps5): direct memory is where a title's
// GPU-visible memory comes from, so the command processor learns the pools from
// here instead of assuming a band.
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_note_aperture(u64 base, u64 size);

namespace kern {

namespace {
// What each VA currently maps, so a re-map of direct memory the guest already
// has is distinguishable from a fresh one. Real dmem is recycled RAM: a remap
// gets whatever was in those pages, NOT zeroes. Our memfd backing reads zero at
// a fresh offset, so every such remap silently wiped what was there (V8 lost a
// just-deserialized read-only heap when it shrank the page holding it).
base::Mutex g_dmem_va_lock;
base::HashMap<u64, size_t> g_dmem_va_len;
}  // namespace

void ForgetDmemVa(u8* ptr, size_t size) {
  if (!ptr || !size)
    return;
  const u64 lo = reinterpret_cast<u64>(ptr), hi = lo + size;
  base::LockGuard<base::Mutex> lk(g_dmem_va_lock);
  for (auto it = g_dmem_va_len.begin(); it != g_dmem_va_len.end();) {
    if (it->first >= lo && it->first < hi)
      it = g_dmem_va_len.erase(it);
    else
      ++it;
  }
}

u8* DmaDevicePs5::Map(void* addr,
                      size_t len,
                      u32 /*prot*/,
                      u32 flags,
                      size_t offset) {
  int fd = DmemBackingFd();
  if (fd < 0 || len == 0 || static_cast<u64>(offset) + len > DmemBackingSize())
    return reinterpret_cast<u8*>(-1);
  u8* va = static_cast<u8*>(addr);
  const bool fixed = (flags & MFlags::kFixed) != 0;
  // A fixed map onto the exact base of an existing region is the guest
  // re-pointing its own region: carry the contents over, like reusing the same
  // physical pages. A map at a DIFFERENT base is a genuine new allocation and
  // must read fresh.
  base::Vector<u8> carry;
  const bool system_base =
      !fixed && reinterpret_cast<uintptr_t>(va) == 0xfe0000000ull;
  if (va && (fixed || system_base) && !kNoCarry) {
    base::LockGuard<base::Mutex> lk(g_dmem_va_lock);
    auto it = g_dmem_va_len.find(reinterpret_cast<u64>(va));
    // Carry as much as both mappings have: the system block is mapped twice,
    // 64 KiB by libSceGnmDriver and then 2 MiB by libSceAgcDriver, and the
    // second map must not take back the register tables the first one just
    // wrote at base+0xf000.
    if (it != g_dmem_va_len.end())
      carry.assign(va, va + base::Min<size_t>(len, it->second));
  }
  void* p = MAP_FAILED;
  // libSceAgcDriver maps a 2 MiB system block with a hint at 0xfe0000000 and
  // treats the placement as ABI (fetch shader table at base+0x40000 compared
  // against 0xfe0040000; a moved block fails sce::Agc::init with 0x8a6c002f).
  // The band is guest-owned space only our PROT_NONE placeholder occupies;
  // commit over it. libSceGnmDriver wants the SAME base (its .data starts
  // there; region 5 must cover 0xfe000f000..0xfe000f300). On hardware both
  // drivers share one system area, so grant the exact base to both; a hint that
  // merely OVERLAPS is still refused (a title's own allocation landing on it by
  // accident).
  constexpr uintptr_t kAgcSystemBase = 0xfe0000000ull;
  constexpr size_t kAgcSystemSize = 0x200000;
  const uintptr_t hint = reinterpret_cast<uintptr_t>(va);
  // Once libSceGnmDriver has taken the base, libSceAgcDriver stops hinting and
  // asks with no address at all (the kernel pins the system block wherever it
  // lives). Recognise that follow-up map and pin it too, or libSceAgc finds its
  // fetch-shader table away from the compiled-in 0xfe0040000.
  static bool s_gnm_took_base = false, s_agc_took_base = false;
  const bool agc_system_block =
      !fixed && hint == kAgcSystemBase && len <= kAgcSystemSize;
  const bool agc_system_follow_up = !fixed && !va && len == kAgcSystemSize &&
                                    s_gnm_took_base && !s_agc_took_base;
  if (agc_system_follow_up) {
    va = reinterpret_cast<u8*>(kAgcSystemBase);
    base::LockGuard<base::Mutex> lk(g_dmem_va_lock);
    auto it = g_dmem_va_len.find(kAgcSystemBase);
    if (it != g_dmem_va_len.end() && !kNoCarry)
      carry.assign(va, va + base::Min<size_t>(len, it->second));
  }
  const bool other_hint_into_agc_block =
      !fixed && !agc_system_block && hint < kAgcSystemBase + kAgcSystemSize &&
      hint + len > kAgcSystemBase;
  // A non-fixed hint is advisory: if the range is taken the host kernel picks
  // an address of its own, which is only page-aligned. Direct memory is 64 KiB
  // aligned on real hardware and titles rely on it: Dead Cells' HashLink GC
  // fatals ("Page memory is not correctly aligned") on a 4 KiB-aligned page. So
  // probe the hint, and on a miss fall back to our own aperture rather than
  // whatever the kernel hands back.
  if (agc_system_block || agc_system_follow_up) {
    p = ::mmap(va, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
    if (p != MAP_FAILED) {
      (len == kAgcSystemSize ? s_agc_took_base : s_gnm_took_base) = true;
    }
  } else if (va && !fixed && !other_hint_into_agc_block) {
    p = ::mmap(va, len, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_FIXED_NOREPLACE, fd,
               static_cast<off_t>(offset));
    if (p != MAP_FAILED && p != va) {
      ::munmap(p, len);
      p = MAP_FAILED;
    }
  }
  if (p == MAP_FAILED) {
    u8* base = (va && fixed) ? va : AllocLowGuest(len);
    if (!base)
      return reinterpret_cast<u8*>(-1);
    p = ::mmap(base, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
  }
  if (p == MAP_FAILED)
    return reinterpret_cast<u8*>(-1);
  NoteGuestTaken(static_cast<u8*>(p), len);
  prosperity_gpu_note_aperture(reinterpret_cast<u64>(p), len);
  if (!carry.empty())
    std::memcpy(p, carry.data(), carry.size());
  {
    base::LockGuard<base::Mutex> lk(g_dmem_va_lock);
    g_dmem_va_len[reinterpret_cast<u64>(p)] = len;
  }
  if (kDmemTrace)
    BASE_LOGI("dmem",
              "devmap off={:#x} len={:#x} want={:p} fixed={} -> {:p} (shared)",
              offset, len, addr, (int)fixed, p);
  return reinterpret_cast<u8*>(p);
}
}  // namespace kern
