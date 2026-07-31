/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 /dev/dmem mapping: back each mmap with the shared physical-dmem store at the
 * requested physical offset (MAP_SHARED), so every VA that maps a given offset
 * aliases the same bytes -- the direct-memory coherency the AGC command buffers
 * rely on. Placed in the low (<2^40) guest aperture the GPU pointers reference.
 */

#include <base.h>
#include <cstdio>
#include <cstdlib>

#include <sys/mman.h>

#include "dma_dev.h"
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_mem.h"  // allocLowGuest, mFlags
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
}  // namespace

namespace krnl {

uint8_t *dmaDevicePs5::map(void *addr, size_t len, uint32_t /*prot*/, uint32_t flags,
                           size_t offset) {
  int fd = dmemBackingFd();
  if (fd < 0 || len == 0 ||
      static_cast<uint64_t>(offset) + len > dmemBackingSize())
    return reinterpret_cast<uint8_t *>(-1);
  uint8_t *va = static_cast<uint8_t *>(addr);
  const bool fixed = (flags & mFlags::fixed) != 0;
  void *p = MAP_FAILED;
  // A non-fixed hint is advisory: if the range is taken the host kernel picks an
  // address of its own, which is only page-aligned. Direct memory is 64 KiB
  // aligned on real hardware and titles rely on it -- Dead Cells' HashLink GC
  // fatals ("Page memory is not correctly aligned") on a 4 KiB-aligned page. So
  // probe the hint, and on a miss fall back to our own aperture rather than
  // whatever the kernel hands back.
  if (va && !fixed) {
    p = ::mmap(va, len, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_FIXED_NOREPLACE, fd, static_cast<off_t>(offset));
    if (p != MAP_FAILED && p != va) {
      ::munmap(p, len);
      p = MAP_FAILED;
    }
  }
  if (p == MAP_FAILED) {
    uint8_t *base = (va && fixed) ? va : allocLowGuest(len);
    if (!base)
      return reinterpret_cast<uint8_t *>(-1);
    p = ::mmap(base, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
  }
  if (p == MAP_FAILED)
    return reinterpret_cast<uint8_t *>(-1);
  if (kDmemTrace)
    std::fprintf(stderr, "[dmem] devmap off=%#zx len=%#zx -> %p (shared)\n", offset,
                 len, p);
  return reinterpret_cast<uint8_t *>(p);
}
}  // namespace krnl
