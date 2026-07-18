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

namespace krnl {

uint8_t *dmaDevicePs5::map(void *addr, size_t len, uint32_t /*prot*/, uint32_t flags,
                           size_t offset) {
  int fd = dmemBackingFd();
  if (fd < 0 || len == 0 ||
      static_cast<uint64_t>(offset) + len > dmemBackingSize())
    return reinterpret_cast<uint8_t *>(-1);
  uint8_t *va = static_cast<uint8_t *>(addr);
  bool fixed = (flags & mFlags::fixed) != 0;
  if (!va) {
    va = allocLowGuest(len);
    if (!va)
      return reinterpret_cast<uint8_t *>(-1);
    fixed = true;  // overlay the shared store exactly at the reserved VA
  }
  int mflags = MAP_SHARED | (fixed ? MAP_FIXED : 0);
  void *p = ::mmap(va, len, PROT_READ | PROT_WRITE, mflags, fd,
                   static_cast<off_t>(offset));
  if (p == MAP_FAILED)
    return reinterpret_cast<uint8_t *>(-1);
  static const bool trace = std::getenv("DELTA_DMEM_TRACE") != nullptr;
  if (trace)
    std::fprintf(stderr, "[dmem] devmap off=%#zx len=%#zx -> %p (shared)\n", offset,
                 len, p);
  return reinterpret_cast<uint8_t *>(p);
}
}  // namespace krnl
