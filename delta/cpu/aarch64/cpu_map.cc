#include "cpu/aarch64/cpu_map.h"

#include <sys/mman.h>

#include "memory_debug/memory_debug.h"

namespace cpu::fex {

void* CpuMap(void* address,
             size_t size,
             int protection,
             int flags,
             int fd,
             off_t offset) {
  void* result = ::mmap(address, size, protection, flags, fd, offset);
  if (result != MAP_FAILED && protection != PROT_NONE)
    memory_debug::TrackMapping(result, size, memory_debug::CurrentBucket(),
                               (flags & MAP_FIXED) != 0);
  else if (result != MAP_FAILED && (flags & MAP_FIXED))
    memory_debug::ForgetMapping(result, size);
  return result;
}

int CpuUnmap(void* address, size_t size) {
  const int result = ::munmap(address, size);
  if (result == 0)
    memory_debug::ForgetMapping(address, size);
  return result;
}

}  // namespace cpu::fex
