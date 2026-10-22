#pragma once

#include <sys/types.h>
#include <cstddef>

namespace cpu::fex {

void* CpuMap(void* address,
             size_t size,
             int protection,
             int flags,
             int fd,
             off_t offset);
int CpuUnmap(void* address, size_t size);

}  // namespace cpu::fex
