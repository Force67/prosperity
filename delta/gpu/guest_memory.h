/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace gpu {

inline bool IsReadableMapping(uint64_t address, uint64_t bytes) {
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps)
    return false;

  const uint64_t end = address + bytes;
  uint64_t cursor = address;
  char line[512];
  while (cursor < end && std::fgets(line, sizeof(line), maps)) {
    unsigned long long begin, mapping_end;
    char permissions[5] = {};
    if (std::sscanf(line, "%llx-%llx %4s", &begin, &mapping_end, permissions) !=
            3 ||
        mapping_end <= cursor)
      continue;
    if (begin > cursor || permissions[0] != 'r')
      break;
    cursor = std::min<uint64_t>(end, mapping_end);
  }
  std::fclose(maps);
  return cursor == end;
}

inline bool IsReadableRange(uint64_t address, uint64_t bytes) {
  if (!bytes || address > std::numeric_limits<uint64_t>::max() - bytes)
    return false;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return false;

  constexpr size_t kBatch = 512;
  unsigned char probe[kBatch];
  iovec remote[kBatch];
  const uint64_t page = static_cast<uint64_t>(page_size);
  const uint64_t end = address + bytes;
  uint64_t cursor = address;
  while (cursor < end) {
    size_t count = 0;
    while (cursor < end && count < kBatch) {
      remote[count++] = {reinterpret_cast<void*>(cursor), 1};
      const uint64_t next_page = (cursor & ~(page - 1)) + page;
      cursor = next_page > cursor ? std::min(end, next_page) : end;
    }
    iovec local{probe, count};
    ssize_t read = -1;
    for (uint32_t attempt = 0;
         attempt < 2 && read != static_cast<ssize_t>(count); attempt++) {
      do {
        read = syscall(SYS_process_vm_readv, getpid(), &local, 1, remote, count,
                       0);
      } while (read < 0 && errno == EINTR);
    }
    if (read != static_cast<ssize_t>(count))
      return IsReadableMapping(address, bytes);
  }
  return true;
}

}  // namespace gpu
