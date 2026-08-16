/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero system-information syscalls. See sys_info.h.
 */

#include "kern/lv2/ps5/sys_info.h"

#include <base/logging.h>
#include <cstring>

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_info.h"

namespace krnl {

// A PS5 reserves one of its eight Zen 2 cores for the system and grants the
// title the other seven; a PS4 grants six of eight. The count matters well
// beyond scheduling: engines size their worker pool from the set bits here, and
// Bluepoint's BPE job system additionally uses each worker's spawn ordinal as
// the bit a job's affinity mask is tested against. One worker short, a job the
// engine pinned to the missing core can never be claimed, its workers spin on
// the scheduler forever and whatever that job was gating (a world load, a
// resource finalize) never completes.
int PS4ABI ps5_cpuset_getaffinity(int /*level*/, int /*which*/, i64 /*id*/,
                                  size_t cpusetsize, void *mask) {
  if (!mask || !cpusetsize)
    return 0;
  std::memset(mask, 0, cpusetsize);
  const u64 bits = 0x7F;  // cores 0..6
  std::memcpy(mask, &bits,
              cpusetsize < sizeof(bits) ? cpusetsize : sizeof(bits));
  return 0;
}

// kern.proc.35 is wider on Prospero than the Orbis reply the shared handler
// builds, and a title reads it as a fixed-size struct, so hand back a zeroed
// block of exactly the length asked for rather than a short one.
int PS4ABI ps5_sysctl(int *name, u32 namelen, void *oldp, size_t *oldlenp,
                      const void *newp, size_t newlen) {
  if (name && namelen == 4 && name[0] == 1 && name[1] == 14 && name[2] == 35) {
    if (!oldp || !oldlenp)
      return -SysError::eINVAL;
    std::memset(oldp, 0, *oldlenp);
    return 0;
  }
  return sys_sysctl(name, namelen, oldp, oldlenp, newp, newlen);
}

}  // namespace krnl
