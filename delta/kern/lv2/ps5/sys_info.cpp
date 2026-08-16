/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero system-information syscalls. See sys_info.h.
 */

#include "kern/lv2/ps5/sys_info.h"

#include <base/logging.h>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <base/strings/string_ref.h>
#include <utl/options.h>

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_info.h"

namespace {
DELTA_OPTION(u64, kPs5Cores, "DELTA_PS5_CORES", 0);
}  // namespace

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
  // DELTA_PS5_CORES overrides the mask so a title whose job system derives
  // worker ordinals from it can be swept without a rebuild.
  const u64 bits = kPs5Cores ? kPs5Cores : 0x7Full;  // cores 0..6
  std::memcpy(mask, &bits,
              cpusetsize < sizeof(bits) ? cpusetsize : sizeof(bits));
  return 0;
}

namespace {
// DELTA_SYSCTL_CENSUS: which oid a title is actually asking for. Demon's Souls
// makes 2.2 million sysctl calls while otherwise idle, and a poll that hot is
// never the point of the call: it is a loop somewhere above it.
DELTA_OPTION(bool, kSysctlCensus, "DELTA_SYSCTL_CENSUS", false);

void censusSysctl(int *name, u32 namelen, const void *newp, size_t newlen) {
  if (!kSysctlCensus)
    return;
  static std::map<std::string, u64> hist;
  static std::mutex lock;
  static u64 calls = 0;
  std::string key;
  if (name && namelen == 2 && name[0] == 0 && name[1] == 3 && newp && newlen)
    key.assign(static_cast<const char *>(newp), newlen);  // name2oid
  else if (name)
    for (u32 i = 0; i < namelen && i < 6; i++)
      key += (i ? "." : "") + std::to_string(name[i]);
  std::lock_guard<std::mutex> lk(lock);
  hist[key]++;
  if (++calls % 200000)
    return;
  BASE_LOGI("sysctlcensus", "--- after {} calls ---", calls);
  for (const auto &[oid, n] : hist)
    if (n > 1000)
      BASE_LOGI("sysctlcensus", "  {:<28} {}", oid.c_str(), n);
}
}  // namespace

// kern.proc.35 is wider on Prospero than the Orbis reply the shared handler
// builds, and a title reads it as a fixed-size struct, so hand back a zeroed
// block of exactly the length asked for rather than a short one.
int PS4ABI ps5_sysctl(int *name, u32 namelen, void *oldp, size_t *oldlenp,
                      const void *newp, size_t newlen) {
  censusSysctl(name, namelen, newp, newlen);
  // kern.proc.35 and kern.proc.55 are both wider on Prospero than the Orbis
  // replies the shared handler builds, and a title reads each as a fixed-size
  // struct, so hand back a zeroed block of exactly the length asked for rather
  // than a short one.
  if (name && namelen >= 3 && name[0] == 1 && name[1] == 14 &&
      (name[2] == 35 || name[2] == 55)) {
    if (!oldp || !oldlenp)
      return -SysError::eINVAL;
    std::memset(oldp, 0, *oldlenp);
    return 0;
  }
  // Prospero libkernel resolves a handful of oids Orbis never had. None of them
  // gates anything we have seen, but leaving them at ENOENT is what put a title
  // into a 9000-a-second retry loop once already, so answer them with the
  // shared zero-filled synthetic oid rather than finding out again.
  if (name && namelen == 2 && name[0] == 0 && name[1] == 3 && newp && newlen &&
      oldp && oldlenp && *oldlenp >= 8) {
    const base::StringRef want(static_cast<const char *>(newp), newlen);
    if (want == "kern.kern_type" || want == "kern.universal_mode" ||
        want == "kern.backup_restore_mode" || want == "kern.rtld_control" ||
        want == "kern.fsst_param" || want == "kern.geom.updtfmt" ||
        want == "hw.availpages" || want == "machdep.openpsid") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 9;  // the zero-filled PS5 config group
      *oldlenp = 8;
      return 0;
    }
  }
  return sys_sysctl(name, namelen, oldp, oldlenp, newp, newlen);
}

}  // namespace krnl
