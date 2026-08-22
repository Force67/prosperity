
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/string_ref.h>
#include <base/strings/xstring.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "kern/proc.h"
#include "kern/ps4/hardware_mode.h"
#include "error_table.h"
#include "kern/crash.h"
#include <sys/random.h>
#include <algorithm>
#include <cstring>
#include <cstdio>

#include <ctime>
#include <mutex>
#include <set>
#include <string>
#include <utl/options.h>

// These three are read unconditionally below (sys_cpuset_getaffinity, the
// sysctl handlers), so they must be DECLARED unconditionally too. They used to
// sit behind `#if defined(DELTA_BACKEND_NATIVE)`, which left the FEX/ARM build
// -- the only one SotC runs under -- with the uses but not the declarations, so
// this file simply did not compile there.
namespace {
DELTA_OPTION(bool, kArndZero, "DELTA_ARND_ZERO", false);
DELTA_OPTION(bool, kSotc7core, "DELTA_SOTC_7CORE", false);
DELTA_OPTION(bool, kSysctlCaller, "DELTA_SYSCTL_CALLER", false);
}  // namespace

namespace krnl {

// The guest's TSC. machdep.tsc_freq must equal the rate the guest's `rdtsc`
// actually advances, or every libkernel timer (frame pacing, timeouts: a guest
// `wait until rdtsc-start >= seconds*tsc_freq` loop) runs at the wrong speed.
// The PS4's invariant TSC is 1.6 GHz, but on the native x86 backend the guest's
// rdtsc IS the host's rdtsc, which ticks at the host TSC rate (often 2-4 GHz) --
// reporting 1.6 GHz there made all guest timers run host_rate/1.6 too fast.
// rdtsc can't be cheaply rescaled in the lifter (it's a 2-byte op: no room for a
// call, and trapping it per use is far too slow for busy-wait loops), so instead
// report the real host rate (calibrated once) so the two agree. On FEX/aarch64
// the guest rdtsc is emulated by the JIT, so keep the PS4-native 1.6 GHz.
static u64 guestTscFreq() {
#if defined(DELTA_BACKEND_NATIVE)
  static const u64 hz = [] {
    auto nowNs = [] {
      timespec t{};
      clock_gettime(CLOCK_MONOTONIC, &t);
      return static_cast<u64>(t.tv_sec) * 1000000000ull + t.tv_nsec;
    };
    u64 t0 = nowNs(), c0 = __builtin_ia32_rdtsc();
    timespec s{0, 20 * 1000 * 1000};  // ~20 ms; actual elapsed is measured below
    nanosleep(&s, nullptr);
    u64 dt = nowNs() - t0, dc = __builtin_ia32_rdtsc() - c0;
    if (dt == 0)
      return u64(1600000000);
    u64 f = static_cast<u64>(static_cast<double>(dc) * 1e9 /
                                       static_cast<double>(dt) + 0.5);
    BASE_LOGI("tsc", "calibrated host TSC = {} Hz (rdtsc==tsc_freq)",
              (unsigned long long)f);
    return f;
  }();
  return hz;
#else
  return u64(1600000000);  // FEX emulates rdtsc; PS4 invariant TSC rate
#endif
}
int sys_budget_get_ptype();

moduleInfo *called_in(void *addr);

// The oid names we have already reported as unhandled, so a polling caller
// cannot flood the log. Guarded because sysctl runs on any guest thread.
static std::set<std::string> &loggedOidNames() {
  static std::set<std::string> names;
  return names;
}
static std::mutex g_loggedOidLock;

int PS4ABI sys_is_in_sandbox() { return 0; }

int PS4ABI sys_cpuset_getaffinity(int /*level*/, int /*which*/, i64 /*id*/,
                                  size_t cpusetsize, void *mask) {
  // Report the CPUs the title is allowed to run on. Base PS4 grants a game 6
  // cores (0..5; the OS keeps 6/7). The KEX engine (Doom64) sizes its worker
  // pool from the set-bit count here (workers = availableCores/2); the old no-op
  // left `mask` unfilled, so it saw 0 cores -> "Max Worker Threads: 0" -> the
  // parallel job manager ran every job serially on the main thread and its job
  // pump spun, throttling the whole engine. Fill the low 6 bits.
  if (mask && cpusetsize) {
    std::memset(mask, 0, cpusetsize);
    u64 bits = 0x3F;  // cores 0..5
    // DELTA_SOTC_7CORE: also grant core 6. SotC's engine hardcodes its "Resource
    // Loading" thread to core 6 (mask 0x40) and its BPE JobSystem sizes its worker
    // pool from the set-bit count here, giving each worker an ordinal = spawn seq.
    // The job CLAIM path tests (job_affinity_mask & (1<<worker_ordinal)); a job the
    // engine pins to core 6 (mask 0x40) is then UNCLAIMABLE when only workers with
    // ordinals 0..5 exist -> the workers hot-spin on the scheduler umutex
    // (0x200004140) forever and the world-load finalize job never dispatches
    // (loader parks on its evf "job done" flag, the game loops on the loading
    // screen). Granting core 6 spawns a 7th worker (ordinal 6, bit 0x40) so that
    // job becomes claimable. Off by default (Isaac/Doom64 keep 6 cores).
    if (kSotc7core)
      bits = 0x7F;  // cores 0..6
    std::memcpy(mask, &bits,
                cpusetsize < sizeof(bits) ? cpusetsize : sizeof(bits));
  }
  return 0;
}

int PS4ABI sys_get_authinfo(int pid, void *infoOut) {
  // SceSelfAuthInfo is 0x88 (136) bytes, copied verbatim from the process
  // ucred (+88). Without privilege 0x2AE the kernel masks the buffer to just
  // the top three bits of qword[1]; a privileged caller gets the full block.
  // We hand back a plausible non-privileged game identity: auth_id of a normal
  // application plus a permissive capability mask. Returning 1 here (the old
  // behaviour) reads as EPERM and aborts libc.
  std::memset(infoOut, 0, 136);
  auto *p = reinterpret_cast<u64 *>(infoOut);
  p[0] = 0x3100000000000001ull; // auth_id: regular application
  p[2] = 0x2000038000000000ull; // capability bits
  p[4] = 0x4000400040000000ull; // attributes / shared
  return 0;
}

/*maybe should be moved to a proc file*/
int PS4ABI sys_get_proc_type_info(void *oinfo) {
  // Kernel output is a fixed 16-byte block:
  //   +0x00  uint64  reserved      (always zeroed)
  //   +0x08  int32   ptype         (budget process type, 0..3)
  //   +0x0c  uint8   cptype        (capability/process-class flags)
  //   +0x0d  uint8[3] padding
  // cptype bits:
  //   0x01 JIT compiler, 0x02 JIT application, 0x04 video player,
  //   0x08 disk-player UI, 0x10 use-video-service capability,
  //   0x20 webcore, 0x40 has sce program attribute.
  // A game SELF is a JIT application (0x02) carrying the sce program attribute
  // (0x40), so cptype = 0x42. Without the JIT-app bit libkernel's process-init
  // path skips the JIT shm setup it later expects to find.
  struct procTypeInfo {
    u64 reserved;
    i32 ptype;
    u8 cptype;
    u8 pad[3];
  };
  static_assert(sizeof(procTypeInfo) == 16);

  auto *out = reinterpret_cast<procTypeInfo *>(oinfo);
  out->reserved = 0;
  out->ptype = sys_budget_get_ptype();
  out->cptype = 0x42;  // JIT application | sce program attribute
  out->pad[0] = out->pad[1] = out->pad[2] = 0;
  return 0;
}

int PS4ABI sys_sysctl(int *name, u32 namelen, void *oldp, size_t *oldlenp,
                      const void *newp, size_t newlen) {
  // for sceKernelGetAppInfo
  if (name[0] == 1 && name[1] == 14 && name[2] == 35 && namelen == 4) {
    std::memset(oldp, 0, 72);
    return 0;
  }

  // PS5 kern.proc.36: the SDK version the title was compiled against.
  // sceKernelGetCompiledSdkVersion reads it from here and libkernel branches on
  // it all over process init -- notably, below SDK 1.70 it carves the initial
  // thread's static TLS out of the small SceKernelInternalMemory arena instead
  // of mmap'ing it, which a title with a 2 MiB PT_TLS (Skyrim) overflows.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 36 && namelen >= 3 &&
           proc::getActive()->getPlatform() == proc::platform::ps5) {
    if (oldp && oldlenp) {
      std::memset(oldp, 0, *oldlenp);
      if (*oldlenp >= sizeof(u32))
        *reinterpret_cast<u32 *>(oldp) = proc::getActive()->getSdkVersion();
    }
    return 0;
  }

  // PS5 kern.proc.68: an 8-byte per-process info block libkernel caches for a
  // getter libSceSaveData calls during sceSaveDataInitialize3. libkernel reads
  // the second dword as the value and treats a non-zero block as "already
  // cached". Left as ENOENT the getter returns 0x80020001 forever and the
  // title's save-data init state machine spins at 100% CPU with no syscalls.
  // libkernel's own caller defaults the value to 0 when the getter fails, so 0
  // is the safe answer.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 68 && namelen >= 3 &&
           proc::getActive()->getPlatform() == proc::platform::ps5) {
    if (oldp && oldlenp) {
      std::memset(oldp, 0, *oldlenp);
      if (*oldlenp >= 2 * sizeof(u32))
        static_cast<u32 *>(oldp)[0] = 1;
    }
    return 0;
  }

  // PS5 kern.proc.69: the geometry of the per-thread TLS/TCB arena. libkernel
  // reserves count * pages * 16 KiB of address space ending at 0x9_0000_0000 and
  // hands each thread one block, indexing it by (block - base) / blocksize.
  // The reply must be exactly this 24-byte struct: a mismatched size or version
  // makes libkernel zero the whole thing, reserve nothing, and abort the first
  // thread with 'Invalid TCB initialization'. A block holds the static TLS plus
  // the TCB; the third count reserves a second arena below 0xf_c2000000, which
  // 0 skips. Both fw 01.14.00 and 13.60 query this before any thread exists.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 69 && namelen >= 3 &&
           proc::getActive()->getPlatform() == proc::platform::ps5) {
    struct tlsArenaInfo {
      u64 size;
      u32 version;
      u32 blocks;
      u32 blockPages;
      u32 secondaryPages;
    };
    static_assert(sizeof(tlsArenaInfo) == 0x18);

    if (!oldp || !oldlenp || *oldlenp < sizeof(tlsArenaInfo))
      return -SysError::eINVAL;

    // The secondary block holds the thread's copy of the static TLS image, so
    // it has to fit every loaded module's PT_TLS at once.
    size_t tls = 0;
    for (auto &m : proc::getActive()->getModuleList())
      tls += m->getInfo().tlsSizeMem + m->getInfo().tlsalign;
    constexpr size_t kPage = 0x4000;
    size_t tlsPages = (tls + 0xFFFF + kPage - 1) / kPage;
    tlsPages = std::clamp<size_t>(tlsPages, 16, 1024);

    auto *out = static_cast<tlsArenaInfo *>(oldp);
    out->size = sizeof(tlsArenaInfo);
    out->version = 1;
    // This count is the process's thread ceiling: libkernel hands thread N the
    // block at base + N * blocksize, and the arena ENDS at 0x9_0000_0000, so the
    // first thread past the count maps outside it, fails, and scePthreadCreate
    // returns without ever reaching the kernel. At 64 Astro Bot stopped dead at
    // 64 threads with one Havok worker left to start. The ceiling on the other
    // side is libkernel's free-slot bitmap, a fixed 0x100-byte BSS array (one
    // BIT a block, but scanned a qword per block), so anything up to 2048 is
    // both in-array and in-arena.
    out->blocks = 256;
    out->blockPages = 16;  // the TCB and the thread's own bookkeeping
    // Never 0: that means "no secondary arena", and libkernel then derives the
    // thread's TLS pointer from the null block address and memsets through it.
    out->secondaryPages = static_cast<u32>(tlsPages);
    *oldlenp = sizeof(tlsArenaInfo);
    return 0;
  }

  // PS5 kern.proc.79: another app/process-info selector the PS5 system-service
  // client polls during net/NP init (kern.proc.35 is GetAppInfo above). Left as
  // ENOENT it reads as "retry", so the client thread spins re-querying and
  // creating/destroying a wait object each pass -- leaking the guest's fixed
  // ScePthread internal heap until it throws bad_alloc. Answer with a zeroed
  // buffer + success (same as .35) so the poll resolves. PS5-only: PS4 titles
  // never query this selector, so the PS4 path stays byte-identical.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 79 && namelen >= 3 &&
           proc::getActive()->getPlatform() == proc::platform::ps5) {
    if (oldp && oldlenp) {
      std::memset(oldp, 0, *oldlenp);
    }
    return 0;
  }

  // PS5 kern.61: a 24-byte status block a libkernel getter reads on behalf of
  // the title. Minecraft's main loop calls it every tick and, while it errors,
  // sits in that tick doing nothing else. The getter zeroes most of the struct
  // itself on the success path, so an all-zero answer is in-band.
  else if (name[0] == 1 && name[1] == 61 && namelen == 2 &&
           proc::getActive()->getPlatform() == proc::platform::ps5) {
    if (oldp && oldlenp)
      std::memset(oldp, 0, *oldlenp);
    return 0;
  }

  // kern.userstack
  else if (name[0] == 1 && name[1] == 33 && namelen == 2) {
    auto &info = proc::getActive()->getEnv();
    *static_cast<void **>(oldp) = info.userStack + info.userStackSize;
    BASE_LOGI("sysctl", "userstack -> base {:p}, end {:p}", info.userStack,
              oldp);
    return 0;
  }

  // kern.pagesize
  else if (name[0] == 6 && name[1] == 7 && namelen == 2) {
    *reinterpret_cast<u32 *>(oldp) = 0x4000;
    if (oldlenp)
      *oldlenp = sizeof(u32);
    return 0;
  }

#if 0
		else if (name[0] == 0x1337 && name[1] == 1 && namelen == 2) {
			*reinterpret_cast<u64*>(oldp) = 1357;
			return 0;
		}
#endif

  else if (name[0] == 0x1337 && name[1] == 1 && namelen == 2) {
    *reinterpret_cast<u64 *>(oldp) = 1;
    return 0;
  }

  // kern.proc.<41>: a "proc image area"/sanitizer flag libkernel reads at init.
  // Bit 0 gates loading libSceDbgUBSanitizer.sprx (a debug-only module). Return
  // 0 so libkernel takes the success path and skips the sanitizer preload.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 41 && namelen == 3) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u32)) {
      *reinterpret_cast<u32 *>(oldp) = 0;
      *oldlenp = sizeof(u32);
    }
    return 0;
  }

  // kern.cpumode (kern.14.42) is selected by the title's PSF attributes, not by
  // the Base/Neo GPU hardware profile.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 42 && namelen == 3) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u32)) {
      *reinterpret_cast<u32 *>(oldp) = ps4::cpuMode();
      *oldlenp = sizeof(u32);
    }
    return 0;
  }

  // kern.arnd (CTL_KERN.37): the kernel entropy source. This used to answer with
  // zeros, which is fine for the stack cookies and ASLR that libc wants but not
  // for a title that seeds a CSPRNG from it: Minecraft's OpenSSL sits in DTLS
  // certificate/key generation on an all-zero pool. DELTA_ARND_ZERO restores the
  // old deterministic fill.
  else if (name[0] == 1 && name[1] == 37 && namelen == 2) {
    auto length = *oldlenp;
    if (length > 256)
      length = 256;
    if (kArndZero || getrandom(oldp, length, 0) != static_cast<ssize_t>(length))
      std::memset(oldp, 0, length);
    *oldlenp = length;
    return 0;
  }

  // answer kern.prot.ptc
  else if (name[0] == 0x1337 && name[1] == 2 && namelen == 2) {
    *reinterpret_cast<u64 *>(oldp) = 1357;
    return 0;
  }

  // answer kern.sched.cpusize
  else if (name[0] == 0x1337 && name[1] == 4 && namelen == 2) {
    *reinterpret_cast<u32 *>(oldp) = 8;
    return 0;
  }

  // answer machdep.tsc_freq (synthetic oid {0x1337,5}). libkernel's
  // sceKernelGetTscFrequency reads this and uses it to convert rdtsc deltas to
  // time, so it MUST match the rate the guest's rdtsc actually advances at (see
  // guestTscFreq): the host TSC rate on native, 1.6 GHz on FEX.
  else if (name[0] == 0x1337 && name[1] == 5 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u64)) {
      *reinterpret_cast<u64 *>(oldp) = guestTscFreq();
      *oldlenp = sizeof(u64);
    }
    return 0;
  }

  // answer kern.sdk_version (synthetic oid {0x1337,6}), encoded 0x0MMMmmpp.
  // The real kernel reports the CALLING PROCESS's SDK version here, and PS5
  // libkernel's checkSdkVersionForLoadStartModule compares every module's
  // param SDK stamp against it: report less than the firmware the modules come
  // from and every fresh sceKernelLoadStartModule is unloaded again with
  // 0x8002002d "Invalid SDK version" (Demon's Souls: libSceMouse/libSceRudp
  // rejected -> Dantelion2 "Required cell system module(s) could not be
  // loaded" panic). PS4 keeps the proven 5.05 constant.
  else if (name[0] == 0x1337 && name[1] == 6 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u32)) {
      u32 v = 0x05050001;
      if (const auto *active = proc::getActive();
          active && active->getPlatform() == proc::platform::ps5 &&
          active->getSdkVersion())
        v = active->getSdkVersion();
      *reinterpret_cast<u32 *>(oldp) = v;
      *oldlenp = sizeof(u32);
    }
    return 0;
  }

  // hw.sce_main_socid (synthetic {0x1337,7}): the SoC identifier, which doubles as
  // the GPU chip revision. libSceAgc's shader-create (f3dg2CSgRKY) gates on it:
  // shaders whose min-GPU-target field (.shader_header[0x4c]) is > 5 are REJECTED
  // (0x8a6c003d) unless chipRev > 0x840f4f. All of Isaac's 38 embedded shaders use
  // target 6, so we must report the real Oberon revision (0x840fc0) or every shader
  // create fails -> empty pipelines -> zero SPI_SHADER_PGM -> nothing renders. This
  // oid is PS5-only (the 0x1337 family is synthetic PS5 config), so PS4 is unaffected.
  else if (name[0] == 0x1337 && name[1] == 7 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u32)) {
      const auto *active = proc::getActive();
      *reinterpret_cast<u32 *>(oldp) =
          active && active->getPlatform() == proc::platform::ps5
              ? 0x840fc0
              : ps4::hardwareModeProfile().mainSocId;
      *oldlenp = sizeof(u32);
    }
    return 0;
  }

  // vm.budgets.mlock_total (synthetic {0x1337,8}): total wired-memory budget in
  // bytes. Report a large pool (6 GiB) so heap-sizing consumers get a sane value
  // instead of 0; matches the order of the reported direct-memory pool.
  else if (name[0] == 0x1337 && name[1] == 8 && namelen == 2) {
    if (oldp && oldlenp) {
      u64 v = 0x180000000ull;
      size_t n = *oldlenp < sizeof(v) ? *oldlenp : sizeof(v);
      std::memcpy(oldp, &v, n);
      *oldlenp = n;
    }
    return 0;
  }

  // vm.budgets.mlock_avail (synthetic {0x1337,11}): the wired-memory budget
  // still AVAILABLE, the companion to mlock_total above. Left at ENOENT
  // libkernel's internal-memory allocator takes its failure path and sizes the
  // SceKernelInternalMemory arena minimally, then reports
  // "[ScePthread/System] Internal Memory is running out." and throws
  // std::bad_alloc -- which terminates the process via a UD2 in
  // libSceLibcInternal. That only bites once a real firmware module allocates
  // from the arena, which is why it stayed hidden while every service module ran
  // as an HLE shim (see DELTA_LLE in runtime/vprx/vprx.cpp).
  // Report the same 6 GiB as mlock_total: nothing has been wired yet.
  else if (name[0] == 0x1337 && name[1] == 11 && namelen == 2) {
    if (oldp && oldlenp) {
      u64 v = 0x180000000ull;
      size_t n = *oldlenp < sizeof(v) ? *oldlenp : sizeof(v);
      std::memcpy(oldp, &v, n);
      *oldlenp = n;
    }
    return 0;
  }

  // kern.rng_pseudo (synthetic {0x1337,12}): whether the kernel's pseudo RNG is
  // available. libSceRandom polls this and only stops once it reads non-zero --
  // answering 0 cost 30 million name2oid resolutions in 80 seconds and hung
  // Minecraft's OpenSSL key generation behind it.
  else if (name[0] == 0x1337 && name[1] == 12 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u32)) {
      *static_cast<u32 *>(oldp) = 1;
      *oldlenp = sizeof(u32);
    }
    return 0;
  }

  // Benign zero-filled PS5 config oids (synthetic {0x1337,9}): kern.amm.param,
  // kern.app.memconf, machdep.auto_update_version, plus the obfuscated
  // kern.gjevmtrb newer firmware adds. Zero is the default/"no-override"/
  // "feature off" answer for each. Reporting one as missing is not survivable:
  // libSceAgc reads kern.gjevmtrb via libkernel, which aborts outright when the
  // query errors.
  else if (name[0] == 0x1337 && name[1] == 9 && namelen == 2) {
    if (oldp && oldlenp) {
      size_t n = *oldlenp;
      if (n > 256)
        n = 256;
      std::memset(oldp, 0, n);
      *oldlenp = n;
    }
    return 0;
  }

  // kern.neomode (synthetic {0x1337,10}). It deliberately has its own oid so it
  // cannot inherit the unrelated zero-filled PS5 config response above.
  else if (name[0] == 0x1337 && name[1] == 10 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(u32)) {
      *reinterpret_cast<u32 *>(oldp) = ps4::isNeoMode() ? 1 : 0;
      *oldlenp = sizeof(u32);
    }
    return 0;
  }

  if (name[0] == 0 && name[1] == 3 && namelen == 2) {
    auto name = base::StringRef(static_cast<const char *>(newp), newlen);
    if (kSysctlCaller)
      BASE_LOGI("sysctl", "name2oid '{}'",
                base::String(static_cast<const char *>(newp), newlen).c_str());

    // PS5 system-info oids the network/system-service init resolves. Left
    // unhandled they returned ENOENT and the KAGE net thread spun (sizing its
    // heap from a missing budget), leaking sync objects until the pthread
    // internal heap ran out. Map them to synthetic oids answered below.
    if (name == "hw.sce_main_socid") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 7;
      *oldlenp = 8;
      return 0;
    } else if (name == "vm.budgets.mlock_total") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 8;
      *oldlenp = 8;
      return 0;
    } else if (name == "vm.budgets.mlock_avail") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 11;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.amm.param" || name == "kern.app.memconf" ||
               name == "machdep.auto_update_version" ||
               name == "kern.gjevmtrb" || name == "kern.nfxtmeqp" ||
               name == "kern.vegccsjk") {
      // The obfuscated names rotate per SDK release: gjevmtrb belongs to a
      // newer firmware, nfxtmeqp/vegccsjk to 08.40's libkernel. Its reader
      // discards the value, and zero is what the wrapper yields when its
      // hw.sce_main_socid gate is closed.
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 9;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.neomode") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 10;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.rng_pseudo") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 12;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.ps4_sdk_version") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 6;  // reuse kern.sdk_version answer
      *oldlenp = 8;
      return 0;
    }

    if (name == "kern.smp.cpus") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 1;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.proc.ptc") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 2;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.sched.cpusetsize") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 4;
      *oldlenp = 8;
      return 0;
    }

    else if (name == "vm.ps4dev.vm1.cpu.pt_total" ||
             name == "vm.ps4dev.vm1.cpu.pt_available" ||
             name == "vm.ps4dev.vm1.gpu.pt_total" ||
             name == "vm.ps4dev.vm1.gpu.pt_available" ||
             name == "vm.ps4dev.trcmem_total" ||
             name == "vm.ps4dev.trcmem_avail") {
      /*devkit-only oid, not present on retail*/
      return -SysError::eNOENT;
    }

    else if (name == "machdep.tsc_freq") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 5;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.sdk_version") {
      static_cast<u32 *>(oldp)[0] = 0x1337;
      static_cast<u32 *>(oldp)[1] = 6;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.cpumode") {
      // resolve to the real kern.14.42 mib handled above.
      static_cast<u32 *>(oldp)[0] = 1;
      static_cast<u32 *>(oldp)[1] = 14;
      static_cast<u32 *>(oldp)[2] = 42;
      *oldlenp = 12;
      return 0;
    }

    // A name we answer with ENOENT is often polled from a retry loop (Demon's
    // Souls asks for kern.nfxtmeqp ~9000 times a second), so log each distinct
    // one once. Unbounded, the log alone costs tens of megabytes and most of
    // the boot time.
    {
      std::string key(static_cast<const char *>(newp), newlen);
      std::lock_guard<std::mutex> lock(g_loggedOidLock);
      if (loggedOidNames().insert(key).second)
        BASE_LOGI("sysctl", "UNHANDLED name2oid: '{}'", key.c_str());
    }
    return -SysError::eNOENT;
  }

  /*for sceKernelGetLibkernelTextLocation*/

  BASE_LOGI("sysctl", "sysctl referenced by {:p}", _ReturnAddress());
  called_in(_ReturnAddress());
  // SCOUT: log the unhandled mib and soft-fail (ENOENT) instead of trapping so
  // the guest can decide how to cope, and we can see what it queries next.
  base::String mib;
  base::FormatTo(mib, "UNHANDLED mib namelen={}:", namelen);
  for (u32 i = 0; i < namelen && i < 8; i++)
    base::FormatTo(mib, " {}", name[i]);
  BASE_LOGI("sysctl", "{}", mib.c_str());
  // The out buffer is usually a caller stack local, so scanning up from it finds
  // the guest frames that wanted this oid.
  if (kSysctlCaller && oldp) {
    auto *sp = reinterpret_cast<const uintptr_t *>(oldp);
    int shown = 0;
    for (int i = 0; i < 512 && shown < 6; i++) {
      char sym[256];
      symbolize(sp[i], sym, sizeof(sym));
      if (std::strstr(sym, "(.text)")) {
        BASE_LOGI("sysctl", "  caller {}", sym);
        shown++;
      }
    }
  }
  return -SysError::eNOENT;
}
} // namespace krnl
