
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <base/strings/string_ref.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "../../proc.h"
#include "error_table.h"
#include <cstdio>

#if defined(DELTA_BACKEND_NATIVE)
#include <ctime>
#endif

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
static uint64_t guestTscFreq() {
#if defined(DELTA_BACKEND_NATIVE)
  static const uint64_t hz = [] {
    auto nowNs = [] {
      timespec t{};
      clock_gettime(CLOCK_MONOTONIC, &t);
      return static_cast<uint64_t>(t.tv_sec) * 1000000000ull + t.tv_nsec;
    };
    uint64_t t0 = nowNs(), c0 = __builtin_ia32_rdtsc();
    timespec s{0, 20 * 1000 * 1000};  // ~20 ms; actual elapsed is measured below
    nanosleep(&s, nullptr);
    uint64_t dt = nowNs() - t0, dc = __builtin_ia32_rdtsc() - c0;
    if (dt == 0)
      return uint64_t(1600000000);
    uint64_t f = static_cast<uint64_t>(static_cast<double>(dc) * 1e9 /
                                       static_cast<double>(dt) + 0.5);
    std::fprintf(stderr, "[tsc] calibrated host TSC = %llu Hz (rdtsc==tsc_freq)\n",
                 (unsigned long long)f);
    return f;
  }();
  return hz;
#else
  return uint64_t(1600000000);  // FEX emulates rdtsc; PS4 invariant TSC rate
#endif
}
int sys_budget_get_ptype();

moduleInfo *called_in(void *addr);

int PS4ABI sys_is_in_sandbox() { return 0; }

int PS4ABI sys_cpuset_getaffinity(int /*level*/, int /*which*/, int64_t /*id*/,
                                  size_t cpusetsize, void *mask) {
  // Report the CPUs the title is allowed to run on. Base PS4 grants a game 6
  // cores (0..5; the OS keeps 6/7). The KEX engine (Doom64) sizes its worker
  // pool from the set-bit count here (workers = availableCores/2); the old no-op
  // left `mask` unfilled, so it saw 0 cores -> "Max Worker Threads: 0" -> the
  // parallel job manager ran every job serially on the main thread and its job
  // pump spun, throttling the whole engine. Fill the low 6 bits.
  if (mask && cpusetsize) {
    std::memset(mask, 0, cpusetsize);
    uint64_t bits = 0x3F;  // cores 0..5
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
    static const bool sevenCore = std::getenv("DELTA_SOTC_7CORE") != nullptr;
    if (sevenCore)
      bits = 0x7F;  // cores 0..6
    std::memcpy(mask, &bits,
                cpusetsize < sizeof(bits) ? cpusetsize : sizeof(bits));
  }
  return 0;
}

int PS4ABI sys_get_authinfo(int pid, void *infoOut) {
  // SceSelfAuthInfo (136 bytes). Hand back a plausible non-privileged game
  // identity: auth_id of a normal application plus a permissive capability
  // mask. Returning 1 here (the old behaviour) reads as EPERM and aborts libc.
  std::memset(infoOut, 0, 136);
  auto *p = reinterpret_cast<uint64_t *>(infoOut);
  p[0] = 0x3100000000000001ull; // auth_id: regular application
  p[2] = 0x2000038000000000ull; // capability bits
  p[4] = 0x4000400040000000ull; // attributes / shared
  return 0;
}

/*maybe should be moved to a proc file*/
int PS4ABI sys_get_proc_type_info(void *oinfo) {
  struct dargs {
    size_t size;
    uint32_t ptype;
    uint32_t pflags;
  };

  auto *args = reinterpret_cast<dargs *>(oinfo);
  args->size = sizeof(dargs);
  args->ptype = sys_budget_get_ptype();
  args->pflags = 0; // TODO: handle flag 0x40 (sceprogramattr)
  return 0;
}

int PS4ABI sys_sysctl(int *name, uint32_t namelen, void *oldp, size_t *oldlenp,
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
      if (*oldlenp >= sizeof(uint32_t))
        *reinterpret_cast<uint32_t *>(oldp) = proc::getActive()->getSdkVersion();
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
      if (*oldlenp >= 2 * sizeof(uint32_t))
        static_cast<uint32_t *>(oldp)[0] = 1;
    }
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

  // kern.userstack
  else if (name[0] == 1 && name[1] == 33 && namelen == 2) {
    auto &info = proc::getActive()->getEnv();
    *static_cast<void **>(oldp) = info.userStack + info.userStackSize;
    std::printf("userstack -> base %p, end %p\n", info.userStack, oldp);
    return 0;
  }

  // kern.pagesize
  else if (name[0] == 6 && name[1] == 7 && namelen == 2) {
    *reinterpret_cast<uint32_t *>(oldp) = 0x4000;
    if (oldlenp)
      *oldlenp = sizeof(uint32_t);
    return 0;
  }

#if 0
		else if (name[0] == 0x1337 && name[1] == 1 && namelen == 2) {
			*reinterpret_cast<uint64_t*>(oldp) = 1357;
			return 0;
		}
#endif

  else if (name[0] == 0x1337 && name[1] == 1 && namelen == 2) {
    *reinterpret_cast<uint64_t *>(oldp) = 1;
    return 0;
  }

  // kern.proc.<41>: a "proc image area"/sanitizer flag libkernel reads at init.
  // Bit 0 gates loading libSceDbgUBSanitizer.sprx (a debug-only module). Return
  // 0 so libkernel takes the success path and skips the sanitizer preload.
  else if (name[0] == 1 && name[1] == 14 && name[2] == 41 && namelen == 3) {
    if (oldp && oldlenp && *oldlenp >= sizeof(uint32_t)) {
      *reinterpret_cast<uint32_t *>(oldp) = 0;
      *oldlenp = sizeof(uint32_t);
    }
    return 0;
  }

  // kern.cpumode (kern.14.42): the CPU mode, base PS4 (6/7-core "normal") vs
  // Neo/Pro. Isaac is a base-PS4 title; report mode 0 (normal). Both the direct
  // mib query and the sysctlbyname("kern.cpumode") name2oid path hit this; left
  // unhandled the game spins re-resolving it (with an uninitialised oid buffer).
  else if (name[0] == 1 && name[1] == 14 && name[2] == 42 && namelen == 3) {
    if (oldp && oldlenp && *oldlenp >= sizeof(uint32_t)) {
      *reinterpret_cast<uint32_t *>(oldp) = 0;  // normal (non-Neo) mode
      *oldlenp = sizeof(uint32_t);
    }
    return 0;
  }

  // kern.arnd (CTL_KERN.37): random bytes used by the C++ runtime / libc for
  // cookies and ASLR. Zero is a benign, deterministic value; a non-zero fill
  // (0x04) was being consumed as garbage allocation sizes downstream.
  else if (name[0] == 1 && name[1] == 37 && namelen == 2) {
    auto length = *oldlenp;
    if (length > 256)
      length = 256;
    memset(oldp, 0, length);
    *oldlenp = length;
    return 0;
  }

  // answer kern.prot.ptc
  else if (name[0] == 0x1337 && name[1] == 2 && namelen == 2) {
    *reinterpret_cast<uint64_t *>(oldp) = 1357;
    return 0;
  }

  // answer kern.sched.cpusize
  else if (name[0] == 0x1337 && name[1] == 4 && namelen == 2) {
    *reinterpret_cast<uint32_t *>(oldp) = 8;
    return 0;
  }

  // answer machdep.tsc_freq (synthetic oid {0x1337,5}). libkernel's
  // sceKernelGetTscFrequency reads this and uses it to convert rdtsc deltas to
  // time, so it MUST match the rate the guest's rdtsc actually advances at (see
  // guestTscFreq): the host TSC rate on native, 1.6 GHz on FEX.
  else if (name[0] == 0x1337 && name[1] == 5 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(uint64_t)) {
      *reinterpret_cast<uint64_t *>(oldp) = guestTscFreq();
      *oldlenp = sizeof(uint64_t);
    }
    return 0;
  }

  // answer kern.sdk_version (synthetic oid {0x1337,6}): the *system* firmware
  // SDK version, encoded as 0x0MMMmmpp (major/minor/patch). 5.05 (0x05050001)
  // is broadly compatible and matches what most retail titles tolerate.
  else if (name[0] == 0x1337 && name[1] == 6 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(uint32_t)) {
      *reinterpret_cast<uint32_t *>(oldp) = 0x05050001;
      *oldlenp = sizeof(uint32_t);
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
    if (oldp && oldlenp && *oldlenp >= sizeof(uint32_t)) {
      *reinterpret_cast<uint32_t *>(oldp) = 0x840fc0;
      *oldlenp = sizeof(uint32_t);
    }
    return 0;
  }

  // vm.budgets.mlock_total (synthetic {0x1337,8}): total wired-memory budget in
  // bytes. Report a large pool (6 GiB) so heap-sizing consumers get a sane value
  // instead of 0; matches the order of the reported direct-memory pool.
  else if (name[0] == 0x1337 && name[1] == 8 && namelen == 2) {
    if (oldp && oldlenp) {
      uint64_t v = 0x180000000ull;
      size_t n = *oldlenp < sizeof(v) ? *oldlenp : sizeof(v);
      std::memcpy(oldp, &v, n);
      *oldlenp = n;
    }
    return 0;
  }

  // Benign zero-filled PS5 config oids (synthetic {0x1337,9}): kern.amm.param,
  // kern.app.memconf, machdep.auto_update_version, kern.neomode. Zero is the
  // default/"non-Neo"/"no-override" answer for each.
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

  if (name[0] == 0 && name[1] == 3 && namelen == 2) {
    auto name = base::StringRef(static_cast<const char *>(newp), newlen);

    // PS5 system-info oids the network/system-service init resolves. Left
    // unhandled they returned ENOENT and the KAGE net thread spun (sizing its
    // heap from a missing budget), leaking sync objects until the pthread
    // internal heap ran out. Map them to synthetic oids answered below.
    if (name == "hw.sce_main_socid") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 7;
      *oldlenp = 8;
      return 0;
    } else if (name == "vm.budgets.mlock_total") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 8;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.amm.param" || name == "kern.app.memconf" ||
               name == "machdep.auto_update_version" || name == "kern.neomode") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 9;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.ps4_sdk_version") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 6;  // reuse kern.sdk_version answer
      *oldlenp = 8;
      return 0;
    }

    if (name == "kern.smp.cpus") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 1;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.proc.ptc") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 2;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.sched.cpusetsize") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 4;
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
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 5;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.sdk_version") {
      static_cast<uint32_t *>(oldp)[0] = 0x1337;
      static_cast<uint32_t *>(oldp)[1] = 6;
      *oldlenp = 8;
      return 0;
    } else if (name == "kern.cpumode") {
      // resolve to the real kern.14.42 mib handled above.
      static_cast<uint32_t *>(oldp)[0] = 1;
      static_cast<uint32_t *>(oldp)[1] = 14;
      static_cast<uint32_t *>(oldp)[2] = 42;
      *oldlenp = 12;
      return 0;
    }

    std::printf("[sysctl] UNHANDLED name2oid: '%.*s'\n", (int)newlen,
                static_cast<const char *>(newp));
    return -SysError::eNOENT;
  }

  /*for sceKernelGetLibkernelTextLocation*/

  std::printf("sysctl referenced by %p\n", _ReturnAddress());
  called_in(_ReturnAddress());
  // SCOUT: log the unhandled mib and soft-fail (ENOENT) instead of trapping so
  // the guest can decide how to cope, and we can see what it queries next.
  std::printf("[sysctl] UNHANDLED mib namelen=%u:", namelen);
  for (uint32_t i = 0; i < namelen && i < 8; i++)
    std::printf(" %d", name[i]);
  std::printf("\n");
  return -SysError::eNOENT;
}
} // namespace krnl
