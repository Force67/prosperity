
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

#include "../proc.h"
#include "error_table.h"

namespace krnl {
int sys_budget_get_ptype();

moduleInfo *called_in(void *addr);

int PS4ABI sys_is_in_sandbox() { return 0; }

int PS4ABI sys_cpuset_getaffinity() { return 0; }

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

  // kern.userstack
  else if (name[0] == 1 && name[1] == 33 && namelen == 2) {
    auto &info = proc::getActive()->getEnv();
    *static_cast<void **>(oldp) = info.userStack + info.userStackSize;
    std::printf("userstack -> base %p, end %p\n", info.userStack, oldp);
    return 0;
  }

  // kern.pagesize
  else if (name[0] == 6 && name[1] == 7 && namelen == 2) {
    *reinterpret_cast<uint32_t *>(oldp) = 4096;
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

  // answer machdep.tsc_freq (synthetic oid {0x1337,5}). The TSC on the PS4 APU
  // runs at a fixed ~1.6 GHz; libkernel's sceKernelGetTscFrequency divides by
  // this during every module's CRT init, so a zero/garbage value risks a
  // divide-by-zero or wildly wrong timing. Report a plausible fixed rate.
  else if (name[0] == 0x1337 && name[1] == 5 && namelen == 2) {
    if (oldp && oldlenp && *oldlenp >= sizeof(uint64_t)) {
      *reinterpret_cast<uint64_t *>(oldp) = 1593600000ull;
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

  if (name[0] == 0 && name[1] == 3 && namelen == 2) {
    auto name = base::StringRef(static_cast<const char *>(newp), newlen);
    if (name == "kern.neomode") {
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