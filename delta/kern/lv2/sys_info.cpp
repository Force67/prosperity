
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

  // cxx init stuff
  else if (name[0] == 1 && name[1] == 37 && namelen == 2) {
    auto length = *oldlenp;
    if (length > 256)
      length = 256;
    memset(oldp, 4, length);
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
      return true;
    }

    else if (name == "vm.ps4dev.vm1.cpu.pt_total" ||
             name == "vm.ps4dev.vm1.cpu.pt_available" ||
             name == "vm.ps4dev.vm1.gpu.pt_total" ||
             name == "vm.ps4dev.vm1.gpu.pt_available" ||
             name == "vm.ps4dev.trcmem_total" ||
             name == "vm.ps4dev.trcmem_avail") {
      /*DK, not present on retail*/
      return 2;
    }
  }

  /*for sceKernelGetLibkernelTextLocation*/

  std::printf("sysctl referenced by %p\n", _ReturnAddress());
  called_in(_ReturnAddress());
  __debugbreak();
  return 0;
}
} // namespace krnl