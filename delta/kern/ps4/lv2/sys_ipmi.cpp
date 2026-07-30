/*
 * PS4Delta : PS4 emulation and research project
 *
 * syscall 622: the kernel side of IPMI. The manager and the services it routes
 * to live in kern/ipmi; this is only the syscall boundary.
 */
#include <base.h>

#include "kern/ipmi/ipmi.h"

namespace krnl {

int PS4ABI sys_ipmimgr_call(uint32_t op, uint32_t kid, void *out, void *in,
                            uint64_t insize, uint64_t arg6) {
  (void)arg6; // libSceIpmi passes a fixed 0xdeadbadecafebeaf here
  return ipmi::managerCall(op, kid, out, in, insize);
}

} // namespace krnl
