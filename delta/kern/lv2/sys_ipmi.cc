/*
 * PS4Delta : PS4 emulation and research project
 *
 * syscall 622: the kernel side of IPMI. The manager and the services it routes
 * to live in kern/ipmi; this is only the syscall boundary.
 */
#include "base/arch.h"
#include "guest_abi.h"

#include "kern/ipmi/ipmi.h"

namespace krnl {

int PS4ABI
sys_ipmimgr_call(u32 op, u32 kid, void* out, void* in, u64 insize, u64 arg6) {
  (void)arg6;  // libSceIpmi passes a fixed 0xdeadbadecafebeaf here
  return ipmi::ManagerCall(op, kid, out, in, insize);
}

}  // namespace krnl
