/*
 * PS4Delta : PS4 emulation and research project
 *
 * sys_ipmimgr_call (622): the kernel side of PS4 IPMI (Inter-Process Method
 * Invocation), the RPC the real Sony libraries use to reach the system-service
 * processes (ShellCore, LNC, AppMessaging, ...). We host no service processes,
 * so the job here is to answer the calls the game's libSceIpmi makes well enough
 * that those clients initialise instead of failing and leaving null singletons.
 *
 * ABI (recovered from libSceIpmi's wrapper at .text+0x3fd0 -> libkernel stub ->
 * syscall 622): the kernel receives
 *     rdi=op  rsi=kid  rdx=out  r10=in  r8=insize  r9=arg6(0xdeadbadecafebeaf)
 * so the result buffer (out) comes BEFORE the request buffer (in). The wrapper
 * pre-sets the 32-bit result to -1 and, on a 0 (success) return, reads the
 * result word back out as the call's value (CreateClient's becomes m_clientKid).
 * A handler therefore returns 0 and writes its result word into `out`.
 */
#include <base.h>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace krnl {

// IPMI manager command numbers (libSceIpmi op -> sys_ipmimgr_call).
enum {
  IPMI_CREATE_CLIENT = 2,  // sceIpmiMgrCreateClient  -> result = client kid
  IPMI_DESTROY_CLIENT = 3, // sceIpmiMgrDestroyClient -> result = 0 (asserted)
};

static std::atomic<uint32_t> g_nextClientKid{1};

int PS4ABI sys_ipmimgr_call(uint32_t op, uint32_t kid, void *out, void *in,
                            uint64_t insize, uint64_t arg6) {
  (void)kid;
  (void)arg6;
  auto setResult = [&](uint32_t v) {
    if (out)
      *static_cast<uint32_t *>(out) = v;
  };

  switch (op) {
  case IPMI_CREATE_CLIENT:
    // Hand back a fresh client kid; the wrapper stores it as m_clientKid.
    setResult(g_nextClientKid.fetch_add(1));
    return 0;
  case IPMI_DESTROY_CLIENT:
    setResult(0);
    return 0;
  case 784:
    // StopSession/Disconnect-style calls pass a pointer to a guest status word
    // in the request payload. libSceIpmi asserts that status is zero after the
    // manager syscall returns; leave it clear when no service process exists.
    if (in && insize >= sizeof(uint64_t)) {
      uint32_t *status = nullptr;
      std::memcpy(&status, in, sizeof(status));
      if (status)
        *status = 0;
    }
    setResult(0);
    return 0;
  default:
    // Connect / InvokeSyncMethod / etc.: report success with a zero result word
    // and no payload so the wrapper takes its success path.
    setResult(0);
    return 0;
  }
}

} // namespace krnl
