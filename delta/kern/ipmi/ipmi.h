#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * IPMI (Inter-Process Method Invocation) is the RPC the Sony libraries use to
 * reach the system-service processes: ShellCore, LNC, NpManager, PlayGo and so
 * on. We run no service processes, so this subsystem stands in for them: the
 * manager keeps a client table and routes each invoke to a Service that answers
 * as that daemon would.
 *
 * ABI (recovered from libSceIpmi's wrapper -> libkernel stub -> syscall 622):
 *     rdi=op  rsi=kid  rdx=out  r10=in  r8=insize  r9=arg6
 * so the result buffer comes BEFORE the request buffer. The wrapper pre-sets the
 * 32-bit result to -1 and, on a 0 return, reads it back as the call's value
 * (CreateClient's becomes the client kid).
 */

#include <cstddef>
#include <cstdint>

namespace krnl::ipmi {

// One sceIpmiClientInvokeSyncMethod call as a service sees it. Every buffer
// points into guest memory and every descriptor comes from the guest, so the
// accessors bounds-check: a service can't smash the host through a bad size.
class Invocation {
public:
  explicit Invocation(void *request);

  uint32_t method() const;
  uint32_t numIn() const;
  uint32_t numOut() const;

  // Input buffer i, or nullptr. `size` receives its length.
  const void *input(uint32_t i, uint64_t &size) const;

  // Fill output buffer i, zero-padding the rest of its capacity. Truncates to
  // that capacity. False if the descriptor is unusable.
  bool reply(uint32_t i, const void *data, uint64_t size);
  bool replyU32(uint32_t i, uint32_t v);
  // Fill output buffer i to its full capacity with `byte`.
  bool replyFill(uint32_t i, uint8_t byte);
  // Zero every output buffer. This is what "the daemon replied, with nothing to
  // say" looks like; leaving the buffers at their pre-call sentinel instead
  // makes status getters read garbage and poll forever.
  void replyEmpty();

  // The method's own return code, separate from the manager syscall's. Defaults
  // to SCE_OK; a service sets it to report a method-level failure.
  void setResult(int32_t r);

private:
  bool outSlot(uint32_t i, uint8_t *&data, uint64_t &cap) const;

  void *req_;
};

// A stand-in for one system-service process. The manager routes every invoke
// from a client connected to name() here.
struct Service {
  virtual ~Service() = default;
  virtual const char *name() const = 0;
  // Default: the daemon accepted the call and had nothing to return.
  virtual void invoke(Invocation &inv) { inv.replyEmpty(); }
};

// sys_ipmimgr_call's body: decode the manager op and dispatch.
int managerCall(uint32_t op, uint32_t kid, void *out, void *in, uint64_t insize);

} // namespace krnl::ipmi
