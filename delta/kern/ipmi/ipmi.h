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
 * Syscall 622 ABI:
 *     rdi=op  rsi=kid  rdx=out  r10=in  r8=insize  r9=arg6
 * so the result buffer comes BEFORE the request buffer. The wrapper pre-sets
 * the 32-bit result to -1 and, on a 0 return, reads it back as the call's value
 * (CreateClient's becomes the client kid).
 */

#include <cstddef>
#include "base/arch.h"
#include "kern/lv2/error_table.h"

namespace kern::ipmi {

// One synchronous method invocation call as a service sees it. Every buffer
// points into guest memory and every descriptor comes from the guest, so the
// accessors bounds-check: a service can't smash the host through a bad size.
class Invocation {
 public:
  explicit Invocation(void* request, u32 client = 0);

  u32 Client() const { return client_; }

  u32 Method() const;
  u32 NumIn() const;
  u32 NumOut() const;

  // Input buffer i, or nullptr. `size` receives its length.
  const void* Input(u32 i, u64& size) const;

  bool CanReply(u32 i, u64 size) const;

  // Fill output buffer i, zero-padding the rest of its capacity. Truncates to
  // that capacity. False if the descriptor is unusable.
  bool Reply(u32 i, const void* data, u64 size);
  bool ReplyU32(u32 i, u32 v);
  // Fill output buffer i to its full capacity with `byte`.
  bool ReplyFill(u32 i, u8 byte);
  // Zero every output buffer. This is what "the daemon replied, with nothing to
  // say" looks like; leaving the buffers at their pre-call sentinel instead
  // makes status getters read garbage and poll forever.
  void ReplyEmpty();

  // The method's own return code, separate from the manager syscall's. Defaults
  // to SCE_OK; a service sets it to report a method-level failure.
  void SetResult(i32 r);

 private:
  bool OutSlot(u32 i, u8*& data, u64& cap) const;

  void* req_;
  u32 client_;
};

// A stand-in for one system-service process. The manager routes every invoke
// from a client connected to name() here.
struct Service {
  virtual ~Service() = default;
  virtual const char* Name() const = 0;
  // A rejected handshake is returned through Connect's service-result slot.
  virtual i32 Connect(const void*, u64) { return 0; }
  // Default: the daemon accepted the call and had nothing to return.
  virtual void Invoke(Invocation& inv) { inv.ReplyEmpty(); }
  virtual i32 InvokeAsync(Invocation&, u32&) {
    return -static_cast<i32>(SysError::eNOSYS);
  }
  virtual i32 GetAsyncResult(u32, u32, u32, i32&) {
    return -static_cast<i32>(SysError::eNOSYS);
  }
  virtual void DestroyClient(u32) {}
  // The manager supplies the client ID and cursor and validates output buffers.
  // A failed receive must leave the cursor and outputs unchanged.
  virtual bool ReceiveEvent(u32, u32&, void*, u64, u64&) { return false; }
};

// sys_ipmimgr_call's body: decode the manager op and dispatch.
int ManagerCall(u32 op, u32 kid, void* out, void* in, u64 insize);

}  // namespace kern::ipmi
