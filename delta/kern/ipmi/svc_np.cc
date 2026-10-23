/*
 * PS4Delta : PS4 emulation and research project
 *
 * The NP (PlayStation Network) daemons. libSceNpManager talks to "SceNpMgrIpc"
 * for account/session state and to "SceNpService" for the web-service side.
 *
 * We have no PSN account and no network, so the state to report is a settled
 * signed-out one. That is not the same as replying with zeros: zero is
 * SCE_NP_STATE_UNKNOWN, i.e. "ask again", and a title waiting for NP to settle
 * before it leaves its boot screen will wait forever.
 */

#include "base/arch.h"

#include "kern/ipmi/services.h"

namespace kern::ipmi {
namespace {

enum {
  // in: u32 userId -> out: u32 internal state. sceNpGetState maps the reply
  // through a 3-entry table {1,1,2}: 1 and 2 become SCE_NP_STATE_SIGNED_OUT, 3
  // becomes SCE_NP_STATE_SIGNED_IN, everything else SCE_NP_STATE_UNKNOWN.
  // Verified identical in PS4 and PS5 (fw 01.14.00) libSceNpManager.
  kMgrGetState = 0xe,
  // in: {u32 userId, u8 flag} -> out: SceNpId (0x24 bytes). sceNpGetNpId
  // returns the method result as is.
  kMgrGetNpId = 0x17,
  // in: {u32 userId, u32} -> out[0] payload, out[1] u32 payload length. The
  // 0x800 capacity libSceNpManager declares is a cap, not the caller's buffer
  // (NpMessageInit passes 0x100 of stack), so only the length may be written.
  kMgrGetPayload = 0x24,
};

// The internal state that maps to SCE_NP_STATE_SIGNED_OUT.
constexpr u32 kInternalSignedOut = 1;
constexpr i32 kSignedOut = static_cast<i32>(0x80550006u);

struct NpManager : Service {
  const char* Name() const override { return "SceNpMgrIpc"; }

  void Invoke(Invocation& inv) override {
    switch (inv.Method()) {
      case kMgrGetState:
        inv.ReplyU32(0, kInternalSignedOut);
        break;
      case kMgrGetPayload:
        inv.ReplyU32(1, 0);
        break;
      case kMgrGetNpId:
        // An all-zero id with SCE_OK reads as a signed-in user.
        inv.ReplyEmpty();
        inv.SetResult(kSignedOut);
        break;
      default:
        // The rest of this daemon's getters share one shape (inputs, then a
        // single u32 the library hands back). Zero reads as "no value" for all
        // of them: no account id, no country, no entitlement.
        inv.ReplyEmpty();
        break;
    }
  }
};

// sceNpCheckCallback pumps this once per frame (method 0x70004, five output
// buffers). With no account there is never an event to deliver, and an empty
// reply is exactly "nothing pending", but the buffers must still be written,
// though, or the caller reads whatever was on its stack.
struct NpWeb : Service {
  const char* Name() const override { return "SceNpService"; }
};

NpManager g_np_manager;
NpWeb g_np_web;

}  // namespace

Service& NpManagerService() {
  return g_np_manager;
}
Service& NpWebService() {
  return g_np_web;
}

}  // namespace kern::ipmi
