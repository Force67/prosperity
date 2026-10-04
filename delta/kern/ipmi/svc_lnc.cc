/*
 * PS4Delta : PS4 emulation and research project
 *
 * The launch daemon. ShellCore owns the lifecycle of the running title, and
 * Launch and system service clients query the launch daemon for its state.
 *
 * We only ever run one title, in the foreground, for the whole process
 * lifetime, so the answer is constant. It is not zero, though: zero is "no such
 * application", and a library that gates on the title being live then refuses
 * to come up at all.
 */

#include "base/arch.h"

#include "kern/ipmi/common_dialog.h"
#include "kern/ipmi/services.h"
#include "kern/lv2/error_table.h"

namespace kern::ipmi {
namespace {

enum {
  // out: {u64 appId, u32 state}. libSceNpWebApi's initialize reads the state
  // and returns SCE_NP_WEBAPI error 0x8055a402 unless it is 4 or 5, which fails
  // the whole NpToolkit2 bring-up above it (GTA:SA treats that as fatal).
  kLncGetAppStatus = 0x30013,
  // PS5 asks the same question with a 32-byte reply and no input;
  // system service client
  // +0xdfa0 caches it at init. A zeroed reply = appId 0, which matches no focus
  // pattern, so GetStatus reports the title overlaid by system UI and it
  // renders nothing.
  kLncGetAppStatusPs5 = 0x30010,
};

// The state that means "the title is running in the foreground". 5 is the same
// thing for a title that still owes the web API a one-time registration, so 4
// is the one that needs no further daemon round trip.
constexpr u32 kAppRunning = 4;

struct Lnc : Service {
  const char* Name() const override { return "SceLncService"; }
  i32 InvokeAsync(Invocation& inv, u32& request) override {
    return LaunchCommonDialogProcess(inv, request);
  }
  i32 GetAsyncResult(u32 client,
                     u32 method,
                     u32 request,
                     i32& result) override {
    return GetCommonDialogLaunchResult(client, method, request, result);
  }
  void DestroyClient(u32 client) override { DestroyCommonDialogClient(client); }

  void Invoke(Invocation& inv) override {
    switch (inv.Method()) {
      case kLncGetAppStatusPs5:
      case kLncGetAppStatus: {
        // system service client caches this appId against the focus flag
        // patterns; a mismatch reads "another app has focus" and titles answer
        // by dropping pad input (Tomb Raider: DE stops calling scePadReadState
        // forever).
        const u32 status[8] = {kForegroundAppId, 0, kAppRunning};
        const u64 size =
            inv.Method() == kLncGetAppStatusPs5 ? sizeof(status) : 12;
        if (!inv.CanReply(0, size)) {
          inv.SetResult(-static_cast<i32>(SysError::eINVAL));
          break;
        }
        inv.Reply(0, status, size);
        break;
      }
      default:
        if (!InvokeCommonDialog(inv))
          inv.ReplyEmpty();
        break;
    }
  }
};

Lnc g_lnc;

}  // namespace

Service& LncService() {
  return g_lnc;
}

}  // namespace kern::ipmi
