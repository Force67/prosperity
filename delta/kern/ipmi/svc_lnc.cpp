/*
 * PS4Delta : PS4 emulation and research project
 *
 * The launch daemon. ShellCore owns the lifecycle of the running title, and
 * libSceLncUtil / libSceSystemService ask "SceLncService" what state it is in.
 *
 * We only ever run one title, in the foreground, for the whole process
 * lifetime, so the answer is constant. It is not zero, though: zero is "no such
 * application", and a library that gates on the title being live then refuses
 * to come up at all.
 */

#include <cstdint>

#include "services.h"

namespace krnl::ipmi {
namespace {

enum {
  // out: {u64 appId, u32 state}. libSceNpWebApi's initialize reads the state
  // and returns SCE_NP_WEBAPI error 0x8055a402 unless it is 4 or 5, which fails
  // the whole NpToolkit2 bring-up above it (GTA:SA treats that as fatal).
  kLncGetAppStatus = 0x30013,
};

// The state that means "the title is running in the foreground". 5 is the same
// thing for a title that still owes the web API a one-time registration, so 4
// is the one that needs no further daemon round trip.
constexpr uint32_t kAppRunning = 4;

struct Lnc : Service {
  const char *name() const override { return "SceLncService"; }

  void invoke(Invocation &inv) override {
    switch (inv.method()) {
    case kLncGetAppStatus: {
      // libSceSystemService caches the appId from this reply and compares it
      // against the SceShellCoreUtilAppFocus/CtrlFocus flag patterns; a
      // mismatch reads as "another app has focus" and GetStatus reports
      // isInBackgroundExecution, which titles answer by dropping pad input
      // (Tomb Raider: DE stops calling scePadReadState forever).
      const uint32_t status[3] = {kForegroundAppId, 0, kAppRunning};
      inv.reply(0, status, sizeof(status));
      break;
    }
    default:
      inv.replyEmpty();
      break;
    }
  }
};

Lnc g_lnc;

} // namespace

Service &lncService() { return g_lnc; }

} // namespace krnl::ipmi
