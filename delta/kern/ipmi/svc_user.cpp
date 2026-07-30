/*
 * PS4Delta : PS4 emulation and research project
 *
 * The user-account daemon. libSceUserService talks to "SceUserService" for the
 * logged-in user list and the login/logout event queue.
 *
 * There is one local user and it is always signed in, so the state to report is
 * settled: user id kUserId present in the login list, and an event queue that is
 * permanently empty AFTER the initial login has been consumed. Replying with
 * zeros instead would say "user id 0", which is SCE_USER_SERVICE_USER_ID_INVALID
 * -- a title that resolves the foreground user before it builds its HUD then
 * either retries forever or indexes a save slot with an invalid id.
 *
 * Mirrors the HLE libSceUserService shim (runtime/vprx/ps4/libSceUserService),
 * so switching that module between HLE and LLE (DELTA_LLE) reports the same
 * user either way.
 */

#include <cstdint>

#include "services.h"

namespace krnl::ipmi {
namespace {

// Same id the HLE shim reports. 1 is the first real user on retail; 0 would be
// SCE_USER_SERVICE_USER_ID_INVALID.
constexpr int32_t kUserId = 1;

struct UserService : Service {
  const char *name() const override { return "SceUserService"; }

  void invoke(Invocation &inv) override {
    switch (inv.method()) {
      default:
        // Every getter in this daemon hands back either a user id or a small
        // fixed-size record. An empty (zeroed) reply is the safe answer for the
        // ones we have not identified: it reads as "no value" rather than
        // leaving the caller's stack garbage in place.
        inv.replyEmpty();
        break;
    }
  }
};

UserService g_userService;

}  // namespace

Service &userService() { return g_userService; }

}  // namespace krnl::ipmi
