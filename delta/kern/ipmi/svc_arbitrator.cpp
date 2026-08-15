/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PS5 resource-arbitrator daemon. libSceResourceArbitrator's worker
 * thread drains notifications through method 0x32 (1 byte in, a 0xc8-byte
 * record out: u32 key at +0, u64 callback cookie at +0xc0) and re-invokes for
 * as long as the call succeeds, so the instant default reply spun one core at
 * ~160k calls/s. Parking the caller before answering keeps that loop cold,
 * and the zeroed record's cookie matches no registered callback. The same
 * method also serves the synchronous state query, which a parked empty reply
 * still answers correctly ("current state, nothing pending").
 */

#include <chrono>
#include <thread>

#include "base/arch.h"

#include "services.h"

namespace krnl::ipmi {
namespace {

enum { kDrainNotification = 0x32 };

struct ArbitratorIpc : Service {
  const char *name() const override { return "SceArbitratorIpc"; }

  void invoke(Invocation &inv) override {
    if (inv.method() == kDrainNotification)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    inv.replyEmpty();
  }
};

ArbitratorIpc g_arbitratorIpc;

} // namespace

Service &arbitratorIpcService() { return g_arbitratorIpc; }

} // namespace krnl::ipmi
