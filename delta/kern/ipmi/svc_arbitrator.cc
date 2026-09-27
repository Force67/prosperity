/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PS5 resource-arbitrator daemon. libSceResourceArbitrator's worker thread
 * (SceResourceArbitratorWorker) blocks on an event flag, then drains whatever
 * arrived through method 0x32: a byte in, a 0xc8-byte record out whose u32 key
 * at +0 selects a registered callback and whose cookie at +0xc0 must match the
 * one that registered it.
 *
 * The drain is NOT the blocking call, so answering it with success means "here
 * is an event" and the worker immediately drains again: replying empty span one
 * core at ~160k calls/s, and parking first only slowed that to ~700/s. An empty
 * queue is reported by FAILING the method, which sends the worker back to its
 * event-flag wait. Anything a client did register is unaffected, and no title
 * we run registers anything: Demon's Souls pulls the library in through
 * libSceSysmodule's preload list without importing a single one of its exports.
 *
 * Reporting the empty queue by FAILING the method (0x80020002) is what the
 * disassembly suggests, and it is much worse in practice: the worker retries
 * the drain immediately rather than going back to its event-flag wait, and the
 * count goes from 28k per run to 14.9 million. Parking before an empty success
 * is the cheapest state we have found. None of it moves the title's stall.
 */

#include "base/arch.h"

#include "base/threading/thread.h"
#include "kern/ipmi/services.h"

namespace kern::ipmi {
namespace {

enum { kDrainNotification = 0x32 };

struct ArbitratorIpc : Service {
  const char* Name() const override { return "SceArbitratorIpc"; }

  void Invoke(Invocation& inv) override {
    if (inv.Method() == kDrainNotification)
      base::SleepForMilliseconds(10);
    inv.ReplyEmpty();
  }
};

ArbitratorIpc g_arbitrator_ipc;

}  // namespace

Service& ArbitratorIpcService() {
  return g_arbitrator_ipc;
}

}  // namespace kern::ipmi
