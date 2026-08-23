/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * See wait_probe.cpp. DELTA_WAIT_PROBE reports guest threads parked in a wait
 * for longer than a couple of seconds -- the signature of a stalled title,
 * which a CPU profiler cannot show because a blocked thread burns no cycles.
 */

#pragma once

namespace krnl {

void waitProbeEnter(const char* what, long a0, long a1);
void waitProbeExit();

// What is guest thread `gtid` parked in? Fills `out` with e.g.
// "osem_wait(0x118) for 42s" and returns false when that thread is not in a
// probed wait at all. Lets a stalled lock name what its OWNER is stuck on,
// which is the half of a deadlock the waiter cannot see.
bool waitProbeDescribeGuest(unsigned gtid, char* out, unsigned long len);

// Scoped form: records the wait for the lifetime of the object.
struct WaitProbe {
  bool on;
  explicit WaitProbe(const char* what, long a0 = 0, long a1 = 0);
  ~WaitProbe();
};

}  // namespace krnl
