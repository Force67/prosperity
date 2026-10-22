#include "cpu/aarch64/crossing_trace.h"

#include <cstdio>

#include "cpu/aarch64/code_map.h"
#include "cpu/backend.h"
#include "kern/lv2/dispatch.h"

namespace cpu::fex {
namespace {

thread_local TraceEvt t_trace[kTraceRing];
thread_local u32 t_trace_pos = 0;

thread_local u32 t_last_syscall = 0xFFFFFFFFu;
thread_local bool t_in_syscall = false;

}  // namespace

TraceEvt& TraceNext() {
  TraceEvt& e = t_trace[t_trace_pos % kTraceRing];
  t_trace_pos++;
  return e;
}

void BeginSyscall(u32 num) {
  t_last_syscall = num;
  t_in_syscall = true;
}

void EndSyscall() {
  t_in_syscall = false;
}

const TraceEvt* ThreadTraceRing() {
  return t_trace;
}

const u32* ThreadTracePos() {
  return &t_trace_pos;
}

const bool* ThreadInSyscall() {
  return &t_in_syscall;
}

}  // namespace cpu::fex

namespace cpu {

int FaultingSyscall() {
  return fex::t_in_syscall ? static_cast<int>(fex::t_last_syscall) : -1;
}

void DumpThreadTrace(void* file_star) {
  auto* f = static_cast<std::FILE*>(file_star);
  if (!f)
    return;
  std::fprintf(
      f, "  --- last guest->host calls (this thread, oldest first) ---\n");
  const u32 pos = fex::t_trace_pos;
  u32 count = pos < fex::kTraceRing ? pos : fex::kTraceRing;
  u32 start = pos - count;
  for (u32 i = 0; i < count; i++) {
    const fex::TraceEvt& e = fex::t_trace[(start + i) % fex::kTraceRing];
    if (e.kind == 's') {
      std::fprintf(f, "  sc  %3u %-22s (%#llx,%#llx,%#llx,%#llx) -> %#llx\n",
                   e.id, kern::SyscallGetname(e.id), (unsigned long long)e.a0,
                   (unsigned long long)e.a1, (unsigned long long)e.a2,
                   (unsigned long long)e.a3, (unsigned long long)e.ret);
    } else if (e.kind == 'h') {
      char cs[256];
      fex::SymbolizeModuleAddress(e.caller, cs, sizeof(cs));
      std::fprintf(f, "  hle %s(%#llx,%#llx,%#llx,%#llx) -> %#llx  from %s\n",
                   e.name ? e.name : "?", (unsigned long long)e.a0,
                   (unsigned long long)e.a1, (unsigned long long)e.a2,
                   (unsigned long long)e.a3, (unsigned long long)e.ret, cs);
    }
  }
}

}  // namespace cpu
