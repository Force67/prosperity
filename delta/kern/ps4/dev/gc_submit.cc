#include "kern/ps4/dev/gc_submit.h"

#include "base/logging.h"
#include "gpu/ps4/cmd_processor.h"
#include "host_memory/host_memory.h"
#include "kern/probe/probe_arm.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kGcSubmit, "DELTA_GC_SUBMIT", false);
DELTA_OPTION(u64, kGcSubmitMax, "DELTA_GC_SUBMIT_MAX", 0);
DELTA_OPTION(bool, kWaitWatch, "DELTA_GPU_WAITWATCH", false);
}  // namespace

namespace kern {
void SubmitGcDescriptors(const void* desc_array, u32 desc_count) {
  static const bool hooked = [] {
    if (kWaitWatch)
      gpu::ps4::SetUnknownWaitHook([](u64 address) {
        kern::probe::StartWriteWatch(address, 4, 0, /*trap_reads=*/false,
                                     /*single_step=*/true);
      });
    return true;
  }();
  (void)hooked;
  auto* d = static_cast<const u32*>(desc_array);
  if (!d)
    return;
  // Guest bug shields: submits arrive with guest-controlled pointers. A stray
  // descriptor array (or a descriptor whose IB address is garbage) must be
  // dropped loudly, not dereferenced. SotC's debug menu produced a submit
  // whose descPtr pointed at unmapped host space and SIGSEGV'd the CP.
  if (desc_count > 4096 ||
      !host_memory::IsMemoryRangeMapped(desc_array, desc_count * 16ull)) {
    static int warned = 0;
    if (warned++ < 8)
      BASE_LOGI("gc", "DROPPED bad submit descArray={:p} count={}", desc_array,
                desc_count);
    return;
  }
  // A dozen lines answers "what does a submit look like"; comparing the set of
  // submitted buffers against something else (which command buffers a title
  // BUILDS but never submits) needs the whole run.
  static int submit_dumps = 0;
  const bool dump_this =
      kGcSubmit && (kGcSubmitMax == 0 ? submit_dumps++ < 12
                                      : submit_dumps++ < (int)kGcSubmitMax);
  if (dump_this)
    BASE_LOGI("gc", "submit descArray={:p} count={}", desc_array, desc_count);
  for (u32 i = 0; i < desc_count; i++) {
    const u32* e = d + i * 4;
    u32 hdr = e[0];
    u64 addr = (static_cast<u64>(e[2] & 0xFF) << 32) | e[1];
    u32 bytes = (e[3] & 0xFFFFF) * 4;  // ib_size is in dwords
    if (dump_this)
      BASE_LOGI(
          "gc",
          "  desc[{}]: {:08x} {:08x} {:08x} {:08x} -> addr={:#x} bytes={}", i,
          e[0], e[1], e[2], e[3], (unsigned long)addr, bytes);
    if (!addr || !bytes)
      continue;
    if (!host_memory::IsMemoryRangeMapped(reinterpret_cast<const void*>(addr),
                                          bytes)) {
      static int warned = 0;
      if (warned++ < 8)
        BASE_LOGI("gc", "DROPPED bad IB addr={:#x} bytes={}",
                  (unsigned long)addr, bytes);
      continue;
    }
    if (hdr == 0xC0023300u)
      gpu::ps4::SubmitCcb(reinterpret_cast<const void*>(addr), bytes);
    else if (hdr == 0xC0023F00u)
      gpu::ps4::SubmitDcb(reinterpret_cast<const void*>(addr), bytes);
  }
}
}  // namespace kern
