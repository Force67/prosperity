/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceGnmDriver: the GPU command-submission entry points only.
 *
 * The game builds PM4 command buffers (which we keep LLE). The real Sony submit
 * path hands them to a GPU command processor backed by hardware we don't have,
 * so we HLE just the submit/flip/done entry points (the graphics exception to
 * the keep-PRX-LLE rule). Each submit feeds the dcb/ccb to our GCN->Vulkan
 * command processor (gpu::ps4::SubmitDcb/submitCcb -> PM4 decode -> recompiled
 * shaders -> the headless Vulkan renderer); the *AndFlip variants additionally
 * end the frame and drive the flip (present + flip event) through the VideoOut
 * HLE.
 *
 * The non-submitting entry points (SubmitDone/AreSubmitsAllowed/DingDong/
 * FlushGarlic/InsertWaitFlipDone) are intentional no-ops: our submit is
 * synchronous (the draws are already rendered when the call returns), so there
 * is no async ring to ding-dong, no EOP to wait on, and no Garlic write-combine
 * buffer to flush.
 */

#include "runtime/vprx/ps4/lib_sce_gnm_driver/lib_sce_gnm_driver.h"
#include "base/arch.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include <cstdio>
#include <cstdlib>

#include "base/logging.h"

#include "gpu/ps4/cmd_processor.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kDingDong, "DELTA_GPU_DINGDONG", false);
DELTA_OPTION(bool, kGcSubmit, "DELTA_GC_SUBMIT", false);
// Dwords of async-compute ring executed per queue per FRAME (0 = off).
DELTA_OPTION(u32, kGcAcbFrame, "DELTA_GPU_ACB_FRAME", 0);
DELTA_OPTION(u64, kGcSubmitMax, "DELTA_GC_SUBMIT_MAX", 0);
DELTA_OPTION(bool, kPm4dump, "DELTA_PM4DUMP", false);
}  // namespace

// VideoOut HLE flip bridge (same delta_runtime library).
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_videoout_set_flip(int buffer_index, i64 flip_arg);
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" u64 prosperity_videoout_buffer(int buffer_index);

// LLE submit bridge: the REAL libSceGnmDriver.sprx (the default now; the HLE
// submit shim below is only used when DELTA_GNM_HLE forces it on) builds PM4
// and submits it through ioctl(/dev/gc, ...) instead of calling the HLE entry
// points below. The gc char device forwards those submit ioctls here. The ioctl
// payload is an array of 16-byte PM4 INDIRECT_BUFFER descriptors, each {header,
// base_lo, base_hi_byte, size_in_dwords}: header 0xC0023300 =
// IT_INDIRECT_BUFFER_CNST (the ccb) and 0xC0023F00 = IT_INDIRECT_BUFFER (the
// dcb). Decode them to the same submitCcb/submitDcb path the HLE entry points
// use. (Layout verified against the 11.00 kernel gc_submit_internal and the
// sprx submit wrappers.)
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_submit(const void* desc_array, u32 desc_count) {
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
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_submit_acb(const void* commands, u32 bytes) {
  gpu::ps4::SubmitDcb(commands, bytes);
}
// NOLINTEND(readability-identifier-naming)

// LLE flip bridge: /dev/dce owns display-buffer registration and supplies the
// selected scanout address to /dev/gc when the real GnmDriver submits the
// frame. endFrame falls back to the last RT if the address was not registered.
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_drain_acb(u32 budget_dw);
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_dingdong(u32 ring_id, u32 offset_dw);

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_flip(u64 scanout_base,
                                   int display_buffer_index,
                                   i64 flip_arg) {
  // Execute the async-compute rings once per frame, before the frame ends.
  // Draining them inside the DingDong handler instead charges a whole backlog
  // to whichever frame rang the doorbell.
  prosperity_gc_drain_acb(kGcAcbFrame);
  gpu::ps4::EndFrame(scanout_base);
  if (display_buffer_index >= 0)
    prosperity_videoout_set_flip(display_buffer_index, flip_arg);
}
// NOLINTEND(readability-identifier-naming)

namespace {
// Feed each command buffer to the GPU command processor. The Constant Engine
// runs ahead of the Draw Engine, so process a submit's ccb (CE RAM -> shader
// constant buffers) before its dcb draws.
void ProcessDcbs(void** dcb_gpu_addrs,
                 u32* dcb_sizes,
                 void** ccb_gpu_addrs,
                 u32* ccb_sizes,
                 u32 count) {
  if (!dcb_gpu_addrs || !dcb_sizes)
    return;
  for (u32 i = 0; i < count; i++) {
    if (ccb_gpu_addrs && ccb_sizes && ccb_gpu_addrs[i] && ccb_sizes[i])
      gpu::ps4::SubmitCcb(ccb_gpu_addrs[i], ccb_sizes[i]);
    gpu::ps4::SubmitDcb(dcb_gpu_addrs[i], dcb_sizes[i]);
  }
}
}  // namespace

namespace {
// Env-gated PM4 dump (DELTA_PM4DUMP=1): walk the dcb type-3 packets and tally
// the IT opcodes so we can see what the title actually submits. Guest GPU
// addresses are identity-mapped, so the dcb is directly readable on the host.
int g_pm4_frames = 0;

const char* ItName(u32 op) {
  switch (op) {
    case 0x10:
      return "NOP";
    case 0x12:
      return "CLEAR_STATE";
    case 0x22:
      return "COND_EXEC";
    case 0x28:
      return "INDEX_BASE";
    case 0x2A:
      return "INDEX_BUFFER_SIZE";
    case 0x2D:
      return "DRAW_INDEX_AUTO";
    case 0x27:
      return "DRAW_INDEX_2";
    case 0x2F:
      return "DRAW_INDEX_OFFSET_2";
    case 0x36:
      return "WAIT_REG_MEM";
    case 0x37:
      return "WRITE_DATA";
    case 0x3C:
      return "ACQUIRE_MEM";
    case 0x40:
      return "DMA_DATA";
    case 0x46:
      return "EVENT_WRITE";
    case 0x47:
      return "EVENT_WRITE_EOP";
    case 0x49:
      return "RELEASE_MEM";
    case 0x4C:
      return "DISPATCH_DIRECT";
    case 0x68:
      return "SET_CONFIG_REG";
    case 0x69:
      return "SET_CONTEXT_REG";
    case 0x76:
      return "SET_SH_REG";
    case 0x79:
      return "SET_UCONFIG_REG";
    default:
      return "?";
  }
}

void DumpPm4(void** dcb_gpu_addrs, u32* dcb_sizes, u32 count) {
  if (!kPm4dump || g_pm4_frames > 3 || !dcb_gpu_addrs || !dcb_sizes)
    return;
  g_pm4_frames++;
  for (u32 b = 0; b < count; b++) {
    auto* p = static_cast<u32*>(dcb_gpu_addrs[b]);
    u32 words = dcb_sizes[b] / 4;
    BASE_LOGI("pm4", "dcb[{}] @{:p} words={}", b, p, words);
    if (!p)
      continue;
    u32 i = 0, draws = 0;
    while (i < words) {
      u32 hdr = p[i];
      u32 type = hdr >> 30;
      u32 cnt = ((hdr >> 16) & 0x3FFF) + 1;  // dword count after header
      if (type == 3) {
        u32 op = (hdr >> 8) & 0xFF;
        BASE_LOGI("pm4", "  T3 {:<20} op={:#04x} cnt={}", ItName(op), op, cnt);
        if (op == 0x2D || op == 0x27 || op == 0x2F || op == 0x4C)
          draws++;
        i += 1 + cnt;
      } else if (type == 2) {
        i += 1;  // filler
      } else if (type == 0) {
        i += 1 + cnt;
      } else {
        break;  // type 1 / desync
      }
    }
    BASE_LOGI("pm4", "  -> {} draw/dispatch packets", draws);
  }
}
}  // namespace

extern "C" {

int PS4ABI sceGnmSubmitCommandBuffers(u32 count,
                                      void** dcb_gpu_addrs,
                                      u32* dcb_sizes,
                                      void** ccb_gpu_addrs,
                                      u32* ccb_sizes) {
  DumpPm4(dcb_gpu_addrs, dcb_sizes, count);
  ProcessDcbs(dcb_gpu_addrs, dcb_sizes, ccb_gpu_addrs, ccb_sizes, count);
  return 0;
}

int PS4ABI sceGnmSubmitCommandBuffersForWorkload(u32 workload,
                                                 u32 count,
                                                 void** dcb_gpu_addrs,
                                                 u32* dcb_sizes,
                                                 void** ccb_gpu_addrs,
                                                 u32* ccb_sizes) {
  // Same as sceGnmSubmitCommandBuffers but tagged with a workload id; the
  // command buffers must still be processed (this was stubbed, silently
  // dropping every draw the game submitted through the workload path).
  DumpPm4(dcb_gpu_addrs, dcb_sizes, count);
  ProcessDcbs(dcb_gpu_addrs, dcb_sizes, ccb_gpu_addrs, ccb_sizes, count);
  return 0;
}

int PS4ABI sceGnmSubmitAndFlipCommandBuffers(u32 count,
                                             void** dcb_gpu_addrs,
                                             u32* dcb_sizes,
                                             void** ccb_gpu_addrs,
                                             u32* ccb_sizes,
                                             u32 video_out_handle,
                                             u32 display_buffer_index,
                                             u32 flip_mode,
                                             i64 flip_arg) {
  // The flip target buffer is what should be scanned out next; record it so the
  // VideoOut flip pump presents it and posts the flip-complete event.
  DumpPm4(dcb_gpu_addrs, dcb_sizes, count);
  ProcessDcbs(dcb_gpu_addrs, dcb_sizes, ccb_gpu_addrs, ccb_sizes, count);
  // Present the render target that this flip displays (the guest scanout
  // buffer).
  gpu::ps4::EndFrame(
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
      prosperity_videoout_buffer(static_cast<int>(display_buffer_index)));
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
  prosperity_videoout_set_flip(static_cast<int>(display_buffer_index),
                               flip_arg);
// NOLINTEND(readability-identifier-naming)
  return 0;
}

int PS4ABI
sceGnmSubmitAndFlipCommandBuffersForWorkload(u32 workload,
                                             u32 count,
                                             void** dcb_gpu_addrs,
                                             u32* dcb_sizes,
                                             void** ccb_gpu_addrs,
                                             u32* ccb_sizes,
                                             u32 video_out_handle,
                                             u32 display_buffer_index,
                                             u32 flip_mode,
                                             i64 flip_arg) {
  // Was a flip-only stub that dropped the submitted command buffers. Process
  // them (and end the frame on the flip) exactly like the non-workload variant.
  DumpPm4(dcb_gpu_addrs, dcb_sizes, count);
  ProcessDcbs(dcb_gpu_addrs, dcb_sizes, ccb_gpu_addrs, ccb_sizes, count);
  gpu::ps4::EndFrame(
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
      prosperity_videoout_buffer(static_cast<int>(display_buffer_index)));
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
  prosperity_videoout_set_flip(static_cast<int>(display_buffer_index),
                               flip_arg);
// NOLINTEND(readability-identifier-naming)
  return 0;
}

int PS4ABI sceGnmSubmitDone() {
  return 0;
}

int PS4ABI sceGnmAreSubmitsAllowed() {
  return 1;
}

// The doorbell. Under LLE this is the real driver storing `offset` into its
// /dev/gc mapping, which tells us nothing, so force this ONE nid onto the shim
// (DELTA_HLE_NIDS_GNM) and the ring runs at the only moment its contents are
// known good. See prosperity_gc_dingdong.
int PS4ABI sceGnmDingDong(u32 ring_id, u32 offset) {
  static int n = 0;
  if (kDingDong && n++ < 20)
    BASE_LOGI("gnm", "sceGnmDingDong ring={} offset={:#x}", ring_id, offset);
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
  prosperity_gc_dingdong(ring_id, offset);
  return 0;
}

int PS4ABI sceGnmDingDongForWorkload(u32 workload, u32 ring_id, u32 offset) {
  return 0;
}

int PS4ABI sceGnmFlushGarlic() {
  return 0;
}

int PS4ABI sceGnmInsertWaitFlipDone(void* cmd_buffer,
                                    u32 size,
                                    u32 video_out_handle,
                                    u32 buffer_index) {
  return 0;
}

}  // extern "C"
