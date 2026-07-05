/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 AGC command processor. The AGC command buffer the guest libSceAgc builds
 * is a PM4 type-3 stream using the SAME IT_ opcode table as the PS4 (verified in
 * memory [[ps5-agc-gpu]]), so the packet walk + completion-label handling here
 * reuse gpu/ps4/pm4.h as-is. What differs on PS5 is the gfx10.3 register offsets
 * and the RDNA2 shader ISA (deferred): this Milestone-0 walker follows the
 * INDIRECT_BUFFER into the real DCB and services the GPU completion labels
 * (EOP / RELEASE_MEM / WRITE_DATA), which is what lets the engine's per-frame
 * command-buffer recycle fences advance. Without it every submit leaves the
 * flip/submit-done labels unwritten, so the engine keeps allocating fresh
 * command-buffer chunks until a pool goes null (the libSceAgc+0x412a NULL-DCB
 * crash).
 */

#include "cmd_processor.h"
#include "pm4.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace gpu::ps5 {
namespace {

std::mutex g_mtx;
const bool g_trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
uint64_t g_totalSubmits = 0;

// PS5 guest allocations sit anywhere across a wide VA (eboot ~0x2014_..., GPU
// dmem tagged regions 0x8000_..., doorbell 0xfe0_...). Accept any plausibly
// mapped, non-tiny address for a completion-label write; reject null/garbage.
inline bool labelAddrOk(uint64_t a) {
  return a >= 0x10000ull && a < 0x1000000000000ull;
}
void writeLabel(uint64_t addr, uint64_t value, bool is64) {
  if (!labelAddrOk(addr))
    return;
  if (is64)
    *reinterpret_cast<volatile uint64_t *>(addr) = value;
  else
    *reinterpret_cast<volatile uint32_t *>(addr) = static_cast<uint32_t>(value);
}

// EOP/RELEASE_MEM DATA_SEL 3/4 ask the GPU to write its running clock counter
// instead of the packet immediate; our submit is synchronous so any advancing,
// non-zero value reads as "already complete".
uint64_t gpuClockTs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

uint32_t g_opHist[256] = {};
int g_dumped = 0;

// Walk one PM4 stream, following INDIRECT_BUFFER into nested buffers and writing
// the completion labels. depth guards against a malformed IB self-reference.
void walk(const uint32_t *p, uint32_t words, bool dumpThis, int depth) {
  if (!p || depth > 8)
    return;
  uint32_t i = 0;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = pm4Type(hdr);
    if (type == Pm4Type::type3) {
      uint32_t op = pm4Opcode(hdr);
      uint32_t count = pm4Count(hdr);  // body dword count
      const uint32_t *body = &p[i + 1];
      if (i + 1 + count > words)
        break;  // truncated / desync
      g_opHist[op & 0xFF]++;
      if (dumpThis)
        std::fprintf(stderr, "[agc]   @%-5u T3 op=%#04x count=%u\n", i, op, count);
      switch (op) {
      case IT_INDIRECT_BUFFER: {  // body: baseLo, baseHi, sizeDwords(+flags)
        if (count >= 3) {
          uint64_t ib = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
          uint32_t ibw = body[2] & 0xFFFFF;
          if (ib >= 0x10000ull && ibw)
            walk(reinterpret_cast<const uint32_t *>(ib), ibw, dumpThis, depth + 1);
        }
        break;
      }
      case IT_WRITE_DATA: {  // control, dstLo, dstHi, data...
        if (count >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          uint32_t ndw = count - 3;
          if (labelAddrOk(addr) && labelAddrOk(addr + (uint64_t)ndw * 4))
            std::memcpy(reinterpret_cast<void *>(addr), &body[3], (size_t)ndw * 4);
        }
        break;
      }
      case IT_EVENT_WRITE_EOP: {  // eventCtrl, addrLo, addrHi+sel, dataLo, dataHi
        if (count >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          uint32_t sel = (body[2] >> 29) & 0x7;  // 1=32b 2=64b 3/4=clock
          uint64_t val = static_cast<uint64_t>(body[3]) |
                         (static_cast<uint64_t>(count >= 5 ? body[4] : 0) << 32);
          if (sel == 1) writeLabel(addr, val, false);
          else if (sel == 2) writeLabel(addr, val, true);
          else if (sel >= 3) writeLabel(addr, gpuClockTs(), true);
        }
        break;
      }
      case IT_RELEASE_MEM: {  // eventCtrl, selBits, addrLo, addrHi, dataLo, dataHi
        if (count >= 5) {
          uint32_t sel = (body[1] >> 29) & 0x7;
          uint64_t addr = (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) |
                          (body[2] & ~0x3u);
          uint64_t val = static_cast<uint64_t>(body[4]) |
                         (static_cast<uint64_t>(count >= 6 ? body[5] : 0) << 32);
          if (sel == 1) writeLabel(addr, val, false);
          else if (sel == 2) writeLabel(addr, val, true);
          else if (sel >= 3) writeLabel(addr, gpuClockTs(), true);
        }
        break;
      }
      case IT_EVENT_WRITE_EOS: {  // eventCtrl, addrLo, addrHi+cmd, data
        if (count >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          writeLabel(addr, body[3], false);
        }
        break;
      }
      default:
        break;
      }
      i += 1 + count;
    } else if (type == Pm4Type::type2 || hdr == 0) {
      i += 1;  // filler / alignment
    } else if (type == Pm4Type::type0) {
      i += 1 + pm4Count(hdr);
    } else {
      break;  // type-1 desync
    }
  }
}

}  // namespace

void submitDcb(const void *dcb, uint32_t sizeBytes) {
  if (!dcb || sizeBytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  uint32_t words = sizeBytes / 4;
  uint64_t sn = ++g_totalSubmits;
  bool dumpThis = g_trace && g_dumped < 3 && sizeBytes > 64;
  if (dumpThis) {
    g_dumped++;
    std::fprintf(stderr, "[agc] === dcb walk #%lu (size=%u words=%u hdr0=%#x) ===\n",
                 (unsigned long)sn, sizeBytes, words,
                 *static_cast<const uint32_t *>(dcb));
  }
  walk(static_cast<const uint32_t *>(dcb), words, dumpThis, 0);
  if (dumpThis) {
    std::fprintf(stderr, "[agc] === dcb walk done; opcode histogram ===\n");
    for (int o = 0; o < 256; o++)
      if (g_opHist[o])
        std::fprintf(stderr, "[agc]   op %#04x x%u\n", o, g_opHist[o]);
  }
}

void submitCcb(const void *ccb, uint32_t sizeBytes) {
  if (!ccb || sizeBytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  walk(static_cast<const uint32_t *>(ccb), sizeBytes / 4, false, 0);
}

void endFrame(uint64_t /*scanoutBase*/) {
  // Present is driven through the shared dce/VideoOut flip path (see gc_dev.cpp),
  // not from the AGC submit; nothing to do here yet for Milestone 0.
}

}  // namespace gpu::ps5

// LLE submit bridge: the kernel /dev/gc AGC ioctls (gc_dev.cpp) forward the DCB
// here, mirroring prosperity_gc_submit on the PS4 path.
extern "C" void prosperity_agc_submit(uint64_t dcbBase, uint32_t sizeBytes) {
  gpu::ps5::submitDcb(reinterpret_cast<const void *>(dcbBase), sizeBytes);
}
