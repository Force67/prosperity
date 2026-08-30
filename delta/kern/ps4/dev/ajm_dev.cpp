/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <cstdlib>
#include <cstring>
#include "ajm_dev.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kAjmResult, "DELTA_AJM_RESULT", nullptr);
DELTA_OPTION(bool, kAjmTrace, "DELTA_AJM_TRACE", false);
}  // namespace

namespace krnl {
ajmDevice::ajmDevice(objectTable &objects) : device(objects) {}

// AJM ioctls (sizes encoded in the command). We don't decode audio; we only need
// libSceAjm's init handshake to report success. Return 0 (success) and hand back
// a benign non-zero context/instance id where the caller reads one back, so the
// register/create steps don't look like failures.
i32 ajmDevice::ioctl(u32 cmd, void *data) {
  if (kAjmTrace) {
    // ioctl size is encoded in bits [29:16] of the command.
    u32 sz = (cmd >> 16) & 0x3FFF;
    base::String bytes;
    base::FormatTo(bytes, "ioctl({:#x}) sz={} data={:p} in:", cmd, sz, data);
    if (data && sz && sz <= 256) {
      const u8 *b = static_cast<const u8 *>(data);
      for (u32 i = 0; i < sz; i++) base::FormatTo(bytes, " {:02x}", b[i]);
    }
    BASE_LOGI("ajm", "{}", bytes.c_str());
  }
  // Most AJM control ioctls pass a struct whose first dword(s) are an
  // out-parameter (instance/module id or a result code). Leave a zeroed result
  // (= AJM_RESULT OK) but provide a non-zero id so a "create" call's handle is
  // usable rather than 0.
  if (cmd == 0xC0288903 && data) {
    // AJM batch ioctl: a 40-byte arg struct. Layout (observed from FMOD):
    //   +0  context/instance id     +4  input batch size
    //   +8  command count           +0xc output buffer size
    //   +0x18 input batch ptr       +0x20 output buffer ptr
    struct AjmBatch {
      u32 ctx, inSize, count, outSize, r0, r1;
      u64 inPtr, outPtr;
    };
    auto *b = static_cast<AjmBatch *>(data);
    if (kAjmTrace) {
      BASE_LOGI("ajm",
                "batch ctx={:#x} inSize={} count={} outSize={} in={:#x} out={:#x}",
                b->ctx, b->inSize, b->count, b->outSize,
                (unsigned long)b->inPtr, (unsigned long)b->outPtr);
      auto hexdump = [](const char *tag, u64 p, u32 n) {
        if (!p) return;
        base::String bytes;
        base::FormatTo(bytes, "  {}:", tag);
        const u8 *q = reinterpret_cast<const u8 *>(p);
        for (u32 i = 0; i < n; i++) base::FormatTo(bytes, " {:02x}", q[i]);
        BASE_LOGI("ajm", "{}", bytes.c_str());
      };
      hexdump("inbatch", b->inPtr, 64);
      hexdump("out(pre)", b->outPtr, b->outSize <= 128 ? b->outSize : 128);
    }
    // NOTE: FMOD's first AJM batch registers its "FMOD DSP Codec AT9" codec and
    // reads a full codec DESCRIPTOR back from the output buffer (incl. function
    // pointers it calls during init). A zeroed result makes FMOD accept the
    // register but then crash calling a null descriptor fn; a non-zeroed (stale)
    // result makes FMOD report FMOD_ERR_INTERNAL. Getting past this needs a real
    // (or convincingly faked) ATRAC9/AJM descriptor -- see DELTA_AJM_RESULT probe.
    // EXPERIMENT (DELTA_AJM_RESULT=N): write a result pattern to the batch output
    // so FMOD's codec-register init accepts it. N selects the pattern.
    if (const char *e = kAjmResult;
        e && b->outPtr && b->outSize && b->outSize <= 0x1000) {
      int n = std::atoi(e);
      auto *o32 = reinterpret_cast<u32 *>(b->outPtr);
      // Minimal writes (no full-buffer memset; 0x40 over-runs the guest frame).
      if (n == 1) { o32[0] = 0; }                  // only result code = OK
      else if (n == 2) { o32[0] = 0; o32[1] = 1; } // result OK + handle 1
      else if (n == 3) { o32[0] = 0; o32[1] = 0; o32[2] = 1; }
      else if (n == 4) { o32[0] = 0; o32[1] = b->ctx; }
      else if (n >= 10) { // memset n*4 bytes then result 0 (size probe)
        std::memset(reinterpret_cast<void *>(b->outPtr), 0, (size_t)(n - 10) * 4);
      }
    }
    return 0;
  }
  if (cmd == 0xC0288001 || cmd == 0xC0208016) {
    if (data)
      *static_cast<u32 *>(data) = 1; // a valid (non-zero) id
  }
  return 0;
}
} // namespace krnl
