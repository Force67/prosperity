/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 AGC command processor (scaffold). Unimplemented; see cmd_processor.h.
 */

#include "cmd_processor.h"

#include <cstdio>

namespace gpu::ps5 {

static void notImplemented() {
  static bool warned = false;
  if (!warned) {
    warned = true;
    std::fprintf(stderr, "[gpu::ps5] PS5 AGC command processor not implemented\n");
  }
}

void submitDcb(const void * /*dcb*/, uint32_t /*sizeBytes*/) { notImplemented(); }

void submitCcb(const void * /*ccb*/, uint32_t /*sizeBytes*/) { notImplemented(); }

void endFrame(uint64_t /*scanoutBase*/) { notImplemented(); }

}  // namespace gpu::ps5
