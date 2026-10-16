#pragma once
#include "base/time/time.h"

#include "base/arch.h"
#include "base/functional/function.h"

namespace gpu::ps5 {
void PublishLabel(base::Function<void()> write,
                  u64 base = 0,
                  u64 bytes = ~0ull,
                  const void* data = nullptr,
                  bool graphics_only = false);
// `ring` readers are ring queues, which see through graphics-only holds.
bool OverlayHeld(u64 base, u32 bytes, u8* out, bool ring = false);
bool LabelPending(u64 base, u64 bytes, bool ring = false);
u64 LabelSequence();
void NoteLabelWritten();
void WaitLabelWritten(u64 seen, base::TimeDelta timeout);
void ReportHeldLabels();
}  // namespace gpu::ps5
