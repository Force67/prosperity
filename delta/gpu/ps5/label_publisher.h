#pragma once
#include "base/time/time.h"

#include "base/arch.h"
#include "base/functional/function.h"

namespace gpu::ps5 {
// The guest must not see a label before the GPU is past the work that writes
// it.
void PublishLabel(base::Function<void()> write,
                  u64 base = 0,
                  u64 bytes = ~0ull,
                  const void* data = nullptr);
bool OverlayHeld(u64 base, u32 bytes, u8* out);
bool LabelPending(u64 base, u64 bytes);
u64 LabelSequence();
void NoteLabelWritten();
void WaitLabelWritten(u64 seen, base::TimeDelta timeout);
void ReportHeldLabels();
}  // namespace gpu::ps5
