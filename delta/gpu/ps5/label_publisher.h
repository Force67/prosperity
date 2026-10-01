#pragma once

#include "base/functional/function.h"

namespace gpu::ps5 {
// Fence writes the guest waits on (EOP/RELEASE_MEM labels, their interrupts,
// WRITE_DATA) are held back while GPU writes into guest memory are on their
// way (render::GuestWritePublishBatch): the guest must not see a label before
// the bytes it announces have landed. A held write runs on a publishing
// thread once the GPU is past them, in the order it was issued.
//
// Runs `write` at once when nothing is pending and nothing is held.
void PublishLabel(base::Function<void()> write);
// Returns once every held write has run: for the walk, before it reads guest
// memory a held write may target.
void DrainLabels();
// What the held writes wait for (a stuck waitOnAddress reports it).
void ReportHeldLabels();
}  // namespace gpu::ps5
