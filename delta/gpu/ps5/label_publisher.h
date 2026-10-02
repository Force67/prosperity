#pragma once

#include "base/arch.h"
#include "base/functional/function.h"

namespace gpu::ps5 {
// Fence writes the guest waits on (EOP/RELEASE_MEM labels, their interrupts,
// WRITE_DATA) are held back while GPU writes into guest memory are on their
// way (render::GuestWritePublishBatch): the guest must not see a label before
// the bytes it announces have landed. A held write runs on a publishing
// thread once the GPU is past them, in the order it was issued.
//
// Runs `write` at once when nothing is pending and nothing is held. It writes
// [base, base+bytes), `data` when given; the default is "anywhere".
void PublishLabel(base::Function<void()> write,
                  u64 base = 0,
                  u64 bytes = ~0ull,
                  const void* data = nullptr);
// [base, base+bytes) as it reads once the held writes have run, laid over
// `out` (which holds the memory now). False when a held write over it does
// not say what it writes. Another queue may go by this: everything it records
// after the wait runs on the GPU after the work the held write waits for.
bool OverlayHeld(u64 base, u32 bytes, u8* out);
// Whether a held write into [base, base+bytes) has yet to run: a wait on it is
// not satisfied yet, and its batch is put on the queue. Waiting for the held
// writes instead blocked every waitOnAddress check on the GPU, with the GPU
// lock held.
bool LabelPending(u64 base, u64 bytes);
// What the held writes wait for (a stuck waitOnAddress reports it).
void ReportHeldLabels();
}  // namespace gpu::ps5
