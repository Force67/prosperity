#pragma once

namespace gpu::ps5 {
// The write tracker sees writes only through the mapping it protected, and
// direct memory is one memfd mapped many times.
void InstallWriteTrackerPolicy();
}  // namespace gpu::ps5
