#pragma once

namespace gpu::ps5 {
// The write tracker sees writes through the mapping it protected, only. PS5
// direct memory is one shared memfd mapped at any number of addresses, so
// memory mapped at two addresses is never armed, and a remap that makes armed
// memory an alias reports it as written.
void InstallWriteTrackerPolicy();
}  // namespace gpu::ps5
