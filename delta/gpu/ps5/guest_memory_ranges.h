#pragma once
#include <string_view>
#include "base/containers/vector.h"
#include "gpu/render/command.h"

namespace gpu::ps5 {
// A readable line of /proc/self/maps.
struct HostMapping {
  unsigned long long begin, end, offset, inode;
  unsigned major, minor;
  char perm[5];
};
bool ParseHostMapping(std::string_view line, HostMapping& mapping);
// Parsed at most once a frame, or again after MarkHostMappingsStale.
const base::Vector<HostMapping>& HostMappings();
// Thread safe: the next HostMappings() parses again.
void MarkHostMappingsStale();
// Advances each time HostMappings() parses.
u64 HostMappingsVersion();
// Whether the guest can write [base, base+bytes): every byte in mappings
// that are writable (or shared, file-backed: direct memory, whatever the
// write tracker's protection says).
bool IsGuestWritable(u64 base, u64 bytes);
// Every remap will be reported through MarkHostMappingsStale: stop parsing
// again once a frame.
void ReportRemapsToHostMappings();

// Readable mappings inside the GPU pools, plus mappings containing explicit
// raw-resource addresses outside those pools (for example module constants).
base::Vector<render::GuestMemoryRange> GuestMemoryRanges(
    const base::Vector<u64>& addresses);
}  // namespace gpu::ps5
