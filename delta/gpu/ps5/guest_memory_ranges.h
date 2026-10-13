#pragma once
#include <string_view>
#include "base/containers/vector.h"
#include "gpu/render/command.h"

namespace gpu::ps5 {
struct HostMapping {
  unsigned long long begin, end, offset, inode;
  unsigned major, minor;
  char perm[5];
};
bool ParseHostMapping(std::string_view line, HostMapping& mapping);
const base::Vector<HostMapping>& HostMappings();
void MarkHostMappingsStale();
u64 HostMappingsVersion();
bool IsGuestWritable(u64 base, u64 bytes);
void ReportRemapsToHostMappings();

base::Vector<render::GuestMemoryRange> GuestMemoryRanges(
    const base::Vector<u64>& addresses);
}  // namespace gpu::ps5
