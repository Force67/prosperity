#pragma once

namespace cli {

// Logs the session's memory footprint every DELTA_MEMORY_REPORT seconds
// (guest direct/flexible, host resident, emulator heap and GPU allocations)
// against the console it emulates, and warns once per GiB over that budget.
void StartMemoryReport(bool ps5);

}  // namespace cli
