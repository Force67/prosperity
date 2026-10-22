#include "cpu/aarch64/code_map.h"

#include <cstdio>

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"

namespace cpu::fex {
namespace {

base::Mutex g_range_mutex;
base::Vector<ExecRange> g_ranges;

struct NamedRange {
  u64 base, size;
  base::String name;
};
base::Mutex g_named_mutex;
base::Vector<NamedRange> g_named;

}  // namespace

void AddExecRange(u64 base, u64 size) {
  base::LockGuard lk(g_range_mutex);
  g_ranges.push_back({base, size});
}

ExecRange FindExecRange(u64 address) {
  base::LockGuard lk(g_range_mutex);
  for (auto& r : g_ranges)
    if (address >= r.base && address < r.base + r.size)
      return r;
  return {0, 0};
}

void AddModuleRange(u64 base, u64 size, const char* name) {
  base::LockGuard lk(g_named_mutex);
  g_named.push_back({base, size, base::String(name)});
}

void SymbolizeModuleAddress(u64 address, char* out, size_t n) {
  base::LockGuard lk(g_named_mutex);
  for (auto& r : g_named)
    if (address >= r.base && address < r.base + r.size) {
      std::snprintf(out, n, "%s[%#llx]+%#llx",
                    r.name.empty() ? "?" : r.name.c_str(),
                    (unsigned long long)r.base,
                    (unsigned long long)(address - r.base));
      return;
    }
  std::snprintf(out, n, "%#llx", (unsigned long long)address);
}

void ResetCodeMap() {
  guest::ResetResource(g_ranges);
  guest::ResetResource(g_named);
}

}  // namespace cpu::fex
