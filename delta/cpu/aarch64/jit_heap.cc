#include <sys/mman.h>

#include <FEXCore/Utils/Allocator.h>

#include "base/atomic.h"
#include "cpu/aarch64/cpu_map.h"
#include "cpu/backend.h"
#include "logger/logger.h"
#include "memory_debug/memory_debug.h"

namespace cpu {
namespace {

// FEXCore's own memory, kept out of guest VA: a guest MAP_FIXED over a
// kernel-placed FEX mapping zero-fills the JIT's block-link map.
#ifdef __ANDROID__
// 39-bit user VA: above the guest arena, below bionic's mmap_base.
constexpr uintptr_t kFexHeapBase = 0x0000'0060'0000'0000ull;
constexpr size_t kFexHeapSize = 32ull * 1024 * 1024 * 1024;
#else
constexpr uintptr_t kFexHeapBase = 0x0000'5000'0000'0000ull;
constexpr size_t kFexHeapSize = 96ull * 1024 * 1024 * 1024;
#endif
base::Atomic<uintptr_t> g_fex_heap_next{0};
uintptr_t g_fex_heap_end = 0;

void* FexInternalMmap(void* addr,
                      size_t len,
                      int prot,
                      int flags,
                      int fd,
                      off_t off) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
  if ((flags & MAP_FIXED) || !g_fex_heap_end)
    return fex::CpuMap(addr, len, prot, flags, fd, off);
  // A bare hint may land where the guest MAP_FIXEDs later; ignore it.
  (void)addr;
  const size_t alen = (len + 0xFFFull) & ~0xFFFull;
  uintptr_t base = g_fex_heap_next.fetch_add(alen, base::memory_order_relaxed);
  if (base + alen > g_fex_heap_end)
    return fex::CpuMap(nullptr, len, prot, flags, fd, off);
  return fex::CpuMap(reinterpret_cast<void*>(base), len, prot, flags | MAP_FIXED,
                     fd, off);
}

int FexInternalMunmap(void* addr, size_t len) {
  return fex::CpuUnmap(addr, len);
}

}  // namespace

void EarlyInit() {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#ifdef MAP_FIXED_NOREPLACE
  void* r = fex::CpuMap(reinterpret_cast<void*>(kFexHeapBase), kFexHeapSize,
                        PROT_NONE, flags | MAP_FIXED_NOREPLACE, -1, 0);
  if (r == MAP_FAILED || r != reinterpret_cast<void*>(kFexHeapBase))
    r = fex::CpuMap(nullptr, kFexHeapSize, PROT_NONE, flags, -1, 0);
#else
  void* r = fex::CpuMap(nullptr, kFexHeapSize, PROT_NONE, flags, -1, 0);
#endif
  if (r == MAP_FAILED) {
    LOG_WARNING(
        "fex: could not reserve internal heap window; JIT memory shares "
        "guest VA (may corrupt under heavy guest mmap use)");
    return;
  }
  g_fex_heap_next.store(reinterpret_cast<uintptr_t>(r),
                        base::memory_order_relaxed);
  g_fex_heap_end = reinterpret_cast<uintptr_t>(r) + kFexHeapSize;
  FEXCore::Allocator::mmap = FexInternalMmap;
  FEXCore::Allocator::munmap = FexInternalMunmap;
  LOG_INFO("fex: reserved internal heap {} +{:#x}", r, kFexHeapSize);
}

}  // namespace cpu
