/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Kept apart from compute.cc so a test that stubs the GDS primitives links
// this without the rest of the compute backend.

#include "base/arch.h"
#include "base/containers/vector.h"
#include "gpu/guest_memory.h"
#include "gpu/render/renderer.h"

namespace gpu::render {

bool DmaGds(Renderer& renderer,
            u32 src_sel,
            u32 dst_sel,
            u64 src,
            u64 dst,
            u32 bytes,
            u32 data) {
  constexpr u64 kGdsBytes = 65536;
  const bool src_is_memory = src_sel == 0 || src_sel == 3;
  const bool dst_is_memory = dst_sel == 0 || dst_sel == 3;
  if (dst_sel == 1) {
    if (dst >= kGdsBytes || bytes > kGdsBytes - dst)
      return false;
    if (src_sel == 2)
      return FillGds(renderer, static_cast<u32>(dst), bytes, data);
    if (src_is_memory)
      return gpu::IsReadableRange(src, bytes) &&
             FlushCsWritesRange(renderer, src, bytes, "dma") &&
             WriteGds(renderer, static_cast<u32>(dst),
                      reinterpret_cast<const void*>(src), bytes);
    if (src_sel != 1 || src >= kGdsBytes || bytes > kGdsBytes - src)
      return false;
    base::Vector<u8> copy(bytes);
    return ReadGds(renderer, static_cast<u32>(src), copy.data(), bytes) &&
           WriteGds(renderer, static_cast<u32>(dst), copy.data(), bytes);
  }
  if (!(src_sel == 1 && src < kGdsBytes && bytes <= kGdsBytes - src &&
        dst_is_memory && gpu::IsReadableRange(dst, bytes)))
    return false;
  return CopyGdsToGuest(static_cast<u32>(src), dst, bytes) ||
         (FlushCsWritesRange(renderer, dst, bytes, "dma") &&
         ReadGds(renderer, static_cast<u32>(src), reinterpret_cast<void*>(dst),
                 bytes));
}

}  // namespace gpu::render
