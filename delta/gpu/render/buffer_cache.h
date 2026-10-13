/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

namespace gpu::rhi {
class Buffer;
}

namespace gpu::render {

struct CachedBuffer {
  rhi::Buffer* buffer = nullptr;
  u64 offset = 0;
};

bool FindCachedBuffer(u64 base, u64 bytes, CachedBuffer& out);

bool CacheGuestBuffer(u64 base, u64 bytes, CachedBuffer& out);

void InvalidateCachedBuffers(u64 first, u64 end);

void BufferCacheEndFrame();

u64 CachedBufferBytes();

}  // namespace gpu::render
