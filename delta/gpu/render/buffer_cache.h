/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

namespace gpu::rhi {
class Buffer;
}

namespace gpu::render {

// Guest vertex data kept in GPU-visible memory across frames. A copy stays
// valid until the guest write tracker reports a write to its pages
// (InvalidateCachedBuffers), so a static mesh is copied once instead of once
// per frame. Copies are immutable: a replaced or evicted one is released only
// after every frame that could still read it has retired.
struct CachedBuffer {
  rhi::Buffer* buffer = nullptr;
  u64 offset = 0;
};

// A copy that covers [base, base+bytes), or false.
bool FindCachedBuffer(u64 base, u64 bytes, CachedBuffer& out);

// Copies [base, base+bytes) of guest memory into the cache. The caller must
// have armed those pages in the write tracker before. False when there is no
// room; the caller then stages the bytes the per-frame way.
bool CacheGuestBuffer(u64 base, u64 bytes, CachedBuffer& out);

// Drops every copy overlapping [first, end).
void InvalidateCachedBuffers(u64 first, u64 end);

// Once a frame: evicts copies no draw has used for a while and releases
// memory no in-flight frame reads any more.
void BufferCacheEndFrame();

// Bytes held by live copies, for the perf report.
u64 CachedBufferBytes();

}  // namespace gpu::render
