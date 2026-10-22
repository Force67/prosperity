#pragma once
// A ring of recent GPU commands that write guest memory (copies and fills
// into imported host buffers), so a crash on corrupted guest data can name the
// command that last wrote there. Recording is lock-free and cheap; entries are
// tagged with whatever the renderer set as the current context.

#include "base/arch.h"

namespace gpu::rhi {
class Buffer;

namespace write_journal {

struct Entry {
  u64 first = 0, end = 0;  // guest address range written
  u64 context = 0;         // the renderer's tag, e.g. the shader address
  u32 frame = 0;
  // 'c' copy, 'f' fill, 'u' update, 't' texture->buffer, 'a' copy-after,
  // 'm' word merge
  char kind = 0;
};

void SetContext(u32 frame, u64 context);

// Records [offset, offset+bytes) of `dst` if it imports guest memory.
void Record(const Buffer* dst, u64 offset, u64 bytes, char kind);

// The newest entries whose range comes within `radius` bytes of `address`,
// newest first; returns how many were written to `out`.
u32 Find(u64 address, u64 radius, Entry* out, u32 max);

}  // namespace write_journal
}  // namespace gpu::rhi
