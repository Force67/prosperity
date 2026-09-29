/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#pragma once

// Guest GPU work on the Tracy timeline, next to the CPU threads. Built on the
// timestamp pairs the renderer writes around render regions and dispatches:
// a zone opens and closes where they are recorded, and its GPU times arrive
// once the submission has retired. Everything is a no-op until a Tracy viewer
// or tracy-capture is connected.

#include "base/arch.h"

namespace gpu::render::timeline {

// Once per frame, before anything records: follows the connection and
// announces (and calibrates) the GPU context to each new viewer.
void Refresh();
bool Active();

// A zone whose begin/end timestamps the caller writes; returns the query id
// that Time() reports the begin under (the end is id + 1).
u16 Begin(const char* name);
void End(u16 query);
void Time(u16 query, u64 gpu_ticks);

}  // namespace gpu::render::timeline
