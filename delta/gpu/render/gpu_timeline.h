/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#pragma once

#include "base/arch.h"

namespace gpu::render::timeline {

void Refresh();
bool Active();

u16 Begin(const char* name);
void End(u16 query);
void Time(u16 query, u64 gpu_ticks);

}  // namespace gpu::render::timeline
