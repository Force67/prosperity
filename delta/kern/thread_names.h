/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * See thread_names.cpp: carries guest sys_mname stack tags onto host threads
 * (pthread names), so the wait probe / gdb / perf can attribute threads.
 */

#pragma once

#include <cstddef>

namespace krnl {

// Called by the CPU backend when a guest thread starts running on the calling
// host thread; names the host thread from the stack's VMA tag if present.
void registerGuestThreadStack(const void *stack, size_t size);
void unregisterGuestThreadStack();

// Called by sys_mname after tagging [ptr, ptr+len): renames any live guest
// thread whose registered stack overlaps the tagged range.
void nameThreadsForRange(const void *ptr, size_t len, const char *name);

}  // namespace krnl
