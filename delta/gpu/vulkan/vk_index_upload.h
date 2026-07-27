/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Guest index decoding and the host upload policy. Vulkan consumes 16- and
// 32-bit indices directly; guest 8-bit indices are widened to 16-bit.

#include <cstdint>

namespace gpu::vk {

uint32_t GuestIndexElementBytes(uint32_t index_type);
uint32_t UploadedIndexElementBytes(uint32_t index_type);
uint32_t MaxGuestIndex(const void* source, uint32_t count, uint32_t index_type);
void CopyGuestIndices(void* destination,
                      const void* source,
                      uint32_t count,
                      uint32_t index_type);

}  // namespace gpu::vk
