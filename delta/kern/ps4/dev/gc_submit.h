#pragma once

#include "base/arch.h"

namespace kern {
// Decode guest graphics indirect-buffer descriptors.
void SubmitGcDescriptors(const void* descriptors, u32 count);
}  // namespace kern
