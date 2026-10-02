#pragma once

#include <vulkan/vulkan.h>

#include "base/arch.h"

namespace ui {

bool HomeBackgroundVkInit(VkDevice device, VkRenderPass pass);
void HomeBackgroundVkDraw(VkCommandBuffer cmd,
                          u32 width,
                          u32 height,
                          float time,
                          u32 style,
                          float pulse);
void HomeBackgroundVkShutdown();

}  // namespace ui
