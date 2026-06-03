# GPU backend: GCN/Liverpool PM4 command processing -> Vulkan rendering.
# Driven by the Gnm HLE (delta_runtime); renders via delta_gfx's Vulkan device.
find_package(Vulkan REQUIRED)
add_delta_module(gpu DEPS Vulkan::Vulkan delta_gfx)
