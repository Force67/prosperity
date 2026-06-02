# Graphics backend: SDL3 window + Vulkan present.
find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED CONFIG)

add_delta_module(gfx DEPS Vulkan::Vulkan SDL3::SDL3)
