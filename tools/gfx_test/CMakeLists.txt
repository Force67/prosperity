# Graphics smoke test: window + Vulkan swapchain + present.
add_executable(gfx_test gfx_test.cpp)
target_link_libraries(gfx_test PRIVATE delta_deps delta_gfx)
