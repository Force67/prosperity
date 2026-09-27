# Window smoke test: host window + Vulkan swapchain + present.
add_executable(window_test window_test.cc)
target_link_libraries(window_test PRIVATE delta_deps delta_host)
