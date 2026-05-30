# Staged execution harness: load a module graph, then drive each post-load
# stage (relocate -> ... -> entry) explicitly so we can fix one layer at a time.
add_executable(modexec modexec.cpp)
target_link_libraries(modexec PRIVATE delta_deps delta_kern delta_runtime)
