# Standalone harness: drive the real smodule loader against a single on-disk
# module so we can exercise / debug module loading outside the full emulator.
add_executable(modload modload.cpp)
target_link_libraries(modload PRIVATE delta_deps delta_kern delta_runtime)
