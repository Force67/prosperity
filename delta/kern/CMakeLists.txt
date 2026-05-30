# Kernel: process model, lv2 syscalls, virtual devices, VM, object table.
# Uses xbyak to emit thunks. Co-dependent with the runtime module (cycle
# declared in delta/CMakeLists.txt).
add_delta_module(kern DEPS xbyak)
