# Kernel: process model, lv2 syscalls, virtual devices, VM, object table.
# Uses xbyak to emit thunks. Co-dependent with the runtime module (cycle
# declared in delta/CMakeLists.txt). Links crypto for fake-SELF -> ELF on load.
add_delta_module(kern DEPS xbyak delta_crypto)
