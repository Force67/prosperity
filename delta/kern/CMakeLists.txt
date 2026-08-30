# Kernel: process model, lv2 syscalls, virtual devices, VM, object table, and
# the guest-code instrumentation the research workflow runs on (probe/).
# Uses xbyak to emit thunks. Co-dependent with the runtime module (cycle
# declared in delta/CMakeLists.txt). Links crypto for fake-SELF -> ELF on load,
# and formats for the container -> VFS providers behind /app0.
add_delta_module(kern DEPS xbyak delta_crypto delta_formats)
