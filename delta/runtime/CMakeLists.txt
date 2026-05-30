# Dynamic runtime: x86 code lifting (capstone + xbyak) and the vprx HLE module
# loader. Co-dependent with kern (cycle declared in delta/CMakeLists.txt).
add_delta_module(runtime DEPS xbyak capstone_iface)
