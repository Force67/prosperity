# Dynamic runtime: x86 code lifting and NID encoding.
# Co-dependent with kern (cycle declared in delta/CMakeLists.txt).
add_delta_module(runtime DEPS xbyak capstone_iface delta_host)
