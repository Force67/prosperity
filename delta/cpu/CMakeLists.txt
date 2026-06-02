# delta/cpu: CPU execution backend.
#
# Both backend .cpp files are compiled but only one has a live body (the other
# is #ifdef'd out by the DELTA_BACKEND_* define): NativeBackend on x86-64,
# FexBackend (linking the embedded FEXCore JIT) on aarch64.
if(DELTA_BACKEND STREQUAL "FEX")
  add_delta_module(cpu DEPS fex::core)
  # FEXCore's headers are written for Clang; GCC's stricter parse rejects a few
  # constructs (e.g. a nested type whose name matches an enclosing one). The
  # canonical downgrade for "changes meaning of X" is -fpermissive.
  target_compile_options(delta_cpu PRIVATE -fpermissive)
else()
  add_delta_module(cpu)
endif()
