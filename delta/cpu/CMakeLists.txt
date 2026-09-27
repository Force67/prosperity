# delta/cpu: CPU execution backend, chosen at build time by the host arch:
# NativeBackend on x86-64, FexBackend (the embedded FEXCore JIT) on aarch64.
if(DELTA_BACKEND STREQUAL "FEX")
  add_delta_module(cpu DEPS fex::core)
  set_source_files_properties(native_backend.cc PROPERTIES HEADER_FILE_ONLY ON)
  # FEXCore's headers are written for Clang; GCC's stricter parse rejects a few
  # constructs (e.g. a nested type whose name matches an enclosing one). The
  # canonical downgrade for "changes meaning of X" is -fpermissive.
  target_compile_options(delta_cpu PRIVATE -fpermissive)
else()
  add_delta_module(cpu)
  set_source_files_properties(fex_backend.cc PROPERTIES HEADER_FILE_ONLY ON)
endif()
