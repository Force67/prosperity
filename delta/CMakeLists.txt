# delta/: the emulator, split into per-module libraries.
#
# Each subdirectory is a self-contained module (see cmake/AddDeltaModule.cmake).
# main/ is the entry point; it links the rest together into the ps4delta
# executable.

# Include root shared by every module so cross-module includes resolve, e.g.
# <kern/proc.h> -> delta/kern/proc.h, "runtime/vprx/vprx.h" -> delta/runtime/...
set(DELTA_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "delta module include root")

# Foundation every module links: the shared utility library (which itself
# brings fmt + equilibrium base) and the delta/ include root.
add_library(delta_deps INTERFACE)
target_include_directories(delta_deps INTERFACE ${DELTA_ROOT})
target_link_libraries(delta_deps INTERFACE shared)
if(UNIX)
  target_link_libraries(delta_deps INTERFACE pthread dl)
endif()

add_subdirectory(crypto)
add_subdirectory(formats)
add_subdirectory(gfx)
add_subdirectory(kern)
add_subdirectory(runtime)
add_subdirectory(main)

# kern and runtime are co-dependent: kern drives the JIT runtime
# (proc -> vprx/code_lift) and the runtime calls back into kern
# (code_lift -> proc::getActive). Declare the static-lib cycle so the linker
# repeats the archives until symbols resolve.
target_link_libraries(delta_kern    PUBLIC delta_runtime)
target_link_libraries(delta_runtime PUBLIC delta_kern)
