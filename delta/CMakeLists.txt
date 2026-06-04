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
if(ANDROID)
  target_link_libraries(delta_deps INTERFACE dl)  # bionic: pthread is in libc
elseif(UNIX)
  target_link_libraries(delta_deps INTERFACE pthread dl)
endif()

add_subdirectory(cpu)
add_subdirectory(crypto)
add_subdirectory(formats)
add_subdirectory(gfx)
add_subdirectory(gpu)
add_subdirectory(kern)
add_subdirectory(runtime)
add_subdirectory(main)

# kern and runtime are co-dependent: kern drives the JIT runtime
# (proc -> vprx/code_lift) and the runtime calls back into kern
# (code_lift -> proc::getActive). delta_cpu is likewise co-dependent with kern:
# the loader/proc/threads call cpu::backend(), and the FEX backend calls back
# into kern (lv2_get, moduleInfo). Declare the static-lib cycles so the linker
# repeats the archives until symbols resolve.
target_link_libraries(delta_kern    PUBLIC delta_runtime delta_cpu)
target_link_libraries(delta_runtime PUBLIC delta_kern delta_gpu)
target_link_libraries(delta_cpu     PUBLIC delta_kern)

# delta_gpu (GCN PM4 -> Vulkan) is driven by the Gnm HLE in delta_runtime and
# renders through delta_gfx's Vulkan device.
target_link_libraries(delta_gpu     PUBLIC delta_gfx)
