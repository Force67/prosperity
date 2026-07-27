# GPU backend: GCN/Liverpool PM4 command processing -> Vulkan rendering.
# Driven by the Gnm HLE (delta_runtime); renders via delta_gfx's Vulkan device.
# The GCN shader recompiler emits SPIR-V directly from the GCN bytecode
# (gcn/spirv) and runs a SPIRV-Tools optimize pass; there is no GLSL/shaderc step.
#
# On Android the NDK sysroot provides <vulkan/vulkan.h> and libvulkan.so on the
# default include/link paths (CMake's FindVulkan doesn't locate them), so link
# the plain `vulkan` library instead of going through find_package.
if(ANDROID)
  add_delta_module(gpu DEPS vulkan delta_gfx)
else()
  find_package(Vulkan REQUIRED)
  add_delta_module(gpu DEPS Vulkan::Vulkan delta_gfx)
endif()

# Layering: rhi/ is the backend-free renderer interface the command processors
# talk to, vulkan/ is the only backend implementing it, and the PS4
# PM4/Liverpool + GCN and PS5 AGC/RDNA2 paths live in ps4/ and ps5/. Every
# include inside the module is spelled from the delta root ("gpu/..."), which
# the shared DELTA_ROOT include dir already resolves -- no extra include roots.

# The recompiler emits SPIR-V directly (no GLSL/shaderc).
find_package(PkgConfig QUIET)

# Direct GCN->SPIR-V backend (gcn/spirv): emits SPIR-V from the GCN bytecode then
# runs a SPIRV-Tools optimize/validate pass. Needs SPIRV-Headers (spirv.hpp11) and
# SPIRV-Tools (static lib: the C++ Optimizer + the C validation API). When the
# packages are unavailable the backend self-disables at build time (no shader
# recompilation at all -- guest draws/dispatches are skipped) via the
# DELTA_HAVE_SPIRV_BACKEND define.
if(PkgConfig_FOUND)
  pkg_check_modules(SPIRVTOOLS QUIET SPIRV-Tools)
  pkg_check_modules(SPIRVHEADERS QUIET SPIRV-Headers)
endif()
if(SPIRVTOOLS_FOUND AND SPIRVHEADERS_FOUND)
  target_link_libraries(delta_gpu PUBLIC ${SPIRVTOOLS_LIBRARIES})
  target_link_directories(delta_gpu PUBLIC ${SPIRVTOOLS_LIBRARY_DIRS})
  target_include_directories(delta_gpu PUBLIC ${SPIRVTOOLS_INCLUDE_DIRS}
                             ${SPIRVHEADERS_INCLUDE_DIRS})
  target_compile_definitions(delta_gpu PUBLIC DELTA_HAVE_SPIRV_BACKEND=1)
else()
  message(STATUS "SPIRV-Tools/Headers not found; direct GCN->SPIR-V backend disabled")
endif()
