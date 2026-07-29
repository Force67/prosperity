# GPU backend: GCN/Liverpool PM4 command processing -> Vulkan rendering.
# Driven by the Gnm HLE (delta_runtime); renders via delta_gfx's Vulkan device.
# The GCN shader recompiler emits SPIR-V directly from the GCN bytecode
# (gcn/spirv) and runs a SPIRV-Tools optimize pass; there is no GLSL/shaderc step.
#
# On Android the NDK sysroot provides <vulkan/vulkan.h> and libvulkan.so on the
# default include/link paths (CMake's FindVulkan doesn't locate them), so link
# the plain `vulkan` library instead of going through find_package.

# Discover the optional SPIR-V backend before add_delta_module enters tests/;
# clean builds must register the backend's tests on their first configure.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(SPIRVTOOLS QUIET SPIRV-Tools)
  pkg_check_modules(SPIRVHEADERS QUIET SPIRV-Headers)
endif()

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

# Direct GCN->SPIR-V backend (gcn/spirv): emits SPIR-V from the GCN bytecode then
# runs a SPIRV-Tools optimize/validate pass. Needs SPIRV-Headers (spirv.hpp11) and
# SPIRV-Tools (static lib: the C++ Optimizer + the C validation API). When the
# packages are unavailable the backend self-disables at build time (no shader
# recompilation at all -- guest draws/dispatches are skipped) via the
# DELTA_HAVE_SPIRV_BACKEND define.
if(SPIRVTOOLS_FOUND AND SPIRVHEADERS_FOUND)
  # PRIVATE: SPIRV-Tools is an implementation detail of the recompiler; no
  # header on the public surface needs it, and keeping it off dependents'
  # include paths stops accidental use outside the module. LINK_LIBRARIES
  # (absolute paths) instead of LIBRARIES+link_directories so the link
  # requirement still reaches the final executable through the static lib.
  target_link_libraries(delta_gpu PRIVATE ${SPIRVTOOLS_LINK_LIBRARIES})
  target_include_directories(delta_gpu PRIVATE ${SPIRVTOOLS_INCLUDE_DIRS}
                             ${SPIRVHEADERS_INCLUDE_DIRS})
  target_compile_definitions(delta_gpu PRIVATE DELTA_HAVE_SPIRV_BACKEND=1)
else()
  message(STATUS "SPIRV-Tools/Headers not found; direct GCN->SPIR-V backend disabled")
endif()
