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

# The D3D12 rhi backend (d3d12/): Windows D3D12, or on Linux vkd3d's
# D3D12-over-Vulkan with the same headers. SPIR-V is lowered to HLSL with the
# vendored SPIRV-Cross and compiled to DXIL by DXC, which is loaded at run
# time (dxcompiler.dll ships next to the executable on Windows). Discovered
# before add_delta_module so tests/ can see the result on a clean configure.
option(DELTA_GPU_D3D12 "Build the D3D12 rhi backend when its dependencies exist" ON)
set(DELTA_HAVE_D3D12 OFF)
if(DELTA_GPU_D3D12 AND NOT ANDROID)
  if(WIN32)
    set(DELTA_HAVE_D3D12 ON)
  elseif(PkgConfig_FOUND)
    pkg_check_modules(VKD3D QUIET libvkd3d libvkd3d-utils)
    find_path(DXC_INCLUDE_DIR dxc/dxcapi.h)
    find_library(DXC_LIBRARY dxcompiler)
    if(VKD3D_FOUND AND DXC_INCLUDE_DIR AND DXC_LIBRARY)
      set(DELTA_HAVE_D3D12 ON)
    endif()
  endif()
  if(NOT DELTA_HAVE_D3D12)
    message(STATUS "vkd3d or DXC not found; the D3D12 rhi backend is disabled")
  endif()
endif()

if(ANDROID)
  add_delta_module(gpu DEPS vulkan delta_gfx)
else()
  find_package(Vulkan REQUIRED)
  add_delta_module(gpu DEPS Vulkan::Vulkan delta_gfx)
endif()

# Header-only BC6H decoder used by compute image staging.
target_include_directories(delta_gpu PRIVATE ${CMAKE_SOURCE_DIR}/vendor/bcdec)

# Layering: render/ is the backend-free renderer interface the command processors
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

if(DELTA_HAVE_D3D12)
  target_compile_definitions(delta_gpu PRIVATE DELTA_HAVE_D3D12=1)
  target_link_libraries(delta_gpu PRIVATE spirv-cross-hlsl)
  if(WIN32)
    target_link_libraries(delta_gpu PRIVATE d3d12 dxgi dxguid)
  else()
    target_include_directories(delta_gpu PRIVATE ${VKD3D_INCLUDE_DIRS}
                               ${DXC_INCLUDE_DIR})
    target_link_libraries(delta_gpu PRIVATE ${VKD3D_LINK_LIBRARIES})
    set_source_files_properties(d3d12/d3d12_dxc.cc PROPERTIES
      COMPILE_DEFINITIONS "DELTA_DXCOMPILER_PATH=\"${DXC_LIBRARY}\"")
  endif()
else()
  get_target_property(_gpu_srcs delta_gpu SOURCES)
  list(FILTER _gpu_srcs EXCLUDE REGEX "/d3d12/")
  set_property(TARGET delta_gpu PROPERTY SOURCES ${_gpu_srcs})
endif()
