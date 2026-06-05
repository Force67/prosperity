# GPU backend: GCN/Liverpool PM4 command processing -> Vulkan rendering.
# Driven by the Gnm HLE (delta_runtime); renders via delta_gfx's Vulkan device.
# The GCN shader recompiler compiles GLSL to SPIR-V at runtime via shaderc
# (gcn/shaderc_compile.cpp).
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

# shaderc (runtime GLSL -> SPIR-V). Provided by the nix shaderc package, which
# ships a pkg-config file (`shaderc` -> -lshaderc_shared) with the header on the
# default include path. Fall back to find_library/find_path if pkg-config or the
# .pc is unavailable.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(SHADERC QUIET shaderc)
endif()
if(SHADERC_FOUND)
  target_link_libraries(delta_gpu PUBLIC ${SHADERC_LIBRARIES})
  target_include_directories(delta_gpu PUBLIC ${SHADERC_INCLUDE_DIRS})
  target_link_directories(delta_gpu PUBLIC ${SHADERC_LIBRARY_DIRS})
else()
  find_library(SHADERC_LIB NAMES shaderc_shared shaderc REQUIRED)
  find_path(SHADERC_INC shaderc/shaderc.h REQUIRED)
  target_link_libraries(delta_gpu PUBLIC ${SHADERC_LIB})
  target_include_directories(delta_gpu PUBLIC ${SHADERC_INC})
endif()
