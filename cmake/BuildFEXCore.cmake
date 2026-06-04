# Builds the vendored FEXCore JIT (vendor/fex) as an isolated ExternalProject
# and exposes it as the imported interface target `fex::core`.
#
# Only used on aarch64, where guest PS4 x86-64 code runs through the FEX JIT
# (see delta/cpu/fex_backend.*). FEX is Clang-only and we keep LTO/jemalloc/
# tests/thunks off so the result is a set of plain static archives that link
# into the (possibly GCC-built) emulator. Building FEX with its own pinned
# Clang isolates that requirement from the rest of the project.
include_guard(GLOBAL)
include(ExternalProject)

set(FEX_SRC ${CMAKE_SOURCE_DIR}/vendor/fex)

# How FEX gets its compiler. On Android we cross-compile with the NDK toolchain
# (bionic, arm64-v8a) the outer build is already using, so forward that file +
# ABI/platform and let it pick the NDK clang. TUNE_CPU=none avoids FEX's
# configure-time /proc/cpuinfo probe (it would read the x86 build host); the JIT
# detects real host features at runtime. Elsewhere (native arm/x86 host build)
# FEX builds with its own pinned Clang so it links into a possibly-GCC emulator.
if(ANDROID)
  # NDK libc++ (r27/r28) lacks std::atomic_ref, which FEX needs; force-include a
  # builtin-based polyfill (see cmake/android_atomic_ref.h).
  set(_fex_toolchain_args
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    -DANDROID_ABI=${ANDROID_ABI}
    -DANDROID_PLATFORM=${ANDROID_PLATFORM}
    -DANDROID_STL=${ANDROID_STL}
    -DTUNE_CPU=none
    "-DCMAKE_CXX_FLAGS=-include ${CMAKE_CURRENT_LIST_DIR}/android_atomic_ref.h")
else()
  # Clang used to build FEX. Override with -DDELTA_FEX_CLANG=/path if needed.
  # Defaults to the nix-provided clang wrapper the bring-up was validated on.
  set(DELTA_FEX_CLANG "/nix/store/bcv5hwb933cp93m9ckgsym21ncnhwm2v-clang-wrapper-18.1.8/bin"
      CACHE PATH "Directory containing the clang/clang++ used to build FEXCore")
  if(NOT EXISTS "${DELTA_FEX_CLANG}/clang++")
    message(FATAL_ERROR
      "FEXCore needs Clang (FEX refuses GCC) but ${DELTA_FEX_CLANG}/clang++ was not found.\n"
      "Set -DDELTA_FEX_CLANG=<dir containing clang/clang++>.")
  endif()
  set(_fex_toolchain_args
    -DCMAKE_C_COMPILER=${DELTA_FEX_CLANG}/clang
    -DCMAKE_CXX_COMPILER=${DELTA_FEX_CLANG}/clang++)
endif()

set(FEX_PREFIX  ${CMAKE_BINARY_DIR}/fex)
set(FEX_BUILD   ${FEX_PREFIX}/src/fexcore_ep-build)

# Static archives the embed links, in --start-group order (resolved below).
set(_fex_archives
  ${FEX_BUILD}/Source/Common/libCommon.a            # FEX::FetchHostFeatures
  ${FEX_BUILD}/FEXCore/Source/libFEXCore.a
  ${FEX_BUILD}/FEXCore/Source/libFEXCore_Base.a
  ${FEX_BUILD}/FEXCore/Source/libJemallocDummy.a    # aligned_alloc/free (glibc)
  ${FEX_BUILD}/Source/Common/cpp-optparse/libcpp-optparse.a
  ${FEX_BUILD}/External/fmt/libfmt.a
  ${FEX_BUILD}/External/xxhash/cmake_unofficial/libxxhash.a
  ${FEX_BUILD}/External/cephes/libcephes_128bit.a
  ${FEX_BUILD}/External/SoftFloat-3e/libsoftfloat_3e.a)

ExternalProject_Add(fexcore_ep
  SOURCE_DIR  ${FEX_SRC}
  PREFIX      ${FEX_PREFIX}
  CMAKE_ARGS
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    ${_fex_toolchain_args}
    -DENABLE_LTO=False
    -DENABLE_JEMALLOC=False
    -DENABLE_JEMALLOC_GLIBC_ALLOC=False
    -DBUILD_THUNKS=False
    -DBUILD_FEX_TOOLS=False
    -DBUILD_FEXCONFIG=False
    -DENABLE_OFFLINE_TELEMETRY=False
    -DBUILD_TESTING=OFF
    -DENABLE_CCACHE=False
    -DUSE_LEGACY_BINFMTMISC=True
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  # Build only the archives we need (and their object-lib/header deps).
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target
    Common FEXCore FEXCore_Base JemallocDummy fmt xxhash cephes_128bit softfloat_3e
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${_fex_archives}
  BUILD_ALWAYS OFF)

# Include dirs: source-tree headers + headers FEX generates into its build dir.
set(_fex_includes
  ${FEX_SRC}/FEXCore/include
  ${FEX_SRC}/Source
  ${FEX_SRC}/Source/Common
  ${FEX_SRC}/FEXHeaderUtils
  ${FEX_SRC}/CodeEmitter
  ${FEX_SRC}/External/fmt/include
  ${FEX_SRC}/External/robin-map/include
  ${FEX_SRC}/External/xxhash
  ${FEX_SRC}/External/cephes/include
  ${FEX_SRC}/External/SoftFloat-3e/include
  ${FEX_SRC}/External/tiny-json
  ${FEX_SRC}/External/range-v3/include
  ${FEX_BUILD}/include
  ${FEX_BUILD}/generated
  ${FEX_BUILD}/Source
  ${FEX_BUILD}/FEXCore/Source)
# The generated-header dirs may not exist at configure time; create so the
# INTERFACE_INCLUDE_DIRECTORIES check passes (ExternalProject fills them later).
foreach(_d ${_fex_includes})
  file(MAKE_DIRECTORY ${_d})
endforeach()

add_library(fex_core INTERFACE)
add_dependencies(fex_core fexcore_ep)
target_include_directories(fex_core INTERFACE ${_fex_includes})
# Whole archive set in one group so the circular FEXCore<->Base<->deps refs
# resolve regardless of order. bionic folds pthread into libc (no -lpthread).
set(_fex_sys_libs pthread dl m)
if(ANDROID)
  set(_fex_sys_libs dl m)
endif()
target_link_libraries(fex_core INTERFACE
  -Wl,--start-group ${_fex_archives} -Wl,--end-group
  ${_fex_sys_libs})
add_library(fex::core ALIAS fex_core)
