# --------------------------------------------------------------------------
# zlib: uses upstream CMakeLists, but we only want the static lib target
#         and we want it to skip examples / minigzip.
# --------------------------------------------------------------------------
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
# Upstream zlib's CMakeLists doesn't expose a normal include interface;
# wrap it.
add_library(zlib_iface INTERFACE)
target_include_directories(zlib_iface INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/zlib
  ${CMAKE_CURRENT_BINARY_DIR}/zlib
)
target_link_libraries(zlib_iface INTERFACE zlibstatic)

# --------------------------------------------------------------------------
# fmtlib
# --------------------------------------------------------------------------
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(fmtlib EXCLUDE_FROM_ALL)

# --------------------------------------------------------------------------
# capstone
# --------------------------------------------------------------------------
set(CAPSTONE_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_CSTOOL OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_DIET OFF CACHE BOOL "" FORCE)
set(CAPSTONE_X86_SUPPORT ON CACHE BOOL "" FORCE)
set(CAPSTONE_ARM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_ARM64_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_M68K_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_MIPS_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_PPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SPARC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SYSZ_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_XCORE_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_TMS320C64X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_M680X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_EVM_SUPPORT OFF CACHE BOOL "" FORCE)
add_subdirectory(capstone EXCLUDE_FROM_ALL)
add_library(capstone_iface INTERFACE)
target_include_directories(capstone_iface INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/capstone/include
)
target_link_libraries(capstone_iface INTERFACE capstone_static)

# --------------------------------------------------------------------------
# xbyak (header-only)
# --------------------------------------------------------------------------
add_library(xbyak INTERFACE)
target_include_directories(xbyak INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/xbyak/xbyak
)

# --------------------------------------------------------------------------
# equilibrium BASE: STL replacement library
#
# We compile a minimal subset: the headers used by PS4Delta
# (containers/vector.h, strings/xstring.h, memory/unique_pointer.h) plus
# the runtime objects those headers transitively need (check / logging /
# format). Everything else in equilibrium pulls in mimalloc / tracy /
# googletest, which we do not need here.
# --------------------------------------------------------------------------
add_library(eq_base STATIC
  equilibrium/base/check.cc
  equilibrium/base/logging.cc
  equilibrium/base/strings/format.cc
)
target_include_directories(eq_base PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/equilibrium
)
target_compile_features(eq_base PUBLIC cxx_std_20)
target_compile_definitions(eq_base PUBLIC
  BASE_STRIP_BUGCHECK     # don't pull in fancy backtrace machinery
)
