# --------------------------------------------------------------------------
# zlib: uses upstream CMakeLists, but we only want the static lib target. Skip
# the shared lib, examples, install rules and — importantly — its CTest
# registration, which would otherwise pollute our test suite.
# --------------------------------------------------------------------------
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
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
# capstone 6 gates architectures behind CAPSTONE_ARCHITECTURE_DEFAULT and
# renamed several per-arch flags. Turn the default off and re-enable only
# x86 so we don't compile (and link) all ~14 backends.
set(CAPSTONE_ARCHITECTURE_DEFAULT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_X86_SUPPORT ON CACHE BOOL "" FORCE)
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
# format), and the filesystem layer base::Option's file loader reads through.
# Everything else in equilibrium pulls in mimalloc / tracy / googletest, which
# we do not need here.
# --------------------------------------------------------------------------
add_library(eq_base STATIC
  equilibrium/base/check.cc
  equilibrium/base/environment_variables_posix.cc
  equilibrium/base/logging.cc
  equilibrium/base/option_file.cc
  equilibrium/base/strings/format.cc
  equilibrium/base/filesystem/file.cc
  equilibrium/base/filesystem/path.cc
  equilibrium/base/filesystem/scoped_file.cc
  equilibrium/base/filesystem/posix/file_posix.cc
  equilibrium/base/filesystem/posix/path_posix.cc
  equilibrium/base/filesystem/linux/scoped_file_linux.cc
  equilibrium/base/text/code_convert.cc
)
target_include_directories(eq_base PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/equilibrium
)
target_compile_features(eq_base PUBLIC cxx_std_20)
target_compile_definitions(eq_base PUBLIC
  BASE_STRIP_BUGCHECK     # don't pull in fancy backtrace machinery
  # base/filesystem picks its char type and syscalls off these; they are what
  # equilibrium's own build sets, not something build_config.h derives.
  OS_LINUX
  OS_POSIX
)

# --------------------------------------------------------------------------
# mbedtls (2.28 LTS): we only want the static mbedcrypto target for the PKG
# loader (AES-128 ECB/CBC, bignum modexp for RSA, SHA-256, HMAC). Skip the
# programs, tests, x509/tls libs and install rules. The stock 2.28 default
# config already enables everything we use, so no custom config is needed.
# --------------------------------------------------------------------------
set(ENABLE_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MBEDTLS_LIBRARY ON CACHE BOOL "" FORCE)
add_subdirectory(mbedtls EXCLUDE_FROM_ALL)
# Thin wrapper so consumers link one name; mbedcrypto already carries the
# include interface, but keep it explicit to match the other vendor ifaces.
add_library(mbedtls_iface INTERFACE)
target_include_directories(mbedtls_iface INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/mbedtls/include
)
target_link_libraries(mbedtls_iface INTERFACE mbedcrypto)

# --------------------------------------------------------------------------
# googletest: only needed when building the module unit tests.
# --------------------------------------------------------------------------
if(DELTA_BUILD_TESTS)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()
