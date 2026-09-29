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
# We compile the subset PS4Delta uses in place of the STL: containers,
# strings, threads, locks, time and random, plus the runtime objects those
# need (check / logging / format) and the filesystem layer base::Option's
# file loader reads through.
# Everything else in equilibrium pulls in mimalloc / tracy / googletest, which
# we do not need here.
# --------------------------------------------------------------------------
add_library(eq_base STATIC
  equilibrium/base/check.cc
  equilibrium/base/environment_variables_posix.cc
  equilibrium/base/logging.cc
  equilibrium/base/option_file.cc
  equilibrium/base/standard_streams_posix.cc
  equilibrium/base/strings/float_format.cc
  equilibrium/base/strings/format.cc
  equilibrium/base/strings/number_parse.cc
  equilibrium/base/filesystem/file.cc
  equilibrium/base/filesystem/path.cc
  equilibrium/base/filesystem/scoped_file.cc
  equilibrium/base/filesystem/posix/file_posix.cc
  equilibrium/base/filesystem/posix/path_posix.cc
  equilibrium/base/filesystem/linux/scoped_file_linux.cc
  equilibrium/base/text/code_convert.cc
  equilibrium/base/random/random.cc
  equilibrium/base/random/random_posix.cc
  equilibrium/base/threading/condition_variable_linux.cc
  equilibrium/base/threading/spinning_mutex.cc
  equilibrium/base/threading/thread.cc
  equilibrium/base/threading/thread_posix.cc
  equilibrium/base/time/time.cc
  equilibrium/base/time/time_posix.cc
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
# unrar (official UnRAR source): decompression backend for booting titles
# straight out of .rar dumps. Library build only: the source list mirrors
# OBJECTS+LIB_OBJ of the upstream makefile's `lib` target (no console main),
# with the same defines (RARDLL puts it in silent library mode). The license
# (unrar/license.txt) permits extraction-only use, which is all we do.
# --------------------------------------------------------------------------
set(UNRAR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unrar)
add_library(unrar STATIC
  ${UNRAR_DIR}/rar.cpp ${UNRAR_DIR}/strlist.cpp ${UNRAR_DIR}/strfn.cpp
  ${UNRAR_DIR}/pathfn.cpp ${UNRAR_DIR}/smallfn.cpp ${UNRAR_DIR}/global.cpp
  ${UNRAR_DIR}/file.cpp ${UNRAR_DIR}/filefn.cpp ${UNRAR_DIR}/filcreat.cpp
  ${UNRAR_DIR}/archive.cpp ${UNRAR_DIR}/arcread.cpp ${UNRAR_DIR}/unicode.cpp
  ${UNRAR_DIR}/system.cpp ${UNRAR_DIR}/crypt.cpp ${UNRAR_DIR}/crc.cpp
  ${UNRAR_DIR}/rawread.cpp ${UNRAR_DIR}/encname.cpp ${UNRAR_DIR}/resource.cpp
  ${UNRAR_DIR}/match.cpp ${UNRAR_DIR}/timefn.cpp ${UNRAR_DIR}/rdwrfn.cpp
  ${UNRAR_DIR}/consio.cpp ${UNRAR_DIR}/options.cpp ${UNRAR_DIR}/errhnd.cpp
  ${UNRAR_DIR}/rarvm.cpp ${UNRAR_DIR}/secpassword.cpp ${UNRAR_DIR}/rijndael.cpp
  ${UNRAR_DIR}/getbits.cpp ${UNRAR_DIR}/sha1.cpp ${UNRAR_DIR}/sha256.cpp
  ${UNRAR_DIR}/blake2s.cpp ${UNRAR_DIR}/hash.cpp ${UNRAR_DIR}/extinfo.cpp
  ${UNRAR_DIR}/extract.cpp ${UNRAR_DIR}/volume.cpp ${UNRAR_DIR}/list.cpp
  ${UNRAR_DIR}/find.cpp ${UNRAR_DIR}/unpack.cpp ${UNRAR_DIR}/headers.cpp
  ${UNRAR_DIR}/threadpool.cpp ${UNRAR_DIR}/rs16.cpp ${UNRAR_DIR}/cmddata.cpp
  ${UNRAR_DIR}/ui.cpp ${UNRAR_DIR}/largepage.cpp ${UNRAR_DIR}/filestr.cpp
  ${UNRAR_DIR}/scantree.cpp ${UNRAR_DIR}/dll.cpp ${UNRAR_DIR}/qopen.cpp
)
target_compile_definitions(unrar PUBLIC
  RARDLL
  _FILE_OFFSET_BITS=64
  _LARGEFILE_SOURCE
  RAR_SMP
)
target_compile_options(unrar PRIVATE -w)
add_library(unrar_iface INTERFACE)
target_include_directories(unrar_iface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unrar_iface INTERFACE unrar)

# --------------------------------------------------------------------------
# googletest: only needed when building the module unit tests.
# --------------------------------------------------------------------------
if(DELTA_BUILD_TESTS)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()

# --------------------------------------------------------------------------
# SPIRV-Cross: lowers the recompiler's SPIR-V to GLSL (OpenGL backend) and
# HLSL (D3D12 backend). Static core + the two emitters, no CLI or tests.
# --------------------------------------------------------------------------
set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_FORCE_PIC ON CACHE BOOL "" FORCE)
add_subdirectory(SPIRV-Cross EXCLUDE_FROM_ALL)

# --------------------------------------------------------------------------
# Tracy client (v0.13.1, matches nixpkgs' tracy viewer and tracy-capture).
# On demand and localhost only. Its crash handler would take over the signals
# the guest fault paths own, so it is off.
# --------------------------------------------------------------------------
if(DELTA_TRACY)
  add_library(tracy_client STATIC tracy/public/TracyClient.cpp)
  target_include_directories(tracy_client SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tracy/public)
  target_compile_definitions(tracy_client PUBLIC
    TRACY_ENABLE TRACY_ON_DEMAND TRACY_ONLY_LOCALHOST TRACY_NO_BROADCAST
    TRACY_NO_CRASH_HANDLER TRACY_NO_FRAME_IMAGE DELTA_TRACY=1)
  target_compile_options(tracy_client PRIVATE -w)
  target_link_libraries(tracy_client PUBLIC pthread dl)
endif()
