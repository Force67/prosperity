cmake_minimum_required(VERSION 3.20)

# Vendored zlib still has cmake_minimum_required(VERSION 2.4.4); CMake 4
# refuses anything below 3.5 unless we override.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

project(PS4Delta CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Resource defines consumed by the .rc scripts in _res. Keep the full set
# defined for the windows resource compiler.
add_compile_definitions(
  FXNAME="PS4Delta"
  FXNAME_WIDE=L"PS4Delta"
  rsc_company="Dev-Force/Alpin-Dev"
  rsc_description="Delta PS4 emulator"
  rsc_fileversion="1.0.0.0"
  rsc_internalname="ps4delta"
  rsc_originalname="ps4delta.exe"
  rsc_productname="PS4Delta"
  rsc_productversion="1.0.0.0"
  rsc_copyright="Force67 2019-2020. All rights reserved"
)

if(WIN32)
  add_compile_definitions(
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    _SCL_SECURE_NO_WARNINGS
  )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DELTA_DBG=1 _DEBUG)
endif()

# Silence warnings from old code we are not fixing right now.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(
    -Wno-deprecated-declarations  # std::is_pod, codecvt
    -Wno-deprecated-builtins
    -Wno-narrowing
    -Wno-multichar
    -Wno-invalid-offsetof
  )
endif()
# GCC-only: clang (e.g. the Android NDK) rejects this and capstone builds -Werror.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wno-alloc-size-larger-than)  # base::String::append SSO range checks
endif()

# NDK libc++ lacks std::atomic_ref (used by the embedded FEX headers); force a
# builtin-based polyfill into C++ TUs. CXX only: C deps (capstone/zlib) must not
# pull in <atomic>. SHELL: keeps "-include <path>" as one un-deduplicated arg.
if(ANDROID)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/android_atomic_ref.h>")
  # The app build is a dlopen'd .so; its thread_local vars (and FEXCore's) must
  # use a dynamic TLS model, not Initial-Exec (dlopen can't satisfy IE TLS).
  # global-dynamic also works in the CLI executable, so apply it everywhere.
  add_compile_options(-ftls-model=global-dynamic)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(AddDeltaModule)

# CPU execution backend selection.
#
# x86-64 host: guest PS4 x86-64 code runs natively (the existing path; only
#   syscalls and fs:-relative TLS are rewritten in-place by runtime/code_lift).
# aarch64 host: guest code runs through an embedded FEXCore JIT and syscalls
#   arrive via a FEXCore callback dispatched to lv2 (delta/cpu/fex_backend).
# Everything else (loader, lv2 HLE, devices) stays host-native either way.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(DELTA_BACKEND "FEX" CACHE STRING "CPU execution backend (FEX|NATIVE)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(DELTA_BACKEND "NATIVE" CACHE STRING "CPU execution backend (FEX|NATIVE)")
else()
  message(FATAL_ERROR "Unsupported host arch '${CMAKE_SYSTEM_PROCESSOR}' (need x86_64 or aarch64)")
endif()
message(STATUS "Delta CPU backend: ${DELTA_BACKEND} (host ${CMAKE_SYSTEM_PROCESSOR})")

if(DELTA_BACKEND STREQUAL "FEX")
  add_compile_definitions(DELTA_BACKEND_FEX=1)
  include(BuildFEXCore)
else()
  add_compile_definitions(DELTA_BACKEND_NATIVE=1)
endif()

# Build an on-screen Android app (NativeActivity APK) instead of the headless
# adb-shell binary: main/ builds libps4delta_app.so and gfx presents to the
# device screen (gfx_android) rather than dumping frames.
option(DELTA_ANDROID_APP "Build the Android NativeActivity app" OFF)
if(DELTA_ANDROID_APP)
  if(NOT ANDROID)
    message(FATAL_ERROR "DELTA_ANDROID_APP requires the Android NDK toolchain")
  endif()
  add_compile_definitions(DELTA_ANDROID_APP=1)
endif()

option(DELTA_BUILD_TESTS "Build the delta module unit tests" ON)
if(DELTA_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(vendor)
add_subdirectory(shared)
add_subdirectory(delta)

# Host-only dev tools (gfx_test needs an SDL3 window). The Android target builds
# just the ps4delta executable, run headless via adb.
if(NOT ANDROID)
  add_subdirectory(tools/modload)
  add_subdirectory(tools/modexec)
  add_subdirectory(tools/pkg_check)
  add_subdirectory(tools/pup_extract)
  add_subdirectory(tools/gfx_test)
endif()
