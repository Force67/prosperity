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
    -Wno-alloc-size-larger-than    # base::String::append SSO range checks
  )
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(AddDeltaModule)

option(DELTA_BUILD_TESTS "Build the delta module unit tests" ON)
if(DELTA_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(vendor)
add_subdirectory(shared)
add_subdirectory(delta)

add_subdirectory(tools/modload)
