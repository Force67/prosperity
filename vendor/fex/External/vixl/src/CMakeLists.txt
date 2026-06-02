set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (SRCS
  code-buffer-vixl.cc
  compiler-intrinsics-vixl.cc
  cpu-features.cc
  utils-vixl.cc
  aarch64/assembler-aarch64.cc
  aarch64/assembler-sve-aarch64.cc
  aarch64/debugger-aarch64.cc
  aarch64/decoder-aarch64.cc
  aarch64/disasm-aarch64.cc
  aarch64/cpu-aarch64.cc
  aarch64/instructions-aarch64.cc
  aarch64/macro-assembler-aarch64.cc
  aarch64/macro-assembler-sve-aarch64.cc
  aarch64/operands-aarch64.cc
  aarch64/pointer-auth-aarch64.cc
  aarch64/registers-aarch64.cc)

if (ENABLE_VIXL_SIMULATOR)
  list(APPEND SRCS
    aarch64/cpu-features-auditor-aarch64.cc
    aarch64/logic-aarch64.cc
    aarch64/simulator-aarch64.cc)
endif()

add_library(vixl OBJECT ${SRCS})

# This is only refrenced by .cc files, so it can be private
target_compile_definitions(vixl PRIVATE -DVIXL_CODE_BUFFER_MALLOC=1)

if (ENABLE_VIXL_SIMULATOR)
  target_compile_definitions(vixl PRIVATE -DVIXL_INCLUDE_SIMULATOR_AARCH64=1)
endif()

# These are refrenced by all the .h files
# we need to export them as public so any target linking vixl
# will alo define these
target_compile_definitions(vixl PUBLIC -DVIXL_INCLUDE_TARGET_AARCH64=1)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  target_compile_definitions(vixl PUBLIC -DVIXL_DEBUG=1)
endif()

target_compile_options(vixl PRIVATE -ffunction-sections)
set_target_properties(vixl PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(vixl PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(vixl PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)

if (CMAKE_BUILD_TYPE MATCHES "RELEASE")
  target_link_options(vixl
    PRIVATE
    "LINKER:--gc-sections"
    "LINKER:--strip-all"
    "LINKER:--as-needed"
  )
endif()

