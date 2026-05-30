# Helpers for declaring delta modules and their tests.
#
# Every subsystem under delta/ is a module: a static library `delta_<name>`
# built from the sources in its directory, sharing one include root (DELTA_ROOT)
# so cross-module includes like <kern/proc.h> resolve, and optionally carrying a
# tests/ subdirectory of GoogleTest unit tests registered with CTest.
include_guard(GLOBAL)

# add_delta_module(<name> [DEPS <libs...>])
#
# Globs the module's .cpp/.h (excluding tests/), builds delta_<name>, links the
# shared delta_deps interface plus any extra DEPS, and pulls in tests/ when
# DELTA_BUILD_TESTS is set.
function(add_delta_module name)
  cmake_parse_arguments(MOD "" "" "DEPS" ${ARGN})

  file(GLOB_RECURSE _srcs CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
  list(FILTER _srcs EXCLUDE REGEX "/tests/")

  if(WIN32)
    file(GLOB _rc ${CMAKE_CURRENT_SOURCE_DIR}/_res/*.rc)
    list(APPEND _srcs ${_rc})
  endif()

  add_library(delta_${name} STATIC ${_srcs})
  add_library(delta::${name} ALIAS delta_${name})
  target_link_libraries(delta_${name} PUBLIC delta_deps ${MOD_DEPS})

  set_property(GLOBAL APPEND PROPERTY DELTA_MODULES delta_${name})

  if(DELTA_BUILD_TESTS AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
  endif()
endfunction()

# add_delta_test(<name> SOURCES <files...> [DEPS <libs...>])
#
# Builds a GoogleTest executable and registers it with CTest. The module under
# test is passed via DEPS.
function(add_delta_test name)
  cmake_parse_arguments(T "" "" "SOURCES;DEPS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE GTest::gtest_main ${T_DEPS})
  add_test(NAME ${name} COMMAND ${name})
endfunction()
