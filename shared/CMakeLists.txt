file(GLOB_RECURSE SHARED_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

# Pick platform-specific impls explicitly.
list(FILTER SHARED_SOURCES EXCLUDE REGEX ".*_win\\.(cpp|h)$")
list(FILTER SHARED_SOURCES EXCLUDE REGEX ".*logimp_win32\\.cpp$")
list(FILTER SHARED_SOURCES EXCLUDE REGEX ".*mem_win\\.cpp$")
if(NOT UNIX)
  list(FILTER SHARED_SOURCES EXCLUDE REGEX ".*_posix\\.(cpp|h)$")
endif()

add_library(shared STATIC ${SHARED_SOURCES})

target_include_directories(shared PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/utl
)

target_link_libraries(shared PUBLIC
  fmt::fmt
  eq_base
)
