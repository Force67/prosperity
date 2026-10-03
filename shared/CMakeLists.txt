file(GLOB_RECURSE SHARED_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

if(NOT UNIX)
  list(FILTER SHARED_SOURCES EXCLUDE REGEX ".*_posix\\.(cc|h)$")
endif()

add_library(shared STATIC ${SHARED_SOURCES})

# One include root, so every header has exactly one spelling ("io/file.h").
target_include_directories(shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shared PUBLIC eq_base)
if(DELTA_MEMORY_DEBUG)
  target_compile_definitions(shared PUBLIC DELTA_MEMORY_DEBUG=1)
endif()
if(DELTA_TRACY)
  target_link_libraries(shared PUBLIC tracy_client)
endif()
