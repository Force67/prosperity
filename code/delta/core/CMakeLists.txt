file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

# UI got nuked; nothing to filter for now. Drop any stale generated bits if
# they ever sneak back in.
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/ui/.*")

# Windows version resource. The glob only picks up cpp/h, so add it here.
if(WIN32)
  list(APPEND CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/_res/core.rc)
endif()

add_library(delta_core STATIC ${CORE_SOURCES})

target_include_directories(delta_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(delta_core PUBLIC
  shared
  capstone_iface
  xbyak
  zlib_iface
  fmt::fmt
  eq_base
)

if(UNIX)
  target_link_libraries(delta_core PUBLIC pthread dl)
endif()
