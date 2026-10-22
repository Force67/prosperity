set(SRCS
  rpmalloc/rpmalloc.c)

add_library(rpmalloc STATIC ${SRCS})
target_include_directories(rpmalloc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
target_compile_definitions(rpmalloc PRIVATE ENABLE_OVERRIDE=0 ENABLE_DYNAMIC_LINK=0 ENABLE_STATISTICS=0 ENABLE_DECOMMIT=1 ENABLE_ASSERTS=0)
if (MINGW)
  target_compile_definitions(rpmalloc PRIVATE DYNAMIC_TLS_NO_FLS=1)
endif()

set_target_properties(rpmalloc PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(rpmalloc PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(rpmalloc PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
