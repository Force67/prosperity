cmake_minimum_required(VERSION 3.14)
set (PROJECT_NAME FEX_jemalloc_glibc)
project(${PROJECT_NAME}
  VERSION 5.3.0
  LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
cmake_policy(SET CMP0083 NEW) # Follow new PIE policy
include(CheckPIESupported)
check_pie_supported()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pregen/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

include(CheckCSourceCompiles)
check_c_source_compiles(
  "
  #include <malloc.h>
  #include <stddef.h>
  size_t malloc_usable_size(const void *ptr);
  int main() { return 0; }
  "
  CMAKE_JEMALLOC_USABLE_SIZE_CONST)

if (CMAKE_JEMALLOC_USABLE_SIZE_CONST)
  set(JEMALLOC_USABLE_SIZE_CONST const)
else()
  # Set to TRUE so it is defined to nothing.
  set(JEMALLOC_USABLE_SIZE_CONST TRUE)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/jemalloc/jemalloc_defs.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/jemalloc/jemalloc_defs.h)

set (SRCS
src/jemalloc.c
src/arena.c
src/background_thread.c
src/base.c
src/bin.c
src/bitmap.c
src/ckh.c
src/ctl.c
src/div.c
src/extent.c
src/extent_dss.c
src/extent_mmap.c
src/hook.c
src/large.c
src/log.c
src/malloc_io.c
src/mutex.c
src/nstime.c
src/pages.c
src/prof.c
src/rtree.c
src/safety_check.c
src/stats.c
src/sc.c
src/sz.c
src/tcache.c
src/test_hooks.c
src/ticker.c
src/tsd.c
src/witness.c
src/pa.c
src/pac.c
src/eset.c
src/edata.c
src/sec.c
src/decay.c
src/hpa_hooks.c
src/pa_extra.c
src/ehooks.c
src/inspect.c
src/emap.c
src/san.c
src/prof_data.c
src/counter.c
src/cache_bin.c
src/thread_event.c
src/fxp.c
src/ecache.c
src/edata_cache.c
src/pai.c
src/exp_grow.c
src/hpa.c
src/hpdata.c
src/psset.c
src/peak_event.c
src/buf_writer.c
src/prof_recent.c
src/san_bump.c
src/bin_info.c
)

function(AddLibrary Name Type)
  add_library(${Name} ${Type} ${SRCS})
  set_property(TARGET ${Name} PROPERTY C_STANDARD 11)
  set_property(TARGET ${Name} PROPERTY CXX_STANDARD 17)

  target_link_libraries(${Name} pthread)
  set_target_properties(${Name} PROPERTIES OUTPUT_NAME FEX_jemalloc_glibc)

  target_include_directories(${Name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
  target_include_directories(${Name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pregen/include/)
  target_include_directories(${Name} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include/)

  target_compile_definitions(${Name} PRIVATE -D_GNU_SOURCE -D_REENTRANT)

  target_compile_options(${Name} PRIVATE -fvisibility=hidden)
endfunction()

AddLibrary(${PROJECT_NAME} STATIC)
