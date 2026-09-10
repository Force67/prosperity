# Dynamic runtime: x86 code lifting (capstone + xbyak) and the vprx HLE module
# loader. Co-dependent with kern (cycle declared in delta/CMakeLists.txt).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(AVCODEC QUIET IMPORTED_TARGET libavcodec>=60 libavutil)
endif()
add_delta_module(runtime DEPS xbyak capstone_iface delta_gfx)
if(AVCODEC_FOUND)
  target_compile_definitions(delta_runtime PUBLIC DELTA_HAVE_AVCODEC=1)
  target_link_libraries(delta_runtime PUBLIC PkgConfig::AVCODEC)
endif()
