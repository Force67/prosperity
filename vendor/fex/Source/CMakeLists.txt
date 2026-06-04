# Disable strict aliasing for all build modes
# See discussion in https://github.com/FEX-Emu/FEX/pull/4494#issuecomment-2800608944
# for background context.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing>)

add_subdirectory(Common/)
if (BUILD_FEX_TOOLS)
  add_subdirectory(Tools/)
endif()

if (MINGW)
  add_subdirectory(Windows/)
endif()
