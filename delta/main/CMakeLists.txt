# The main module: process bootstrap (dapi), the deltaCore orchestrator
# (dcore) and the executable entry point (main.cpp). Links every other module
# into the final ps4delta binary.
file(GLOB MAIN_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Windows version resources + app icon.
if(WIN32)
  list(APPEND MAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/_res/host.rc
    ${CMAKE_CURRENT_SOURCE_DIR}/_res/core.rc)
endif()

add_executable(ps4delta ${MAIN_SOURCES})

# host.rc / dapi resource defines include resource.h out of _res.
target_include_directories(ps4delta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/_res)

target_link_libraries(ps4delta PRIVATE
  delta_deps
  delta_kern
  delta_runtime
  delta_formats
  delta_crypto
  delta_gfx
  delta_cpu
)

# dapi.cpp uses xbyak_util for the CPU feature probe only on the native x86 path.
if(DELTA_BACKEND STREQUAL "NATIVE")
  target_link_libraries(ps4delta PRIVATE xbyak)
endif()

# On linux the .ico is useless, so install the converted png plus a desktop
# entry that points at it.
if(UNIX AND NOT APPLE)
  install(TARGETS ps4delta RUNTIME DESTINATION bin)
  install(FILES _res/logo.png
          DESTINATION share/icons/hicolor/256x256/apps
          RENAME ps4delta.png)
  install(FILES _res/ps4delta.desktop
          DESTINATION share/applications)
  # Register the application/x-ps4-pkg MIME type so .pkg files open with us.
  install(FILES _res/ps4delta-pkg.xml
          DESTINATION share/mime/packages)
endif()
