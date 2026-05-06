set(HOST_SOURCES main.cpp)

# Windows pulls in the version resource and the app icon.
if(WIN32)
  list(APPEND HOST_SOURCES _res/host.rc)
endif()

add_executable(ps4delta ${HOST_SOURCES})

# host.rc includes resource.h out of _res.
target_include_directories(ps4delta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/_res)

target_link_libraries(ps4delta PRIVATE delta_core)

# On linux the .ico is useless, so install the converted png plus a desktop
# entry that points at it.
if(UNIX AND NOT APPLE)
  install(TARGETS ps4delta RUNTIME DESTINATION bin)
  install(FILES _res/logo.png
          DESTINATION share/icons/hicolor/256x256/apps
          RENAME ps4delta.png)
  install(FILES _res/ps4delta.desktop
          DESTINATION share/applications)
endif()
