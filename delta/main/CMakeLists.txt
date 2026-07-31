# The main module: process bootstrap (dapi), the deltaCore orchestrator
# (dcore) and the executable entry point (main.cpp). Links every other module
# into the final ps4delta binary. android_main.cpp is the app-only entry.
file(GLOB MAIN_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# android_main.cpp / android_jni.cpp are app-only (NativeActivity + JNI bridge).
list(REMOVE_ITEM MAIN_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/android_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/android_jni.cpp)

set(MAIN_LIBS delta_deps delta_kern delta_runtime delta_formats delta_crypto
              delta_gfx delta_cpu)

if(DELTA_ANDROID_APP)
  # NativeActivity shared lib loaded by android.app.NativeActivity. android_main
  # (via native_app_glue) replaces main.cpp/dapi.cpp; dcore drives the boot.
  set(_glue ${ANDROID_NDK}/sources/android/native_app_glue)
  add_library(ps4delta_app SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/android_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/android_jni.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dcore.cpp
    ${_glue}/android_native_app_glue.c)
  target_include_directories(ps4delta_app PRIVATE ${_glue}
    ${CMAKE_CURRENT_SOURCE_DIR}/_res)
  target_link_libraries(ps4delta_app PRIVATE ${MAIN_LIBS} android log)
  # native_app_glue exports the activity entry (pulled from an archive, so it
  # needs -u). The NativeBridge JNI exports are JNIEXPORT (default visibility)
  # and compiled straight into the .so, so they survive --gc-sections as
  # dynamic-symbol roots without an explicit -u.
  target_link_options(ps4delta_app PRIVATE -u ANativeActivity_onCreate)
  return()
endif()

# Windows version resources + app icon.
if(WIN32)
  list(APPEND MAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/_res/host.rc
    ${CMAKE_CURRENT_SOURCE_DIR}/_res/core.rc)
endif()

add_executable(ps4delta ${MAIN_SOURCES})

# host.rc / dapi resource defines include resource.h out of _res.
target_include_directories(ps4delta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/_res)

target_link_libraries(ps4delta PRIVATE ${MAIN_LIBS})

# dapi.cpp uses xbyak_util for the CPU feature probe only on the native x86 path.
if(DELTA_BACKEND STREQUAL "NATIVE")
  target_link_libraries(ps4delta PRIVATE xbyak)
endif()

# The per-title settings are looked up next to the binary, so link the source
# tree's copy there: editing a profile then takes effect without a rebuild.
if(UNIX)
  add_custom_command(TARGET ps4delta POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_SOURCE_DIR}/game_profiles
            $<TARGET_FILE_DIR:ps4delta>/game_profiles)
else()
  add_custom_command(TARGET ps4delta POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/game_profiles
            $<TARGET_FILE_DIR:ps4delta>/game_profiles)
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
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/game_profiles
          DESTINATION bin)
  # Register the application/x-ps4-pkg MIME type so .pkg files open with us.
  install(FILES _res/ps4delta-pkg.xml
          DESTINATION share/mime/packages)
endif()
