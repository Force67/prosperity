set(DELTA_PACKAGE_VERSION "0.0.0" CACHE STRING "Linux package version")

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME prosperity)
set(CPACK_PACKAGE_VERSION "${DELTA_PACKAGE_VERSION}")
set(CPACK_PACKAGE_CONTACT "Force67 <https://github.com/Force67/prosperity>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Prosperity PS4 and PS5 emulator")
set(CPACK_PACKAGING_INSTALL_PREFIX /usr)
set(CPACK_STRIP_FILES ON)
set(CPACK_COMPONENTS_ALL Runtime)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME prosperity)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SECTION games)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/Force67/prosperity")
# SDL loads these desktop libraries at runtime, so shlibdeps cannot detect them.
set(CPACK_DEBIAN_PACKAGE_DEPENDS
  "libasound2t64, libpulse0, libudev1, libdbus-1-3, libx11-6, libxext6, libxrandr2, libxcursor1, libxfixes3, libxi6, libxss1, libxtst6, libxkbcommon0, libwayland-client0, libwayland-cursor0, libwayland-egl1, libdecor-0-0")

install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
        DESTINATION share/doc/prosperity COMPONENT Runtime)
include(CPack)
