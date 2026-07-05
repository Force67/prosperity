# UFS2 (*.ffpkg) lister/extractor: walks a PS5 game backup through the native
# Ufs2Filesystem reader.
add_executable(ffpkg_extract ffpkg_extract.cpp)
target_link_libraries(ffpkg_extract PRIVATE delta_deps delta_formats)
