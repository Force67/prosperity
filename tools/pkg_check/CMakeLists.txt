# Fake-pkg dump/cross-check tool: lists files and rebuilds eboot.bin -> ELF via
# the native PkgFilesystem (mirror of tools/pkg_extract.py).
add_executable(pkg_check pkg_check.cpp)
target_link_libraries(pkg_check PRIVATE delta_deps delta_formats delta_crypto)
