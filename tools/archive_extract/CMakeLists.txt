# Container (.rar/.zip) lister/extractor: walks a game still inside its
# distribution archive through the native ArchiveFilesystem reader.
add_executable(archive_extract archive_extract.cpp)
target_link_libraries(archive_extract PRIVATE delta_deps delta_formats)
