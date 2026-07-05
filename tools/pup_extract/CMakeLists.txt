# Firmware PUP unpacker: dumps a decrypted PS4/PS5 update's container segments
# through the native pupReader (mirror of the Android pupExtract JNI path).
add_executable(pup_extract pup_extract.cpp)
target_link_libraries(pup_extract PRIVATE delta_deps delta_formats)
