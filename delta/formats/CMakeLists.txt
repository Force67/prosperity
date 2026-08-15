# PS4 container format parsers (PUP, SLB2, PKG). Uses zlib for decompression,
# mbedcrypto (AES/RSA/SHA) for the fake-pkg PFS decryption and unrar for the
# .rar archive backend.
add_delta_module(formats DEPS zlib_iface mbedtls_iface unrar_iface)
