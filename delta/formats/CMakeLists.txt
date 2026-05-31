# PS4 container format parsers (PUP, SLB2, PKG). Uses zlib for decompression and
# mbedcrypto (AES/RSA/SHA) for the fake-pkg PFS decryption.
add_delta_module(formats DEPS zlib_iface mbedtls_iface)
